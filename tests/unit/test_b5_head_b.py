"""B5 — Head B (DSP artifacts + scene consistency) unit tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

from ml.data.channel.base import ChannelRecord, resample
from ml.data.channel.codecs import G711, FFmpegCodec, codec_available
from ml.data.channel.rir import synthetic_rir
from ml.training import train_head_b
from packages.vg_core.models import AbstainReason, AnalysisWindow, SessionContext
from packages.vg_core.sample_store import put_samples
from packages.vg_models.heads.head_b_dsp import codec_id, features, scene, vocoder
from packages.vg_models.heads.head_b_dsp.gbdt import GBDT
from packages.vg_models.heads.head_b_dsp.head import HeadB
from packages.vg_models.heads.head_b_dsp.pipeline import analyse, feature_names

SR = 16000
SID = "01900b1a-0000-7000-8000-00000000b5b1"
CTX = SessionContext(session_id=SID, tenant_id="t")


# ---------------------------------------------------------------- synthetic scenes
def _voice(
    rng: np.random.Generator, starts: tuple[float, ...] = (0.1, 1.3, 2.5), n: int = 48000
) -> np.ndarray:
    x, t = np.zeros(n), np.arange(n) / SR
    for st in starts:
        a, b = int(st * SR), int((st + 0.25) * SR)
        f0 = rng.uniform(110, 180)
        tt = t[a:b] - t[a]
        x[a:b] = (
            sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, int(7500 / f0)))
            * np.hanning(b - a) ** 0.2
        )
    return 0.5 * x / np.abs(x).max()


def _clicks(
    rng: np.random.Generator, starts: tuple[float, ...] = (0.75, 1.95), n: int = 48000
) -> np.ndarray:
    x = np.zeros(n)
    for st in starts:
        a = int(st * SR)
        x[a : a + 320] = rng.standard_normal(320) * np.exp(-np.arange(320) / 60)
    return 0.4 * x / np.abs(x).max()


def _scene(seed: int, composite: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = synthetic_rir(SR, 0.4, rng)

    def rev(z: np.ndarray) -> np.ndarray:
        return fftconvolve(z, h)[: len(z)]

    v = _voice(rng) if composite else rev(_voice(rng))
    return (v + rev(_clicks(rng)) + 0.001 * rng.standard_normal(48000)).astype(np.float32)


def _speechlike(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = _voice(rng, starts=(0.05, 0.5, 0.95, 1.4, 1.85, 2.3))
    return (v + 0.01 * rng.standard_normal(48000)).astype(np.float32)


# ---------------------------------------------------------------- features (T01)
def test_features_are_finite_named_and_stable() -> None:
    feats = features.extract(_speechlike())
    names = feature_names()
    assert len(names) > 90 and names[: len(feats)] == list(feats)
    finite = {k: v for k, v in feats.items() if k != "breath_low_band_ratio"}
    assert all(np.isfinite(v) for v in finite.values())


def test_head_b_analysis_fits_cpu_budget() -> None:
    x = _speechlike()
    analyse(x, SR)
    t0 = time.perf_counter()
    for _ in range(10):
        analyse(x, SR)
    per_window_ms = (time.perf_counter() - t0) / 10 * 1000
    # Plan target is 15 ms; allow slack for loaded CI runners.
    assert per_window_ms < 40, per_window_ms


# ---------------------------------------------------------------- vocoder fingerprints (T02)
def test_upsampling_tones_detected_and_not_on_clean_voice() -> None:
    x = _speechlike()
    t = np.arange(len(x)) / SR
    tones = x + sum(0.01 * np.sin(2 * np.pi * f * t) for f in (2000, 4000, 6000))
    clean = vocoder.fingerprint(features.compute_spectra(x))
    art = vocoder.fingerprint(features.compute_spectra(tones.astype(np.float32)))
    assert clean.upsampling_tones == 0.0
    assert art.upsampling_tones > 0.5 and any(abs(f - 4000) < 50 for f in art.tone_freqs_hz)


def test_brick_wall_band_edge_detected() -> None:
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(48000)
    spec = np.fft.rfft(noise)
    spec[np.fft.rfftfreq(48000, 1 / SR) > 5000] = 0
    brick = (0.1 * np.fft.irfft(spec, 48000)).astype(np.float32)
    soft = (0.1 * fftconvolve(noise, np.hanning(9) / 4.5, mode="same")).astype(np.float32)
    assert vocoder.fingerprint(features.compute_spectra(brick)).band_edge_sharpness > 0.5
    assert vocoder.fingerprint(features.compute_spectra(soft)).band_edge_sharpness < 0.5


# ---------------------------------------------------------------- scene consistency (T04)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dry_voice_over_reverberant_background_is_flagged(seed: int) -> None:
    res = scene.analyse(features.compute_spectra(_scene(seed, composite=True)))
    assert res.known and res.inconsistency > 0.8
    assert res.rt60_background_s is not None and 0.3 < res.rt60_background_s < 0.6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_room_is_not_flagged(seed: int) -> None:
    res = scene.analyse(features.compute_spectra(_scene(seed, composite=False)))
    assert res.inconsistency == 0.0


def test_scene_check_survives_g711() -> None:
    x = _scene(0, composite=True)
    y, sr = G711().apply(x, SR, np.random.default_rng(0), ChannelRecord())
    res = scene.analyse(features.compute_spectra(resample(y, sr, SR)))
    assert res.inconsistency > 0.8


def test_scene_unknown_without_enough_events() -> None:
    res = scene.analyse(features.compute_spectra(_speechlike()))
    assert not res.known and res.inconsistency == 0.0


# ---------------------------------------------------------------- codec identifier (T05)
def test_codec_identifier_wideband_vs_narrowband() -> None:
    x = _speechlike()
    wide = codec_id.estimate(features.compute_spectra(x), 16000)
    y, sr = G711().apply(x, SR, np.random.default_rng(0), ChannelRecord())
    narrow = codec_id.estimate(features.compute_spectra(resample(y, sr, SR)), 8000)
    assert wide.label.startswith("wideband") and wide.bandwidth_hz > 6000
    assert narrow.label.startswith("narrowband") and narrow.bandwidth_hz < 4600
    assert "kHz" in narrow.describe()


@pytest.mark.skipif(not codec_available("opus"), reason="ffmpeg libopus not available")
def test_codec_identifier_flags_low_bitrate_opus_as_compressed() -> None:
    x = _speechlike()
    y, _ = FFmpegCodec("opus", 8).apply(x, SR, np.random.default_rng(0), ChannelRecord())
    assert codec_id.estimate(features.compute_spectra(y), 16000).label == "wideband_compressed"


# ---------------------------------------------------------------- GBDT (T03)
def test_gbdt_learns_nonlinear_rule_and_roundtrips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(2000, 4))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(float)  # XOR: needs depth >= 2
    x[rng.random(x.shape) < 0.02] = np.nan
    m = GBDT.fit(x[:1500], y[:1500], ["a", "b", "c", "d"], n_trees=60, depth=3, lr=0.3)
    acc = float(((m.decision(x[1500:]) > 0) == y[1500:]).mean())
    assert acc > 0.9
    assert set(list(m.feature_importance())[:2]) == {"a", "b"}
    m.save(tmp_path / "m.json", {"k": 1})
    m2, meta = GBDT.load(tmp_path / "m.json")
    assert meta == {"k": 1} and np.allclose(m.decision(x[1500:]), m2.decision(x[1500:]))


def test_gbdt_prediction_is_fast() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 100))
    m = GBDT.fit(
        x, (x[:, 0] > 0).astype(float), [f"f{i}" for i in range(100)], n_trees=200, depth=4
    )
    row = x[:1]
    m.decision(row)
    t0 = time.perf_counter()
    for _ in range(50):
        m.decision(row)
    assert (time.perf_counter() - t0) / 50 * 1000 < 5


# ---------------------------------------------------------------- head + explanations (T06)
def _window(
    pcm: np.ndarray | None, wid: int, voiced_ms: int = 2500, quality_ok: bool = True
) -> AnalysisWindow:
    ref = f"shm://{SID}/b{wid}"
    if pcm is not None:
        put_samples(ref, pcm)
    return AnalysisWindow(
        session_id=SID,
        window_id=wid,
        start_ms=0,
        end_ms=3000,
        samples_ref=ref,
        voiced_ms=voiced_ms,
        snr_db=20.0,
        clipping_ratio=0.0,
        quality_ok=quality_ok,
        original_sample_rate=16000,
    )


def test_untrained_head_abstains_with_plain_english_evidence() -> None:
    h = HeadB()
    h.warmup()
    s = h.score(_window(_scene(0, composite=True), 0), CTX)
    assert s.abstain and s.abstain_reason == AbstainReason.UNTRAINED
    reasons = s.evidence["reasons"]
    assert (
        1 <= len(reasons) <= 3
        and any("echo" in r for r in reasons)
        and reasons[-1].startswith("Channel:")
    )
    assert s.evidence["scene"]["inconsistency"] > 0.8


def test_head_b_abstains_on_quality_and_missing_audio() -> None:
    h = HeadB()
    assert (
        h.score(_window(_speechlike(), 1, quality_ok=False), CTX).abstain_reason
        == AbstainReason.QUALITY_GATE
    )
    assert (
        h.score(_window(_speechlike(), 2, voiced_ms=100), CTX).abstain_reason
        == AbstainReason.INSUFFICIENT_SPEECH
    )
    assert h.score(_window(None, 3), CTX).abstain_reason == AbstainReason.INSUFFICIENT_SPEECH


def test_head_b_never_raises() -> None:
    h = HeadB()
    h.warmup()
    s = h.score(_window(np.full(48000, np.nan, dtype=np.float32), 4), CTX)
    assert s.abstain or s.p_spoof is not None


def test_training_smoke_then_trained_head_scores(tmp_path: Path) -> None:
    cfg = {
        "run_name": "hb",
        "lineage": "research",
        "allow_noncommercial": True,
        "seed": 3,
        "out_dir": str(tmp_path),
        "data": {"source": "synthetic", "n_train": 40, "n_dev": 20},
        "gbdt": {"n_trees": 20, "depth": 2, "min_leaf": 3},
    }
    result = train_head_b.train(cfg)
    h = HeadB(model_path=result["model"], budget_ms=1000)
    h.warmup()
    s = h.score(_window(_speechlike(), 5), CTX)
    assert not s.abstain and 0 <= (s.p_spoof or 0) <= 1
    assert s.model_version == "B@dsp-gbdt-v0.1.0" and s.evidence["reasons"]
