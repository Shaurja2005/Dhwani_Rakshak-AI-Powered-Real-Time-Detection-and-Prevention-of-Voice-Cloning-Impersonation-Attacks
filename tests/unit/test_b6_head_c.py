"""B6 — Head C (prosody) unit tests on controlled synthetic speech-like signals."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.training import train_head_c
from packages.vg_core.models import (
    AbstainReason,
    AnalysisWindow,
    CallMetadata,
    Channel,
    ConsentBasis,
    SessionContext,
)
from packages.vg_core.sample_store import put_samples
from packages.vg_models.heads.head_c_prosody import breath_rhythm as br
from packages.vg_models.heads.head_c_prosody import pitch
from packages.vg_models.heads.head_c_prosody.disfluency import Token, disfluency_features
from packages.vg_models.heads.head_c_prosody.head import HeadC
from packages.vg_models.heads.head_c_prosody.model import (
    FRAME_FEATURES,
    ProsodyNet,
    analyse,
    load,
    save,
)
from packages.vg_models.heads.head_c_prosody.normalize import (
    LanguageNorm,
    min_pause_ms,
    per_language_report,
)

SR = 16000
SID = "01900b1a-0000-7000-8000-00000000c6c1"


def _tone(
    rng: np.random.Generator, f0: float, dur: float, jitter: float = 0.0, decl: float = 0.0
) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * 2 ** (decl * t / 12)
    if jitter:
        # Piecewise-constant per 10 ms: the tracker measures frame-level period variation.
        f = f * (1 + jitter * np.repeat(rng.standard_normal(n // 160 + 1), 160)[:n])
    ph = 2 * np.pi * np.cumsum(f) / SR
    return 0.3 * sum(np.sin(k * ph) / k for k in range(1, 10))


def _utterance(
    rng: np.random.Generator, gaps: list[float], breaths: bool = False, seconds: float = 6.0
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for g in gaps:
        if breaths:
            b = int(0.3 * SR)
            parts += [
                np.zeros(int(max(g - 0.35, 0.05) * SR)),
                0.01 * rng.standard_normal(b) * np.hanning(b),
                np.zeros(int(0.05 * SR)),
            ]
        else:
            parts.append(np.zeros(int(g * SR)))
        parts.append(_tone(rng, rng.uniform(120, 180), rng.uniform(0.25, 0.6)))
    x = np.concatenate(parts)
    x = np.pad(x, (0, max(0, int(seconds * SR) - len(x))))[: int(seconds * SR)]
    return (x + 0.0005 * rng.standard_normal(len(x))).astype(np.float32)


# ---------------------------------------------------------------- pitch (T01)
@pytest.mark.parametrize("f0", [85.0, 150.0, 240.0, 350.0])
def test_pitch_tracker_accuracy(f0: float) -> None:
    p = pitch.track(_tone(np.random.default_rng(0), f0, 1.0))
    assert p.voiced.mean() > 0.9
    assert abs(np.median(p.f0_hz[p.voiced]) - f0) / f0 < 0.02


def test_unvoiced_noise_is_not_pitched() -> None:
    p = pitch.track(0.1 * np.random.default_rng(0).standard_normal(SR))
    assert p.voiced.mean() < 0.1


def test_declination_jitter_and_range() -> None:
    rng = np.random.default_rng(1)
    pad = np.zeros(1600)
    falling = pitch.contour_features(
        pitch.track(np.concatenate([pad, _tone(rng, 160, 0.8, decl=-4), pad]))
    )
    assert -5 < falling["f0_declination_st_per_s"] < -3
    steady = pitch.contour_features(pitch.track(_tone(rng, 150, 1.0)))
    shaky = pitch.contour_features(pitch.track(_tone(rng, 150, 1.0, jitter=0.02)))
    assert shaky["jitter_rel"] > 3 * steady["jitter_rel"]
    assert steady["f0_range_st"] < 0.5


# ---------------------------------------------------------------- breath + rhythm (T02/T03)
def test_breaths_before_phrases_are_detected() -> None:
    rng = np.random.default_rng(2)
    with_b = analyse(_utterance(rng, [0.5, 0.7, 0.6, 0.8, 0.5], breaths=True), "en").features
    without = analyse(_utterance(rng, [0.5, 0.7, 0.6, 0.8, 0.5]), "en").features
    assert with_b["breath_count"] >= 3 and with_b["breath_to_phrase_ratio"] > 0.5
    assert without["breath_count"] == 0


def test_metronomic_rhythm_scores_more_regular_than_natural() -> None:
    rng = np.random.default_rng(3)
    regular = analyse(_utterance(rng, [0.3] * 8), "en").features
    natural = analyse(_utterance(rng, [0.1, 0.6, 0.25, 0.9, 0.2, 0.45, 0.15, 0.7]), "en").features
    assert regular["pause_cv"] < 0.1 < natural["pause_cv"]
    assert regular["over_regularity"] > natural["over_regularity"] + 0.3


def test_indic_closures_are_not_counted_as_pauses() -> None:
    rng = np.random.default_rng(4)
    x = _utterance(rng, [0.18] * 8)  # 180 ms gaps ~ geminate closures
    assert min_pause_ms("hi") > 180 > min_pause_ms("en")
    assert analyse(x, "en").features["pause_count"] >= 4
    assert analyse(x, "hi-IN").features["pause_count"] == 0


def test_filled_pause_detected() -> None:
    rng = np.random.default_rng(5)
    uhh = np.concatenate([np.zeros(4000), _tone(rng, 130, 0.6), np.zeros(4000)])
    assert br.filled_pause_features(pitch.track(uhh))["filled_pause_count"] == 1


# ---------------------------------------------------------------- disfluency (T04)
def test_disfluencies_from_asr_tokens_english_and_hindi() -> None:
    en = [
        Token(w, i * 400, i * 400 + 300)
        for i, w in enumerate("so um I I need uh I went I go to the bank".split())
    ]
    f = disfluency_features(en, "en")
    minute_scale = 60000 / (en[-1].end_ms - en[0].start_ms)
    assert f["filler_per_min"] == pytest.approx(2 * minute_scale)
    assert f["repetition_per_min"] == pytest.approx(1 * minute_scale)
    assert f["restart_per_min"] >= minute_scale - 1e-9
    hi = [
        Token("मतलब", 0, 300),
        Token("वो", 400, 600),
        Token("पैसे", 700, 1000),
        Token("भेजो", 1100, 1500),
    ]
    assert disfluency_features(hi, "hi")["filler_per_min"] > 0
    assert np.isnan(disfluency_features(None, "hi")["filler_per_min"])


# ---------------------------------------------------------------- Indic normalisation (T06)
def test_language_norm_uses_own_language_and_falls_back() -> None:
    rows = [{"pause_cv": 0.5 + 0.01 * i} for i in range(40)] + [
        {"pause_cv": 0.2 + 0.01 * i} for i in range(40)
    ]
    langs = ["ta"] * 40 + ["en"] * 40
    norm = LanguageNorm.fit(rows, langs)
    z_ta, src_ta = norm.apply({"pause_cv": 0.695}, "ta-IN")
    z_other, src_other = norm.apply({"pause_cv": 0.695}, "kn")
    assert src_ta == "ta" and abs(z_ta["pause_cv"]) < 0.1
    assert src_other == "global" and z_other["pause_cv"] > 0.5
    assert LanguageNorm.from_json(norm.to_json()).per_language.keys() == norm.per_language.keys()


def test_per_language_report_flags_fpr_gap() -> None:
    langs = ["hi"] * 50 + ["en"] * 50
    is_spoof = np.zeros(100)
    p = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])  # every Hindi bona fide flagged
    report = per_language_report(p, is_spoof, langs)
    assert report["languages"]["hi"]["fpr"] == 1.0 and not report["fairness_ok"]


# ---------------------------------------------------------------- temporal model (T05)
def test_prosody_net_shapes_and_roundtrip(tmp_path: Path) -> None:
    a = analyse(_utterance(np.random.default_rng(6), [0.3, 0.5, 0.4]), "hi")
    names = list(a.features)
    net = ProsodyNet(len(names)).eval()
    frames = torch.from_numpy(a.frames)[None]
    assert a.frames.shape[1] == len(FRAME_FEATURES)
    logit = net(frames, torch.zeros(1, len(names)))
    save(net, names, LanguageNorm(), tmp_path / "c.pt", {"lineage": "research"})
    net2, names2, _, meta = load(tmp_path / "c.pt")
    assert names2 == names and meta["lineage"] == "research"
    assert torch.allclose(logit, net2(frames, torch.zeros(1, len(names))))


def test_analysis_within_cpu_budget() -> None:
    x = _utterance(np.random.default_rng(7), [0.3] * 6, seconds=3.0)
    analyse(x)
    t0 = time.perf_counter()
    for _ in range(10):
        analyse(x)
    assert (time.perf_counter() - t0) / 10 * 1000 < 60  # target 25 ms; slack for CI


# ---------------------------------------------------------------- head
def _ctx(language: str | None = "ta-IN") -> SessionContext:
    meta = CallMetadata(
        session_id=SID,
        tenant_id="t",
        direction="inbound",
        started_at=datetime.now(tz=UTC),
        channel=Channel.FILE,
        codec_hint="pcm",
        source_sample_rate=16000,
        consent_basis=ConsentBasis.LEGITIMATE_USE,
        language_hint=language,
    )
    return SessionContext(session_id=SID, tenant_id="t", call_metadata=meta)


def _window(pcm: np.ndarray | None, wid: int, voiced_ms: int = 2500) -> AnalysisWindow:
    ref = f"shm://{SID}/c{wid}"
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
        quality_ok=True,
        original_sample_rate=8000,
    )


def test_head_c_abstains_below_two_seconds_and_when_untrained() -> None:
    h = HeadC()
    x = _utterance(np.random.default_rng(8), [0.2, 0.4, 0.3, 0.5], seconds=3.0)
    assert (
        h.score(_window(x, 0, voiced_ms=1800), _ctx()).abstain_reason
        == AbstainReason.INSUFFICIENT_SPEECH
    )
    s = h.score(_window(x, 1), _ctx())
    assert s.abstain and s.abstain_reason == AbstainReason.UNTRAINED
    assert s.evidence["language"] == "ta-IN" and "f0_range_st" in s.evidence["prosody"]
    assert s.model_version == "C@prosody-untrained-v0.0.0"


def test_head_c_never_raises() -> None:
    h = HeadC()
    assert h.score(_window(np.full(48000, np.nan, dtype=np.float32), 2), _ctx(None)).abstain


def test_training_smoke_then_trained_head_scores(tmp_path: Path) -> None:
    cfg = {
        "run_name": "hc",
        "lineage": "research",
        "allow_noncommercial": True,
        "seed": 5,
        "out_dir": str(tmp_path),
        "data": {"source": "synthetic", "n_train": 40, "n_dev": 20},
        "train": {"epochs": 2, "batch_size": 16},
        "fairness": {"enforce": False},
    }
    result = train_head_c.train(cfg)
    h = HeadC(model_path=result["model"], budget_ms=2000)
    h.warmup()
    x = _utterance(np.random.default_rng(9), [0.2, 0.4, 0.3, 0.5], seconds=3.0)
    s = h.score(_window(x, 3), _ctx("hi"))
    assert not s.abstain and 0 <= (s.p_spoof or 0) <= 1
    assert s.evidence["normalised_with"] in ("hi", "global")


def test_training_fairness_gate_can_fail_the_run(tmp_path: Path) -> None:
    cfg = {
        "run_name": "hc2",
        "lineage": "research",
        "allow_noncommercial": True,
        "seed": 5,
        "out_dir": str(tmp_path),
        "data": {"source": "synthetic", "n_train": 20, "n_dev": 10},
        "train": {"epochs": 1},
        "fairness": {"enforce": True, "max_fpr_gap": -1.0},
    }
    with pytest.raises(train_head_c.FairnessGateFailed):
        train_head_c.train(cfg)
