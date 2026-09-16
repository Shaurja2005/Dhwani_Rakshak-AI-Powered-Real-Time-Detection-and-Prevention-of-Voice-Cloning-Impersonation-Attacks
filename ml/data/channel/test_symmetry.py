"""CI gate for invariant I3: augmentation must be symmetric (B3-T08)."""

from __future__ import annotations

import numpy as np

from ml.data.channel.symmetry import check_manifest_symmetry
from ml.data.channel.webrtc_chain import ChannelConfig, ChannelSimulator
from ml.data.manifest import ManifestRow

SR = 16000


def _utterance(rng: np.random.Generator, label: str) -> np.ndarray:
    # Content differs by label (harmonic vs noisy excitation); the channel must not.
    t = np.arange(int(0.5 * SR)) / SR
    f0 = rng.uniform(100, 220)
    if label == "spoof":
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    else:
        x = np.sin(2 * np.pi * f0 * t) + 0.3 * rng.standard_normal(len(t))
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def _rows(sim: ChannelSimulator, n: int, label_seed_leak: bool = False) -> list[ManifestRow]:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        label = "spoof" if i % 2 else "bona_fide"
        utt_id = f"utt{i:05d}"
        x = _utterance(rng, label)
        # The negative control leaks the label into the chain config.
        sim_i = sim
        if label_seed_leak and label == "spoof":
            sim_i = ChannelSimulator(
                ChannelConfig(p_noise=1.0, p_loss=1.0, codec_weights={"g711u": 1})
            )
        _, sr, rec = sim_i.simulate(x, SR, utt_id)
        rows.append(
            ManifestRow(
                utt_id=utt_id,
                path=f"mem://{utt_id}",
                label=label,  # type: ignore[arg-type]
                generator_family="synthetic_test" if label == "spoof" else None,
                language="hi",
                speaker_id=f"spk{i % 40}",
                source_corpus="vg_indic_telephony",
                license="internal",
                commercial_use=False,
                codec_chain=rec.codec_chain,
                snr_db=rec.snr_db,
                rir_id=rec.rir_id,
                duration_s=0.5,
                sample_rate=sr,
            )
        )
    return rows


def _fast_sim(**overrides: object) -> ChannelSimulator:
    # ffmpeg codecs are slow per-utterance; the symmetry property does not depend on them.
    cfg = ChannelConfig(codec_weights={"clean": 1, "g711u": 1, "g711a": 1}, **overrides)  # type: ignore[arg-type]
    return ChannelSimulator(cfg)


def test_channel_features_cannot_separate_classes() -> None:
    res = check_manifest_symmetry(_rows(_fast_sim(), 400))
    assert res.passed, str(res)


def test_symmetry_check_detects_asymmetric_augmentation() -> None:
    """Negative control: the gate must have the power to catch a leak."""
    res = check_manifest_symmetry(_rows(_fast_sim(), 400, label_seed_leak=True))
    assert not res.passed, str(res)
