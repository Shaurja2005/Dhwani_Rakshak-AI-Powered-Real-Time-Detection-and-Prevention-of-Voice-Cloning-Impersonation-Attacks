"""Head B analysis pipeline: one window -> features + interpretable detector outputs.

Shared by the live head and the training script so both see identical features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.vg_models.heads.head_b_dsp import codec_id, features, scene, vocoder


@dataclass
class Analysis:
    features: dict[str, float]
    scene: scene.SceneConsistency
    vocoder: vocoder.VocoderFingerprint
    codec: codec_id.CodecEstimate


def analyse(pcm: np.ndarray, original_sample_rate: int | None = None) -> Analysis:
    s = features.compute_spectra(pcm)
    sc = scene.analyse(s)
    voc = vocoder.fingerprint(s)
    cod = codec_id.estimate(s, original_sample_rate)
    feats = features.extract(pcm, s)
    feats.update(sc.as_features())
    feats.update(voc.as_features())
    feats.update(cod.as_features())
    return Analysis(feats, sc, voc, cod)


def feature_names() -> list[str]:
    """Stable feature order (derived from a silent-ish probe window)."""
    rng = np.random.default_rng(0)
    return list(analyse((0.01 * rng.standard_normal(48000)).astype(np.float32), 16000).features)
