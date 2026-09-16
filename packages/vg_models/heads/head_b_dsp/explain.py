"""Plain-English explanations for Head B (B5-T06).

Every Head B score ships with 1–3 reasons written for a frontline agent, not a
DSP engineer. Reasons come from interpretable detectors (scene, vocoder,
codec, breath) and, when a trained model is loaded, from the features where
this window deviates most from the bona fide reference statistics stored in
the model file.
"""

from __future__ import annotations

import math

from packages.vg_models.heads.head_b_dsp.codec_id import CodecEstimate
from packages.vg_models.heads.head_b_dsp.scene import SceneConsistency
from packages.vg_models.heads.head_b_dsp.vocoder import VocoderFingerprint

FEATURE_PHRASES = {
    "breath_low_band_ratio": "breathing / low-frequency noise between words",
    "bicoherence": "nonlinear phase coupling across frequencies",
    "ltas_tilt_db_per_oct": "overall spectral balance (tilt)",
    "rolloff95_mean": "high-frequency energy extent",
    "voc_phase_regularity": "regularity of the waveform phase",
}


def reasons(
    scene: SceneConsistency,
    voc: VocoderFingerprint,
    codec: CodecEstimate,
    feats: dict[str, float],
    reference: dict[str, tuple[float, float]] | None = None,
    limit: int = 3,
) -> list[str]:
    """Return up to ``limit`` reasons, most salient first. Always at least one (the channel)."""
    scored: list[tuple[float, str]] = []
    if (
        scene.inconsistency > 0.3
        and scene.rt60_voice_s is not None
        and scene.rt60_background_s is not None
    ):
        scored.append(
            (
                1.0 + scene.inconsistency,
                "Voice sounds recorded in a dry, echo-free space "
                f"(≈{scene.rt60_voice_s:.2f} s decay) but the background has room echo "
                f"(≈{scene.rt60_background_s:.2f} s): the voice and background may have "
                "been mixed together.",
            )
        )
    if voc.upsampling_tones > 0.3:
        hz = ", ".join(f"{f / 1000:.1f}" for f in voc.tone_freqs_hz[:3])
        scored.append(
            (
                0.8 + voc.upsampling_tones,
                f"Steady narrow tones at {hz} kHz that do not follow the voice, "
                "typical of neural vocoders.",
            )
        )
    if voc.band_edge_sharpness > 0.5:
        scored.append(
            (
                0.6 + voc.band_edge_sharpness,
                f"Unnaturally sharp frequency cut-off at {voc.band_edge_hz / 1000:.1f} kHz, "
                "typical of synthetic or neural-codec audio.",
            )
        )
    breath = feats.get("breath_low_band_ratio", float("nan"))
    if not math.isnan(breath) and breath < 0.02 and feats.get("voiced_frame_ratio", 0) > 0.3:
        scored.append((0.5, "Almost no breath or low-frequency noise between words."))
    if reference:
        for name, value in feats.items():
            if name not in reference or math.isnan(value):
                continue
            mu, sd = reference[name]
            z = abs(value - mu) / (sd + 1e-9)
            if z > 3.0:
                phrase = FEATURE_PHRASES.get(name)
                if phrase:
                    scored.append(
                        (
                            0.3 + min(z, 10) / 20,
                            f"Unusual {phrase} compared with genuine speech ({z:.0f}σ).",
                        )
                    )
    scored.sort(key=lambda p: -p[0])
    out = [text for _, text in scored[: limit - 1]]
    out.append(f"Channel: {codec.describe()}.")
    return out[:limit]
