"""Channel / codec identifier (B5-T05).

Infers the likely channel from what survives in the 16 kHz window plus the
source sample rate recorded by the conditioner. Used as (a) features for the
classifier, so it can learn codec-conditional thresholds, and (b) an
operator-facing explanation. It is an estimate, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.vg_models.heads.head_b_dsp.features import _FREQS, EPS, Spectra

LABELS = (
    "wideband_clean",
    "wideband_compressed",
    "narrowband_g711",
    "narrowband_lowrate",
    "unknown",
)


@dataclass
class CodecEstimate:
    label: str
    confidence: float
    bandwidth_hz: float
    spectral_hole_ratio: float

    def as_features(self) -> dict[str, float]:
        feats = {f"codec_is_{name}": float(self.label == name) for name in LABELS}
        feats["codec_bandwidth"] = self.bandwidth_hz / 8000.0
        feats["codec_hole_ratio"] = self.spectral_hole_ratio
        return feats

    def describe(self) -> str:
        pretty = {
            "wideband_clean": "wideband audio with no obvious compression",
            "wideband_compressed": "wideband audio with lossy compression (e.g. Opus/VoIP)",
            "narrowband_g711": "narrowband telephone audio (G.711-like, ~4 kHz)",
            "narrowband_lowrate": "narrowband low-bitrate mobile codec (AMR/GSM-like)",
            "unknown": "unrecognised channel",
        }[self.label]
        return f"{pretty}, bandwidth ≈ {self.bandwidth_hz / 1000:.1f} kHz"


def estimate(s: Spectra, original_sample_rate: int | None = None) -> CodecEstimate:
    voiced = s.power[s.voiced] if s.voiced.sum() >= 5 else s.power
    ltas_db = 10 * np.log10(voiced.mean(axis=0) + EPS)
    # Bandwidth: highest frequency whose smoothed level is within 50 dB of the
    # spectral peak (relative to the peak, so resampler leakage doesn't count).
    smooth = np.convolve(ltas_db, np.ones(5) / 5, mode="same")
    above = np.where(smooth > smooth.max() - 50.0)[0]
    bandwidth = float(_FREQS[above.max()]) if len(above) else 0.0

    # Spectral holes: time-frequency cells in the speech band far below their frame's level.
    band = (_FREQS > 300) & (_FREQS < min(bandwidth, 7000) if bandwidth else _FREQS < 3400)
    cells = 10 * np.log10(voiced[:, band] + EPS)
    holes = float((cells < cells.max(axis=1, keepdims=True) - 50).mean()) if cells.size else 0.0

    narrow = bandwidth < 4300 or (original_sample_rate is not None and original_sample_rate <= 8000)
    if bandwidth == 0.0:
        return CodecEstimate("unknown", 0.0, bandwidth, holes)
    if narrow:
        if holes > 0.15:
            return CodecEstimate(
                "narrowband_lowrate", float(np.clip(holes * 3, 0.3, 0.9)), bandwidth, holes
            )
        return CodecEstimate("narrowband_g711", 0.7, bandwidth, holes)
    if holes > 0.12 or bandwidth < 7000:
        return CodecEstimate(
            "wideband_compressed", float(np.clip(0.4 + holes * 2, 0.4, 0.85)), bandwidth, holes
        )
    return CodecEstimate("wideband_clean", 0.6, bandwidth, holes)
