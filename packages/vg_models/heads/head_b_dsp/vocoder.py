"""Vocoder / neural-codec fingerprinting (B5-T02).

Hand-built detectors for artifacts that neural waveform generators leave behind.
Each returns a strength in [0, 1]; they are *features* for the classifier and
sources of explanations, not verdicts on their own.

* Upsampling tones — transposed-convolution vocoders (HiFi-GAN, BigVGAN family)
  leave narrow spectral peaks at multiples of ``sr / upsample_factor`` that do
  not move with the voice's pitch. Detected as stationary narrow peaks in the
  long-term spectrum that are stronger than their neighbourhood in every frame.
* Band-edge brick wall — neural codecs (EnCodec / SoundStream family) and many
  TTS systems synthesise at a fixed bandwidth, producing an unnaturally sharp
  roll-off edge compared to acoustic low-pass behaviour.
* Phase regularity — overly consistent frame-to-frame phase advance in the
  harmonic region (vocoders that predict phase poorly or too smoothly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.vg_models.heads.head_b_dsp.features import _FREQS, EPS, Spectra


@dataclass
class VocoderFingerprint:
    upsampling_tones: float
    tone_freqs_hz: list[float]
    band_edge_sharpness: float
    band_edge_hz: float
    phase_regularity: float

    def as_features(self) -> dict[str, float]:
        return {
            "voc_upsampling_tones": self.upsampling_tones,
            "voc_band_edge_sharpness": self.band_edge_sharpness,
            "voc_band_edge_hz": self.band_edge_hz / 8000.0,
            "voc_phase_regularity": self.phase_regularity,
        }


def _upsampling_tones(s: Spectra) -> tuple[float, list[float]]:
    db = 10 * np.log10(s.power)  # [T, F]
    ltas = np.median(db, axis=0)
    # Local neighbourhood median (±6 bins) — peaks must stand out from it.
    pad = np.pad(ltas, 6, mode="edge")
    neigh = np.median(np.lib.stride_tricks.sliding_window_view(pad, 13), axis=1)
    prominence = ltas - neigh
    cand = np.where((prominence > 6.0) & (_FREQS > 500))[0]
    if len(cand) == 0:
        return 0.0, []
    # Stationarity: the peak must be present in most frames, unlike pitch harmonics that move.
    frame_prom = db[:, cand] - np.median(
        db[:, np.clip(cand[:, None] + np.arange(-6, 7), 0, db.shape[1] - 1)], axis=2
    )
    stationary = (frame_prom > 3.0).mean(axis=0) > 0.8
    tones = cand[stationary]
    if len(tones) == 0:
        return 0.0, []
    strength = float(np.clip(prominence[tones].mean() / 20.0, 0, 1) * min(1.0, len(tones) / 3))
    return strength, [float(_FREQS[t]) for t in tones[:5]]


def _band_edge(s: Spectra) -> tuple[float, float]:
    ltas_db = 10 * np.log10(s.power.mean(axis=0) + EPS)
    # Edge relative to the spectral peak: window sidelobe leakage above a hard
    # cut-off sits far below the peak but far above the digital floor.
    smooth = np.convolve(ltas_db, np.ones(3) / 3, mode="same")
    above = np.where(smooth > smooth.max() - 40.0)[0]
    if len(above) == 0:
        return 0.0, 0.0
    edge = int(above.max())
    if edge >= len(ltas_db) - 3:
        return 0.0, float(_FREQS[edge])  # full band, no edge
    lo = max(edge - 4, 0)
    hi = min(edge + 4, len(ltas_db) - 1)
    drop_db_per_bin = (ltas_db[lo] - ltas_db[hi]) / (hi - lo)
    # Acoustic/analog low-pass: a few dB/bin. Brick wall: >= ~6 dB/bin.
    return float(np.clip((drop_db_per_bin - 2.0) / 6.0, 0, 1)), float(_FREQS[edge])


def _phase_regularity(s: Spectra) -> float:
    if s.voiced.sum() < 6:
        return 0.0
    frames = s.phase_frames[s.voiced][:40]
    spec = np.fft.rfft(frames, axis=1)
    band = (_FREQS > 100) & (_FREQS < 2000)
    ph = np.angle(spec[:, band])
    mag = np.abs(spec[:, band])
    dphi = np.angle(np.exp(1j * np.diff(ph, n=2, axis=0)))  # 2nd difference: 0 for steady sinusoids
    w = mag[2:] / (mag[2:].sum() + EPS)
    circ_var = 1 - np.abs((w * np.exp(1j * dphi)).sum()) / (w.sum() + EPS)
    # Natural speech: high variance (~0.8-1). Very regular synthetic phase: low.
    return float(np.clip((0.7 - circ_var) / 0.7, 0, 1))


def fingerprint(s: Spectra) -> VocoderFingerprint:
    tones, freqs = _upsampling_tones(s)
    sharp, edge_hz = _band_edge(s)
    return VocoderFingerprint(tones, freqs, sharp, edge_hz, _phase_regularity(s))
