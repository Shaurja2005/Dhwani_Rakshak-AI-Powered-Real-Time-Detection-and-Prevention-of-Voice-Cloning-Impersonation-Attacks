"""Watermark detectors (B8-T04, B8-T05).

* ``AudioSealDetector`` — Meta AudioSeal (``facebookresearch/audioseal``),
  sample-level localized detection. Lazy import; requires ``pip install audioseal``
  and the ``audioseal_detector_16bits`` checkpoint.
* ``SpreadSpectrumDetector`` — a keyed spread-spectrum watermark for TTS systems
  you control or partners who share a key (e.g. your own demo/IVR voices). It is
  also the reference implementation that lets the present/absent asymmetry be
  tested end to end without downloads. ``embed_spread_spectrum`` adds the mark.
* ``PartnerAPIDetector`` — placeholder for commercial vendors whose detectors are
  only available as hosted APIs. Disabled unless the tenant explicitly opts in,
  because sending audio off-premise violates I6 by default.

Detectors report *presence evidence only*; see asymmetry.py for how absence is
handled (I4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SR = 16000


@dataclass
class WatermarkResult:
    detector: str
    score: float  # detector-native detection statistic
    probability: float  # probability the watermark is present, 0..1
    present: bool
    detail: dict[str, object]


class WatermarkDetector(Protocol):
    name: str

    def detect(self, pcm: np.ndarray, sample_rate: int = SR) -> WatermarkResult: ...


# ---------------------------------------------------------------- AudioSeal
class AudioSealDetector:
    name = "audioseal"

    def __init__(
        self, checkpoint: str = "audioseal_detector_16bits", threshold: float = 0.5
    ) -> None:
        import torch
        from audioseal import AudioSeal  # lazy, optional dependency

        self._torch = torch
        self._model = AudioSeal.load_detector(checkpoint).eval()
        self._threshold = threshold

    def detect(self, pcm: np.ndarray, sample_rate: int = SR) -> WatermarkResult:
        t = self._torch
        with t.inference_mode():
            wav = t.tensor(pcm, dtype=t.float32).view(1, 1, -1)
            prob, message = self._model.detect_watermark(wav, sample_rate)
        p = float(prob)
        bits = None if message is None else message.squeeze().tolist()
        return WatermarkResult(self.name, p, p, p >= self._threshold, {"message_bits": bits})


# ---------------------------------------------------------------- keyed spread spectrum
PERIOD = 4096


def _pn(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(f"vg-wm:{key}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=PERIOD)


def embed_spread_spectrum(pcm: np.ndarray, key: str, strength_db: float = -26.0) -> np.ndarray:
    """Add a periodic keyed ±1 sequence at ``strength_db`` relative to the signal RMS."""
    x = np.asarray(pcm, dtype=np.float64)
    reps = int(np.ceil(len(x) / PERIOD))
    pn = np.tile(_pn(key), reps)[: len(x)]
    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    return np.clip(x + rms * 10 ** (strength_db / 20) * pn, -1, 1).astype(np.float32)


class SpreadSpectrumDetector:
    """Blind-synchronised correlation detector.

    The signal is whitened (first difference), folded modulo the PN period, and
    circularly cross-correlated with the key sequence via FFT, so detection works
    from any window offset. The statistic is the peak correlation as a robust
    z-score (median / MAD over all 4096 lags); without the keyed watermark the
    peak is the max of ~4096 near-normal values (≈ 4), so the default threshold
    of 8 keeps false positives negligible.
    """

    def __init__(self, keys: dict[str, str], threshold_z: float = 8.0) -> None:
        self.name = "spread_spectrum"
        self._keys = {kid: np.diff(_pn(k), append=_pn(k)[:1]) for kid, k in keys.items()}
        self._threshold = threshold_z

    def detect(self, pcm: np.ndarray, sample_rate: int = SR) -> WatermarkResult:
        x = np.diff(np.asarray(pcm, dtype=np.float64), append=0.0)
        n_periods = len(x) // PERIOD
        if n_periods < 2:
            return WatermarkResult(self.name, 0.0, 0.0, False, {"note": "window too short"})
        folded = x[: n_periods * PERIOD].reshape(n_periods, PERIOD)
        best_z, best_key, best_lag = 0.0, None, 0
        fx = np.fft.rfft(folded, axis=1)
        for kid, ref in self._keys.items():
            fr = np.conj(np.fft.rfft(ref))
            corr = np.fft.irfft(fx * fr, n=PERIOD, axis=1)  # [periods, lags]
            total = corr.sum(axis=0)
            # Robust z against the correlation's own background across all lags: a genuine
            # key produces one outlier lag; a wrong key (even on watermarked audio) does not.
            med = np.median(total)
            mad = 1.4826 * np.median(np.abs(total - med))
            z = (total - med) / (mad + 1e-12)
            lag = int(np.argmax(z))
            if z[lag] > best_z:
                best_z, best_key, best_lag = float(z[lag]), kid, lag
        prob = float(1 / (1 + np.exp(-(best_z - self._threshold))))
        return WatermarkResult(
            self.name,
            best_z,
            prob,
            best_z >= self._threshold,
            {"key_id": best_key if best_z >= self._threshold else None, "lag": best_lag},
        )


# ---------------------------------------------------------------- partner APIs (opt-in only)
class PartnerAPIDetector:
    """Hosted commercial watermark detector. Requires explicit tenant opt-in (I6)."""

    def __init__(self, vendor: str, tenant_opted_in: bool) -> None:
        if not tenant_opted_in:
            raise PermissionError(
                f"{vendor} watermark detection sends audio off-premise; tenant opt-in required (I6)"
            )
        self.name = f"partner:{vendor}"

    def detect(self, pcm: np.ndarray, sample_rate: int = SR) -> WatermarkResult:
        raise NotImplementedError("vendor integration not configured (B8-T05)")
