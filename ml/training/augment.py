"""ml.training.augment — on-the-fly augmentation for Head A (B4-T05).

* RawBoost (Tak et al., 2022): (1) linear+nonlinear convolutive noise,
  (2) impulsive signal-dependent noise, (3) stationary coloured additive noise.
* Channel augmentation: the B3 channel simulator.

Both are label-blind by construction — they take only audio and an RNG
(invariant I3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter

from ml.data.channel.webrtc_chain import ChannelSimulator


def _random_fir(rng: np.random.Generator, sr: int, n_bands: int = 5) -> np.ndarray:
    taps = np.zeros(int(rng.integers(10, 100)) | 1)
    taps[len(taps) // 2] = 1.0
    for _ in range(int(rng.integers(1, n_bands + 1))):
        fc = rng.uniform(20, sr / 2 - 1000)
        bw = rng.uniform(100, 1000)
        lo, hi = max(fc - bw / 2, 20), min(fc + bw / 2, sr / 2 - 1)
        taps = np.convolve(taps, firwin(len(taps), [lo, hi], pass_zero=False, fs=sr), mode="same")
    return taps / (np.max(np.abs(taps)) + 1e-9)


def rawboost_convolutive(x: np.ndarray, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    y = np.zeros_like(x)
    for k in range(1, int(rng.integers(2, 5))):  # nonlinear orders
        y += lfilter(_random_fir(rng, sr), [1.0], x**k) * rng.uniform(0.5, 1.0) / k
    return y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))


def rawboost_impulsive(x: np.ndarray, rng: np.random.Generator, p: float = 0.1) -> np.ndarray:
    y = x.copy()
    n = int(len(x) * rng.uniform(0, p))
    idx = rng.choice(len(x), size=n, replace=False)
    y[idx] += x[idx] * rng.uniform(-1, 1, size=n) * 2.0
    return y


def rawboost_additive(x: np.ndarray, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    noise = lfilter(_random_fir(rng, sr), [1.0], rng.standard_normal(len(x)))
    snr = rng.uniform(10, 40)
    scale = np.sqrt(np.mean(x**2) / (np.mean(noise**2) * 10 ** (snr / 10) + 1e-12))
    return x + scale * noise


@dataclass
class Augmenter:
    rawboost_algo: int = 0  # 0 off, 1..3 single, 4 = 1+2+3 serial (RawBoost paper numbering)
    p_rawboost: float = 0.7
    p_channel: float = 0.5
    sr: int = 16000
    channel: ChannelSimulator | None = None

    def __call__(self, x: np.ndarray, rng: np.random.Generator, key: str) -> np.ndarray:
        y = x.astype(np.float64)
        if self.rawboost_algo and rng.random() < self.p_rawboost:
            if self.rawboost_algo in (1, 4):
                y = rawboost_convolutive(y, rng, self.sr)
            if self.rawboost_algo in (2, 4):
                y = rawboost_impulsive(y, rng)
            if self.rawboost_algo in (3, 4):
                y = rawboost_additive(y, rng, self.sr)
        if self.channel is not None and rng.random() < self.p_channel:
            # Fresh chain per draw: key + a random nonce, never the label.
            y, _, _ = self.channel.simulate(
                y.astype(np.float32), self.sr, f"{key}:{rng.integers(1 << 30)}"
            )
        return np.clip(y, -1, 1).astype(np.float32)
