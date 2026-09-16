"""Channel destruction: additive noise at a target SNR (B3-T07).

Uses a MUSAN noise bank when provided, otherwise synthetic coloured noise so
the simulator works (and is testable) without any downloaded corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.data.channel.base import ChannelRecord


def colored_noise(n: int, rng: np.random.Generator, exponent: float) -> np.ndarray:
    """1/f^exponent noise; 0 = white, 1 = pink, 2 = brown."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec /= f ** (exponent / 2)
    y = np.fft.irfft(spec, n)
    return (y / (np.std(y) + 1e-12)).astype(np.float32)


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    px = float(np.mean(x**2)) + 1e-12
    pn = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return (x + scale * noise).astype(np.float32)


@dataclass
class AdditiveNoise:
    snr_range_db: tuple[float, float] = (5.0, 20.0)
    noise_bank: list[np.ndarray] = field(default_factory=list)

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        snr = float(rng.uniform(*self.snr_range_db))
        if self.noise_bank:
            idx = int(rng.integers(len(self.noise_bank)))
            noise = self.noise_bank[idx]
            if len(noise) > len(x):
                start = int(rng.integers(len(noise) - len(x) + 1))
                noise = noise[start : start + len(x)]
            tag = f"noise:bank{idx}"
        else:
            kind = int(rng.integers(3))
            noise = colored_noise(len(x), rng, float(kind))
            tag = f"noise:{('white', 'pink', 'brown')[kind]}"
        rec.snr_db = round(snr, 2)
        rec.codec_chain.append(tag)
        return mix_at_snr(x, noise, snr), sr
