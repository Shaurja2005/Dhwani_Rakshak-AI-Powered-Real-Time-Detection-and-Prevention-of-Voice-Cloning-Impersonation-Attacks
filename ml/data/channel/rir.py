"""Channel destruction: room impulse response convolution (B3-T07).

Uses RIRS_NOISES impulse responses when provided, otherwise a synthetic
exponentially-decaying RIR with a sampled RT60.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from ml.data.channel.base import ChannelRecord


def synthetic_rir(sr: int, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * min(rt60_s * 1.2, 1.5))
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / rt60_s)  # -60 dB at rt60
    h = rng.standard_normal(n) * decay
    h[0] = 1.0  # direct path
    return (h / np.max(np.abs(h))).astype(np.float32)


@dataclass
class RIRConvolve:
    rt60_range_s: tuple[float, float] = (0.15, 0.8)
    rir_bank: dict[str, np.ndarray] = field(default_factory=dict)

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        if self.rir_bank:
            keys = sorted(self.rir_bank)
            rir_id = keys[int(rng.integers(len(keys)))]
            h = self.rir_bank[rir_id]
        else:
            rt60 = float(rng.uniform(*self.rt60_range_s))
            rir_id = f"synthetic_rt60_{rt60:.2f}"
            h = synthetic_rir(sr, rt60, rng)
        y = fftconvolve(x, h, mode="full")[: len(x)]
        # Preserve level so RIR does not change loudness statistics.
        y *= np.sqrt((np.mean(x**2) + 1e-12) / (np.mean(y**2) + 1e-12))
        rec.rir_id = rir_id
        rec.codec_chain.append("rir")
        return y.astype(np.float32), sr
