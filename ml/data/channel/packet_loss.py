"""Channel destruction: packet loss with simple concealment and jitter (B3-T07)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.data.channel.base import ChannelRecord


@dataclass
class PacketLoss:
    """Drop 20 ms frames (Gilbert-style bursts) and conceal by fading repetition."""

    loss_rate: float = 0.03
    frame_ms: int = 20
    mean_burst: float = 1.5

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        y = x.astype(np.float32).copy()
        flen = max(1, sr * self.frame_ms // 1000)
        n_frames = len(y) // flen
        # Two-state Markov chain whose stationary loss probability == loss_rate.
        p_exit = 1.0 / max(self.mean_burst, 1.0)
        p_enter = self.loss_rate * p_exit / max(1e-9, 1.0 - self.loss_rate)
        lost = False
        last = np.zeros(flen, dtype=np.float32)
        gain = 1.0
        for i in range(n_frames):
            lost = rng.random() < (1 - p_exit if lost else p_enter)
            s = slice(i * flen, (i + 1) * flen)
            if lost:
                gain *= 0.5
                y[s] = last * gain
            else:
                last = y[s].copy()
                gain = 1.0
        rec.codec_chain.append(f"loss:{self.loss_rate:g}")
        return y, sr


@dataclass
class Jitter:
    """Small time-warp from playout-clock drift (resampling by a tiny factor)."""

    max_ppm: float = 300.0

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        ppm = rng.uniform(-self.max_ppm, self.max_ppm)
        n = len(x)
        t = np.arange(n) * (1 + ppm * 1e-6)
        y = np.interp(t, np.arange(n), x, right=0.0).astype(np.float32)
        rec.codec_chain.append("jitter")
        return y, sr
