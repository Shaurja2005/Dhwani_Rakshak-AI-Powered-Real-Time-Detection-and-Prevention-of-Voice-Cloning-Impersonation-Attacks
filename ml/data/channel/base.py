"""Shared types for the channel destruction simulator (B3-T07).

A transform never sees the utterance label. That is the structural half of
invariant I3; ``symmetry.py`` is the statistical half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly


@dataclass
class ChannelRecord:
    """What was applied — written into the manifest row."""

    codec_chain: list[str] = field(default_factory=list)
    snr_db: float | None = None
    rir_id: str | None = None


class Transform(Protocol):
    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]: ...


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to_float32(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
