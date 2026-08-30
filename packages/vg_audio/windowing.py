"""packages/vg_audio/windowing.py — B2-T04: Fixed-length windowing with hop.

Accumulates 16 kHz float32 PCM samples from the jitter buffer + resampler
and emits fixed-size windows on a configurable hop schedule.

Defaults (SOURCE_OF_TRUTH §3):
  window_duration_s = 3.0
  hop_s            = 1.0

Window IDs are monotonically increasing per session, starting at 0.

The Windower is stateful per session. Instantiate one per session_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from packages.vg_core.logging import get_logger

log = get_logger(__name__)

TARGET_SR = 16_000
DEFAULT_WINDOW_S = 3.0
DEFAULT_HOP_S = 1.0


@dataclass
class WindowSlice:
    """One analysis window of float32 PCM ready for the quality gate."""
    window_id: int
    start_ms: int
    end_ms: int
    pcm: np.ndarray          # float32, shape=(window_samples,), at TARGET_SR
    original_sample_rate: int


class Windower:
    """Accumulates PCM and emits fixed-length windows.

    Usage::

        windower = Windower(session_id="...", original_sample_rate=8000)
        for pcm_chunk in resampled_chunks:
            for window in windower.push(pcm_chunk):
                quality_result = assess_quality(window.pcm, ...)
                # ... build AnalysisWindow and publish
        # On session end:
        for window in windower.flush():
            ...
    """

    def __init__(
        self,
        session_id: str,
        original_sample_rate: int,
        window_s: float = DEFAULT_WINDOW_S,
        hop_s: float = DEFAULT_HOP_S,
        sample_rate: int = TARGET_SR,
    ) -> None:
        self.session_id = session_id
        self.original_sample_rate = original_sample_rate
        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.hop_samples = int(hop_s * sample_rate)

        self._buffer = np.array([], dtype=np.float32)
        self._window_id = 0
        self._samples_consumed = 0  # total samples pushed so far

    def push(self, pcm: np.ndarray) -> list[WindowSlice]:
        """Append PCM data and return all complete windows ready for scoring."""
        self._buffer = np.concatenate([self._buffer, pcm])
        return self._drain()

    def _drain(self) -> list[WindowSlice]:
        windows: list[WindowSlice] = []
        while len(self._buffer) >= self.window_samples:
            window_pcm = self._buffer[: self.window_samples].copy()

            start_sample = self._samples_consumed
            end_sample = start_sample + self.window_samples
            start_ms = int(start_sample * 1000 / self.sample_rate)
            end_ms = int(end_sample * 1000 / self.sample_rate)

            windows.append(WindowSlice(
                window_id=self._window_id,
                start_ms=start_ms,
                end_ms=end_ms,
                pcm=window_pcm,
                original_sample_rate=self.original_sample_rate,
            ))

            self._window_id += 1
            self._samples_consumed += self.hop_samples
            # Advance by hop, not full window (overlapping windows)
            self._buffer = self._buffer[self.hop_samples :]

        return windows

    def flush(self) -> list[WindowSlice]:
        """Emit partial window at end of session (pad with silence to full length)."""
        if len(self._buffer) < int(0.5 * self.window_samples):
            # Too short — discard
            return []
        padded = np.zeros(self.window_samples, dtype=np.float32)
        padded[: len(self._buffer)] = self._buffer
        start_ms = int(self._samples_consumed * 1000 / self.sample_rate)
        end_ms = start_ms + int(self.window_samples * 1000 / self.sample_rate)
        return [WindowSlice(
            window_id=self._window_id,
            start_ms=start_ms,
            end_ms=end_ms,
            pcm=padded,
            original_sample_rate=self.original_sample_rate,
        )]

    @property
    def window_duration_s(self) -> float:
        return self.window_samples / self.sample_rate

    @property
    def next_window_id(self) -> int:
        return self._window_id
