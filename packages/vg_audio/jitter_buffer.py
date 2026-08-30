"""packages/vg_audio/jitter_buffer.py — B2-T01: Jitter buffer + Packet-Loss Concealment.

Handles the reordering and gap-filling that RTP/WebSocket paths require
before audio can be safely resampled and windowed.

Design:
- Sorted buffer keyed by seq number, max depth MAX_BUFFER_DEPTH packets.
- Packets that arrive out-of-order within the window are reordered.
- If a gap is detected (seq jump > 1), PLC fills with the last-known frame
  or zero-padding (silence) up to MAX_CONCEAL_FRAMES frames.
- Packets older than MAX_BUFFER_DEPTH are flushed (late arrival = dropped).

PLC strategy: simple zero-order hold (repeat last frame) for up to
MAX_CONCEAL_FRAMES; beyond that fall back to silence. This is adequate
for the detection task — we're not trying to produce perceptually perfect
audio, just maintain accurate timing for the windowing step.

This module produces float32 PCM at the SOURCE sample rate.
The resampler in resample.py converts to 16 kHz.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.vg_audio.codecs import decode_payload
from packages.vg_core.logging import get_logger

log = get_logger(__name__)

# Maximum number of out-of-order packets to buffer before flushing.
MAX_BUFFER_DEPTH: int = 20
# Maximum number of consecutive lost packets to conceal before inserting silence.
MAX_CONCEAL_FRAMES: int = 5


@dataclass(order=True)
class _BufferedChunk:
    """A packet stored in the jitter buffer, sortable by seq."""
    seq: int
    ts_ms: int = field(compare=False)
    payload: bytes = field(compare=False)
    encoding: str = field(compare=False)
    sample_rate: int = field(compare=False)
    received_at: float = field(default_factory=time.monotonic, compare=False)


class JitterBuffer:
    """Per-session jitter buffer with PLC for a single audio stream.

    Usage::

        buf = JitterBuffer()
        for chunk in incoming_chunks:
            for pcm in buf.push(chunk.seq, chunk.ts_ms, chunk.payload,
                                chunk.encoding.value, chunk.sample_rate):
                yield pcm   # float32 at source_rate, in seq order
    """

    def __init__(self) -> None:
        self._buffer: list[_BufferedChunk] = []
        self._next_seq: Optional[int] = None
        self._last_frame: Optional[np.ndarray] = None
        self._stats = {"reordered": 0, "dropped_late": 0, "concealed_frames": 0}

    def push(
        self,
        seq: int,
        ts_ms: int,
        payload: bytes,
        encoding: str,
        sample_rate: int,
    ) -> list[np.ndarray]:
        """Insert a packet and return a list of in-order float32 PCM arrays.

        May return 0, 1, or more arrays depending on how many are ready
        (including PLC frames for any gap before the new packet).
        """
        chunk = _BufferedChunk(seq, ts_ms, payload, encoding, sample_rate)

        # Initialise next_seq on first packet
        if self._next_seq is None:
            self._next_seq = seq

        # Drop very late packets (already past the buffer window)
        if seq < self._next_seq:
            self._stats["dropped_late"] += 1
            log.debug("jitter_late_drop", seq=seq, expected=self._next_seq)
            return []

        # Insert into sorted buffer
        self._buffer.append(chunk)
        self._buffer.sort()

        return self._flush()

    def _flush(self) -> list[np.ndarray]:
        """Flush all packets from the buffer that are now in order."""
        out: list[np.ndarray] = []

        while self._buffer:
            head = self._buffer[0]
            assert self._next_seq is not None

            if head.seq == self._next_seq:
                # Perfect: emit in order
                self._buffer.pop(0)
                pcm = decode_payload(head.payload, head.encoding)
                self._last_frame = pcm
                out.append(pcm)
                self._next_seq += 1

            elif head.seq > self._next_seq:
                gap = head.seq - self._next_seq

                if len(self._buffer) < MAX_BUFFER_DEPTH and gap <= MAX_BUFFER_DEPTH:
                    # Wait for more packets — may arrive out of order
                    break

                # Gap is too large or buffer is full — apply PLC
                conceal_n = min(gap, MAX_CONCEAL_FRAMES)
                log.debug(
                    "jitter_concealing",
                    gap=gap,
                    next_seq=self._next_seq,
                    concealing=conceal_n,
                )
                for _ in range(conceal_n):
                    if self._last_frame is not None:
                        out.append(self._last_frame.copy())  # zero-order hold
                    else:
                        # No prior frame — silence
                        out.append(np.zeros(160, dtype=np.float32))
                    self._next_seq += 1
                    self._stats["concealed_frames"] += 1

                # If gap is beyond conceal window, skip silently
                if gap > MAX_CONCEAL_FRAMES:
                    self._next_seq = head.seq
            else:
                # Should not happen after drop-late guard, but handle gracefully
                self._buffer.pop(0)

        return out

    def drain(self) -> list[np.ndarray]:
        """Force-flush all remaining buffered packets (end of session)."""
        out: list[np.ndarray] = []
        while self._buffer:
            chunk = self._buffer.pop(0)
            pcm = decode_payload(chunk.payload, chunk.encoding)
            self._last_frame = pcm
            out.append(pcm)
        return out

    @property
    def stats(self) -> dict:
        return dict(self._stats)
