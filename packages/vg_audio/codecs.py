"""packages/vg_audio/codecs.py — G.711 µ-law / A-law codec support.

Provides pure-Python decode tables so we have zero hard C-extension
dependencies in the core audio path.  Production deployments can swap
the implementation with audioop or numpy vectorised versions via the
same public API.

Also provides a `decode_chunk` entrypoint used by the resampler so it
receives float32 PCM regardless of the ingest encoding.
"""
from __future__ import annotations

import array
import struct
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# G.711 µ-law decode (RFC 3551 §4.5.14)
# ---------------------------------------------------------------------------
# Build the 256-entry lookup table once at import time.

def _build_mulaw_table() -> list[int]:
    table = []
    for i in range(256):
        u = ~i & 0xFF
        sign = (u & 0x80) >> 7
        exp  = (u & 0x70) >> 4
        mant = (u & 0x0F)
        val  = ((mant << 1) + 33) << exp
        val -= 33
        table.append(-val if sign else val)
    return table

_MULAW_TABLE: list[int] = _build_mulaw_table()


def mulaw_to_linear(data: bytes) -> np.ndarray:
    """Decode G.711 µ-law bytes → float32 array in [-1, 1]."""
    samples = np.array([_MULAW_TABLE[b] for b in data], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# G.711 A-law decode (ITU-T G.711)
# ---------------------------------------------------------------------------

def _build_alaw_table() -> list[int]:
    table = []
    for i in range(256):
        a = i ^ 0x55
        sign = (a & 0x80) >> 7
        exp  = (a & 0x70) >> 4
        mant = (a & 0x0F)
        if exp == 0:
            val = (mant << 1) | 1
        else:
            val = ((mant | 0x10) << 1) | 1
            val <<= (exp - 1)
        table.append(-val if sign else val)
    return table

_ALAW_TABLE: list[int] = _build_alaw_table()


def alaw_to_linear(data: bytes) -> np.ndarray:
    """Decode G.711 A-law bytes → float32 array in [-1, 1]."""
    samples = np.array([_ALAW_TABLE[b] for b in data], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# PCM s16le decode
# ---------------------------------------------------------------------------

def pcm_s16le_to_float32(data: bytes) -> np.ndarray:
    """Decode raw signed 16-bit little-endian PCM → float32 in [-1, 1]."""
    samples = np.frombuffer(data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Unified decode gateway
# ---------------------------------------------------------------------------

def decode_payload(payload: bytes, encoding: str) -> np.ndarray:
    """Decode an AudioChunk payload bytes → float32 PCM array.

    Args:
        payload:  raw bytes from AudioChunk.payload
        encoding: AudioChunk.encoding value string ("pcm_s16le", "mulaw", "alaw")

    Returns:
        float32 numpy array in [-1.0, 1.0].  Length = number of samples.
    """
    enc = encoding.lower()
    if enc in ("pcm_s16le", "pcm", "linear16"):
        return pcm_s16le_to_float32(payload)
    elif enc in ("mulaw", "g711u", "pcmu"):
        return mulaw_to_linear(payload)
    elif enc in ("alaw", "g711a", "pcma"):
        return alaw_to_linear(payload)
    else:
        raise ValueError(f"Unsupported encoding: {encoding!r}")
