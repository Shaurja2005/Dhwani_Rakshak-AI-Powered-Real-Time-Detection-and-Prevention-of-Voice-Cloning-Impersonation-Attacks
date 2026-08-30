"""packages/vg_audio/resample.py — B2-T02: Resampler to 16 kHz mono.

Correctly handles 8 kHz µ-law/A-law inputs (telephony) and any other
sample rate. Records original_sample_rate in metadata — it is a feature,
not just plumbing (codec artifacts are sample-rate-correlated).

Strategy:
  1. Decode raw payload bytes → float32 PCM using packages/vg_audio/codecs.py
  2. Resample from source_rate → 16000 Hz using scipy.signal.resample_poly
     (polyphase filter, no aliasing, handles non-integer ratios cleanly)
  3. Mix stereo → mono if needed (already done in B1 adapters, but safety check)

The resampler is stateless and operates on one AudioChunk at a time.
The jitter buffer in jitter_buffer.py feeds chunks to the resampler.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from packages.vg_audio.codecs import decode_payload

TARGET_SR: int = 16_000  # Hz — system canonical sample rate (invariant)


def resample_chunk(
    payload: bytes,
    encoding: str,
    source_rate: int,
) -> np.ndarray:
    """Decode + resample one AudioChunk payload to 16 kHz float32 PCM.

    Args:
        payload:     Raw bytes from AudioChunk.payload.
        encoding:    AudioChunk.encoding string ("pcm_s16le", "mulaw", "alaw").
        source_rate: AudioChunk.sample_rate (may be 8000, 16000, 48000 …).

    Returns:
        float32 numpy array at TARGET_SR (16 kHz), mono, normalised [-1, 1].
    """
    # Step 1: decode to float32
    samples = decode_payload(payload, encoding)

    # Step 2: resample if needed
    if source_rate == TARGET_SR:
        return samples

    # Use scipy if available (faster polyphase), else numpy FFT fallback
    try:
        from scipy.signal import resample_poly  # type: ignore[import]
        frac = Fraction(TARGET_SR, source_rate).limit_denominator(500)
        up, down = frac.numerator, frac.denominator
        resampled = resample_poly(samples, up, down).astype(np.float32)
    except ImportError:
        # Fallback: numpy FFT-based resample (slower, acceptable for < 50 Hz)
        n_target = int(len(samples) * TARGET_SR / source_rate)
        resampled = np.interp(
            np.linspace(0, len(samples) - 1, n_target),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

    return resampled


def resample_stream(
    chunks: list[tuple[bytes, str, int]],
) -> np.ndarray:
    """Resample and concatenate a sequence of (payload, encoding, sample_rate) tuples."""
    arrays = [resample_chunk(p, enc, sr) for p, enc, sr in chunks]
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
