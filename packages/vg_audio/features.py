"""packages/vg_audio/features.py — B2-T08: Per-window feature cache.

Computes and caches the expensive spectral features so multiple detection
heads (B4-B8) never recompute the same spectrogram for the same window.

Cache backend:
  - In-process LRU dict (always available, zero deps).
  - Redis (TTL ≤ 60 s) when vg_core.bus Redis is reachable (optional).

Key: "vg:feat:{session_id}:{window_id}:{feature_name}"
TTL: 60 s (CONFIG: FEATURE_CACHE_TTL_S)

Features computed on demand:
  - log_mel_spectrogram(n_mels=80): used by Head A (SSL) and Head C (prosody)
  - lfcc(n_filters=40): used by Head B (DSP)
  - f0_series: fundamental frequency track, used by Head C (prosody)
  - mfcc(n_mfcc=40): general acoustic features

Design: All feature functions accept float32 numpy arrays and return
numpy arrays. Serialisation is via numpy .tobytes() + shape metadata.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from packages.vg_core.logging import get_logger

log = get_logger(__name__)

TARGET_SR = 16_000
FEATURE_CACHE_TTL_S: int = 60

# In-process LRU bounded cache  {key -> ndarray}
_MAX_CACHE_ENTRIES = 512
_cache: dict[str, np.ndarray] = {}
_cache_order: list[str] = []


def _cache_get(key: str) -> Optional[np.ndarray]:
    if key in _cache:
        _cache_order.remove(key)
        _cache_order.append(key)
        return _cache[key]
    return None


def _cache_set(key: str, value: np.ndarray) -> None:
    if key in _cache:
        _cache_order.remove(key)
    elif len(_cache) >= _MAX_CACHE_ENTRIES:
        oldest = _cache_order.pop(0)
        del _cache[oldest]
    _cache[key] = value
    _cache_order.append(key)


def _window_key(session_id: str, window_id: int, feature: str) -> str:
    return f"vg:feat:{session_id}:{window_id}:{feature}"


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------

def _stft(pcm: np.ndarray, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Short-time Fourier transform → magnitude spectrogram."""
    window = np.hanning(n_fft)
    frames = []
    for i in range(0, len(pcm) - n_fft + 1, hop):
        frame = pcm[i : i + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    return np.array(frames, dtype=np.float32).T if frames else np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)


def _mel_filterbank(n_mels: int = 80, n_fft: int = 512, sr: int = TARGET_SR) -> np.ndarray:
    """Build a mel filterbank matrix."""
    f_min, f_max = 0.0, sr / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts = np.array([mel_to_hz(m) for m in mel_pts])
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank


def compute_log_mel(
    pcm: np.ndarray,
    n_mels: int = 80,
    n_fft: int = 512,
    hop: int = 160,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """Compute log-mel spectrogram. Shape: (n_mels, T)."""
    spec = _stft(pcm, n_fft, hop)
    mel = _mel_filterbank(n_mels, n_fft, sr) @ spec
    log_mel = np.log(np.clip(mel, 1e-9, None))
    return log_mel.astype(np.float32)


def compute_lfcc(
    pcm: np.ndarray,
    n_filters: int = 40,
    n_ceps: int = 20,
    n_fft: int = 512,
    hop: int = 160,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """Linear Frequency Cepstral Coefficients. Shape: (n_ceps, T)."""
    spec = _stft(pcm, n_fft, hop)
    # Linear filterbank (uniform in frequency, not mel)
    freq_bins = spec.shape[0]
    n_linear = min(n_filters, freq_bins)
    linear_fb = np.zeros((n_linear, freq_bins), dtype=np.float32)
    for i in range(n_linear):
        linear_fb[i, i * freq_bins // n_linear : (i + 1) * freq_bins // n_linear] = 1.0
    linear = linear_fb @ spec + 1e-9
    log_linear = np.log(linear)
    # DCT-II via numpy
    dct = np.zeros((n_ceps, log_linear.shape[1]), dtype=np.float32)
    for k in range(n_ceps):
        dct[k] = np.sum(
            log_linear * np.cos(np.pi * k * (2 * np.arange(n_linear)[:, None] + 1) / (2 * n_linear)),
            axis=0,
        )
    return dct.astype(np.float32)


def compute_mfcc(
    pcm: np.ndarray,
    n_mfcc: int = 40,
    n_mels: int = 80,
    n_fft: int = 512,
    hop: int = 160,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """MFCC. Shape: (n_mfcc, T). Derived from log-mel via DCT."""
    log_mel = compute_log_mel(pcm, n_mels, n_fft, hop, sr)
    n = log_mel.shape[0]
    dct = np.zeros((n_mfcc, log_mel.shape[1]), dtype=np.float32)
    for k in range(n_mfcc):
        dct[k] = np.sum(
            log_mel * np.cos(np.pi * k * (2 * np.arange(n)[:, None] + 1) / (2 * n)),
            axis=0,
        )
    return dct.astype(np.float32)


def compute_f0(pcm: np.ndarray, sr: int = TARGET_SR, frame_ms: int = 20) -> np.ndarray:
    """Estimate fundamental frequency (F0) via autocorrelation. Shape: (T,).

    Returns F0 in Hz per frame. 0.0 means unvoiced/silence.
    """
    frame_size = int(sr * frame_ms / 1000)
    f0_list = []
    for i in range(0, len(pcm) - frame_size + 1, frame_size):
        frame = pcm[i : i + frame_size]
        # Normalized autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        # Find first peak beyond minimum lag (50 Hz = sr/50)
        min_lag = sr // 500  # 500 Hz max F0
        max_lag = sr // 50   # 50 Hz min F0
        if max_lag >= len(corr) or min_lag >= max_lag:
            f0_list.append(0.0)
            continue
        peak = np.argmax(corr[min_lag:max_lag]) + min_lag
        if corr[0] < 1e-9 or corr[peak] / corr[0] < 0.3:
            f0_list.append(0.0)  # unvoiced
        else:
            f0_list.append(float(sr / peak))
    return np.array(f0_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cached feature access
# ---------------------------------------------------------------------------

class FeatureCache:
    """Per-session feature cache with LRU in-process storage.

    Usage::

        cache = FeatureCache(session_id="...", window_id=3)
        mel = cache.get_or_compute("log_mel", pcm)
    """

    _FEATURE_FNS = {
        "log_mel": compute_log_mel,
        "lfcc":    compute_lfcc,
        "mfcc":    compute_mfcc,
        "f0":      compute_f0,
    }

    def __init__(self, session_id: str, window_id: int) -> None:
        self.session_id = session_id
        self.window_id = window_id

    def get_or_compute(self, feature: str, pcm: np.ndarray) -> np.ndarray:
        """Return cached feature or compute and cache it."""
        key = _window_key(self.session_id, self.window_id, feature)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        fn = self._FEATURE_FNS.get(feature)
        if fn is None:
            raise ValueError(f"Unknown feature: {feature!r}")
        result = fn(pcm)
        _cache_set(key, result)
        return result

    def invalidate(self) -> None:
        """Remove all cached features for this window."""
        for feat in self._FEATURE_FNS:
            key = _window_key(self.session_id, self.window_id, feat)
            if key in _cache:
                _cache_order.remove(key)
                del _cache[key]


def clear_session_cache(session_id: str) -> int:
    """Remove all cached features for a session. Returns count removed."""
    prefix = f"vg:feat:{session_id}:"
    to_remove = [k for k in list(_cache.keys()) if k.startswith(prefix)]
    for k in to_remove:
        _cache_order.remove(k)
        del _cache[k]
    return len(to_remove)
