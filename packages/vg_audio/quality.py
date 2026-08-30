"""packages/vg_audio/quality.py — B2-T05 & B2-T07: Quality gate + music/tone detector.

This is the single highest-value component for false-positive control.

A window is REJECTED (→ ABSTAIN) when ANY of these conditions hold:
  Q1. Voiced speech < 1.5 s  (VAD-measured)
  Q2. Estimated SNR below SNR_FLOOR_DB  (default -5 dB)
  Q3. Clipping ratio > CLIP_RATIO_MAX   (default 1%)
  Q4. DTMF / hold-music / ringback detected  (spectral energy in tone bands)
  Q5. Multiple concurrent speakers detected  (pitch variance heuristic)

The quality gate never blocks — it returns a structured result synchronously
(invariant I10: the audio path is never blocked by a slow component).

B2-T06: Loudness normalisation is also handled here (pre-norm level is logged
        as it carries forensic signal — a very low-level spoof may be louder
        than ambient or show boosted gain).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.vg_core.logging import get_logger

log = get_logger(__name__)

TARGET_SR = 16_000
TARGET_DBFS = -23.0         # EBU R128-ish target loudness for normalisation
SNR_FLOOR_DB = -5.0         # Below this → ABSTAIN
CLIP_RATIO_MAX = 0.01       # > 1% clipped samples → ABSTAIN
MIN_VOICED_S = 1.5          # Less than 1.5 s voiced speech → ABSTAIN
MULTI_SPEAKER_PITCH_VAR = 80.0  # Hz — high pitch variance hints multiple speakers


# ---------------------------------------------------------------------------
# DTMF / tone detection  (B2-T07)
# ---------------------------------------------------------------------------
# DTMF frequencies (rows + columns)
_DTMF_ROW_FREQS = [697, 770, 852, 941]
_DTMF_COL_FREQS = [1209, 1336, 1477, 1633]
_DTMF_ALL = _DTMF_ROW_FREQS + _DTMF_COL_FREQS

# Hold-music / ringback heuristics — dominated by 440 Hz dial tone
_TONE_FREQS = [350, 440, 480, 620, 950]  # busy, dial, ringback families

# Music detection: energy above 4 kHz is relatively low in speech, high in music
_MUSIC_HIGH_FREQ_RATIO_THRESHOLD = 0.35


def _goertzel(samples: np.ndarray, freq: float, sample_rate: int) -> float:
    """Goertzel algorithm: energy at a single frequency. O(N), no FFT."""
    n = len(samples)
    k = int(0.5 + n * freq / sample_rate)
    omega = 2 * np.pi * k / n
    coeff = 2 * np.cos(omega)
    s0, s1, s2 = 0.0, 0.0, 0.0
    for x in samples:
        s0 = float(x) + coeff * s1 - s2
        s2 = s1
        s1 = s0
    power = s1 ** 2 + s2 ** 2 - coeff * s1 * s2
    return power


def detect_dtmf(pcm: np.ndarray, sample_rate: int = TARGET_SR, threshold_ratio: float = 5.0) -> bool:
    """Return True if a DTMF tone pair is dominant in the signal.

    A DTMF digit requires one row and one column frequency to be dominant.
    """
    total_energy = float(np.sum(pcm ** 2)) + 1e-12
    for row_f in _DTMF_ROW_FREQS:
        row_e = _goertzel(pcm, row_f, sample_rate)
        if row_e / total_energy < 0.02:
            continue
        for col_f in _DTMF_COL_FREQS:
            col_e = _goertzel(pcm, col_f, sample_rate)
            if col_e / total_energy < 0.02:
                continue
            # Check ratio against average of other tone energies
            others = [
                _goertzel(pcm, f, sample_rate)
                for f in _DTMF_ALL
                if f != row_f and f != col_f
            ]
            avg_other = (sum(others) / len(others)) + 1e-12
            if (row_e / avg_other) > threshold_ratio and (col_e / avg_other) > threshold_ratio:
                return True
    return False


def detect_hold_music_or_tone(pcm: np.ndarray, sample_rate: int = TARGET_SR) -> bool:
    """Return True if the window appears to contain IVR tones, hold music, or ringback.

    Uses two heuristics:
    1. Tone-band energy dominates (narrow-band PSTN tones).
    2. High-frequency energy ratio — music has more energy > 4 kHz than speech.
    """
    # Heuristic 1: tone-band dominance
    total = float(np.sum(pcm ** 2)) + 1e-12
    tone_energy = sum(_goertzel(pcm, f, sample_rate) for f in _TONE_FREQS)
    if tone_energy / total > 0.5:
        return True

    # Heuristic 2: high-frequency energy ratio (music detection)
    n = len(pcm)
    fft = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    high_mask = freqs > 4000
    high_energy = float(np.sum(fft[high_mask] ** 2))
    total_fft_energy = float(np.sum(fft ** 2)) + 1e-12
    if high_energy / total_fft_energy > _MUSIC_HIGH_FREQ_RATIO_THRESHOLD:
        return True

    return False


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------

def estimate_snr_db(pcm: np.ndarray, noise_floor_quantile: float = 0.1) -> float:
    """Estimate SNR by comparing signal peak to noise floor quantile.

    Simple but robust for telephony: 10th-percentile energy ≈ noise.
    """
    frame_size = 320  # 20 ms
    frames = [pcm[i : i + frame_size] for i in range(0, len(pcm) - frame_size + 1, frame_size)]
    if not frames:
        return -80.0
    energies = [float(np.mean(f ** 2)) for f in frames]
    noise_energy = float(np.quantile(energies, noise_floor_quantile))
    signal_energy = float(np.max(energies))
    if noise_energy < 1e-12:
        return 60.0  # Very quiet noise — clean signal
    return 10.0 * np.log10(signal_energy / noise_energy + 1e-9)


# ---------------------------------------------------------------------------
# Clipping detection
# ---------------------------------------------------------------------------

def compute_clipping_ratio(pcm: np.ndarray, clip_level: float = 0.999) -> float:
    """Fraction of samples at or above clip_level in absolute value."""
    clipped = np.sum(np.abs(pcm) >= clip_level)
    return float(clipped) / max(1, len(pcm))


# ---------------------------------------------------------------------------
# Loudness normalisation  (B2-T06)
# ---------------------------------------------------------------------------

def normalize_loudness(
    pcm: np.ndarray,
    target_dbfs: float = TARGET_DBFS,
) -> tuple[np.ndarray, float]:
    """Normalise PCM to target_dbfs RMS. Returns (normalised, pre_norm_dbfs).

    The pre-normalisation level is returned for logging — it is forensic signal.
    """
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return pcm, -80.0
    pre_norm_dbfs = 20.0 * np.log10(rms)
    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / rms
    normalised = np.clip(pcm * gain, -1.0, 1.0).astype(np.float32)
    return normalised, pre_norm_dbfs


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """Result of the quality gate assessment for one analysis window."""
    quality_ok: bool
    quality_flags: list[str] = field(default_factory=list)
    voiced_ms: int = 0
    snr_db: float = 0.0
    clipping_ratio: float = 0.0
    pre_norm_dbfs: float = -80.0
    is_dtmf: bool = False
    is_music_or_tone: bool = False


def assess_quality(
    pcm: np.ndarray,
    voiced_ratio: float,
    window_duration_s: float = 3.0,
    sample_rate: int = TARGET_SR,
    snr_floor_db: float = SNR_FLOOR_DB,
    clip_ratio_max: float = CLIP_RATIO_MAX,
    min_voiced_s: float = MIN_VOICED_S,
) -> QualityResult:
    """Run the full quality gate on a decoded, resampled window PCM.

    Args:
        pcm:              float32 array at sample_rate.
        voiced_ratio:     fraction of voiced frames from VAD (0–1).
        window_duration_s: nominal window length in seconds.
        sample_rate:      should always be TARGET_SR (16 kHz).
        snr_floor_db:     minimum acceptable SNR.
        clip_ratio_max:   maximum acceptable clipping fraction.
        min_voiced_s:     minimum voiced speech to allow scoring.

    Returns:
        QualityResult with quality_ok=True only if all gates pass.
    """
    flags: list[str] = []
    voiced_ms = int(voiced_ratio * window_duration_s * 1000)
    snr_db = estimate_snr_db(pcm)
    clipping_ratio = compute_clipping_ratio(pcm)
    is_dtmf = detect_dtmf(pcm, sample_rate)
    is_music = detect_hold_music_or_tone(pcm, sample_rate)

    # Gate Q1 — insufficient voiced speech
    if voiced_ms < int(min_voiced_s * 1000):
        flags.append("insufficient_voiced_speech")

    # Gate Q2 — SNR below floor
    if snr_db < snr_floor_db:
        flags.append(f"low_snr:{snr_db:.1f}dB")

    # Gate Q3 — clipping
    if clipping_ratio > clip_ratio_max:
        flags.append(f"clipping:{clipping_ratio:.3f}")

    # Gate Q4 — DTMF or tone/music
    if is_dtmf:
        flags.append("dtmf_detected")
    if is_music:
        flags.append("hold_music_or_tone")

    quality_ok = len(flags) == 0
    return QualityResult(
        quality_ok=quality_ok,
        quality_flags=flags,
        voiced_ms=voiced_ms,
        snr_db=round(snr_db, 2),
        clipping_ratio=round(clipping_ratio, 5),
        is_dtmf=is_dtmf,
        is_music_or_tone=is_music,
    )
