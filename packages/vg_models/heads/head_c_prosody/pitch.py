"""F0 extraction and contour dynamics (B6-T01).

YIN-style pitch tracker (cumulative-mean-normalised difference computed from an
FFT autocorrelation, fully vectorised) on 32 ms frames with a 10 ms hop.

Features:
* f0 median / range (semitones, 5th–95th percentile)
* slope entropy — entropy of frame-to-frame pitch slopes; TTS contours are often
  smoother / more stereotyped (lower entropy)
* declination — median within-phrase pitch trend (st/s); natural phrases drift down
* jitter — relative variation of consecutive periods (frame-level approximation
  of cycle-to-cycle jitter, adequate for 10 ms frames)
* shimmer — mean absolute dB change of frame peak amplitude between voiced frames
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME = 512
HOP = 160
FMIN, FMAX = 60.0, 400.0
_LAG_MIN, _LAG_MAX = int(SR / FMAX), int(SR / FMIN)
_WIN = np.hanning(FRAME)


def frames(x: np.ndarray, n: int = FRAME, hop: int = HOP) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    count = 1 + (len(x) - n) // hop
    return x[np.arange(n)[None, :] + hop * np.arange(count)[:, None]]


@dataclass
class PitchTrack:
    f0_hz: np.ndarray  # [T], 0 when unvoiced
    voiced: np.ndarray  # [T] bool
    frame_db: np.ndarray  # [T]
    peak_db: np.ndarray  # [T]

    @property
    def semitones(self) -> np.ndarray:
        st = np.zeros_like(self.f0_hz)
        st[self.voiced] = 12 * np.log2(self.f0_hz[self.voiced] / 100.0)
        return st


def track(x: np.ndarray, threshold: float = 0.2) -> PitchTrack:
    fr = frames(np.asarray(x, dtype=np.float64))
    fr = fr - fr.mean(axis=1, keepdims=True)
    power = (fr**2).mean(axis=1) + 1e-12
    frame_db = 10 * np.log10(power)
    peak_db = 20 * np.log10(np.abs(fr).max(axis=1) + 1e-9)

    spec = np.fft.rfft(fr, n=2 * FRAME, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, : _LAG_MAX + 2]
    # Energy of the lagged segments shrinks with tau; approximate with a linear taper.
    taus = np.arange(acf.shape[1])
    taper = 1 - taus / FRAME
    diff = acf[:, :1] * (taper + taper) - 2 * acf  # ≈ d(tau) = r_t(0) + r_{t+tau}(0) - 2 r_t(tau)
    diff[:, 0] = 0
    cum = np.cumsum(diff[:, 1:], axis=1) / np.arange(1, diff.shape[1])
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] / (cum + 1e-12)

    search = cmnd[:, _LAG_MIN : _LAG_MAX + 1]
    below = search < threshold
    first = np.where(below.any(axis=1), below.argmax(axis=1), search.argmin(axis=1))
    # Walk to the local minimum after the first threshold crossing.
    idx = first.copy()
    for _ in range(8):
        nxt = np.minimum(idx + 1, search.shape[1] - 1)
        better = search[np.arange(len(idx)), nxt] < search[np.arange(len(idx)), idx]
        idx = np.where(better, nxt, idx)
    best = search[np.arange(len(idx)), idx]
    lag = (idx + _LAG_MIN).astype(np.float64)

    # Parabolic interpolation for sub-sample lag.
    i = np.clip(idx, 1, search.shape[1] - 2)
    a, b, c = (search[np.arange(len(i)), i + k] for k in (-1, 0, 1))
    denom = a - 2 * b + c
    safe = np.where(np.abs(denom) > 1e-9, denom, 1.0)
    lag = np.where(np.abs(denom) > 1e-9, i + _LAG_MIN + 0.5 * (a - c) / safe, lag)

    # Energy gate relative to the loudest frame (a percentile floor fails when
    # the window has no silence at all), plus an absolute -70 dBFS floor.
    loud = (frame_db > frame_db.max() - 35.0) & (frame_db > -70.0)
    voiced = (best < threshold + 0.15) & loud
    f0 = np.where(voiced, SR / np.maximum(lag, 1.0), 0.0)
    voiced &= (f0 >= FMIN) & (f0 <= FMAX)
    f0[~voiced] = 0.0
    return PitchTrack(f0_hz=f0, voiced=voiced, frame_db=frame_db, peak_db=peak_db)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


def contour_features(p: PitchTrack) -> dict[str, float]:
    nan = float("nan")
    v = p.voiced
    out = {
        "f0_voiced_ratio": float(v.mean()),
        "f0_median_st": nan,
        "f0_range_st": nan,
        "f0_slope_entropy": nan,
        "f0_declination_st_per_s": nan,
        "jitter_rel": nan,
        "shimmer_db": nan,
    }
    if v.sum() < 10:
        return out
    st = p.semitones
    out["f0_median_st"] = float(np.median(st[v]))
    out["f0_range_st"] = float(np.percentile(st[v], 95) - np.percentile(st[v], 5))

    both = v[1:] & v[:-1]
    if both.sum() >= 5:
        slopes = (st[1:] - st[:-1])[both]
        hist, _ = np.histogram(np.clip(slopes, -2, 2), bins=20, range=(-2, 2))
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        out["f0_slope_entropy"] = float(-(prob * np.log(prob)).sum() / np.log(20))
        periods = 1.0 / np.where(v, p.f0_hz, 1.0)
        dper = np.abs(periods[1:] - periods[:-1])[both]
        out["jitter_rel"] = float(dper.mean() / periods[v].mean())
        out["shimmer_db"] = float(np.abs(p.peak_db[1:] - p.peak_db[:-1])[both].mean())

    decl = []
    for a, b in _runs(v):
        if b - a >= 20:  # >= 200 ms phrase
            t = np.arange(b - a) * HOP / SR
            decl.append(np.polyfit(t, st[a:b], 1)[0])
    if decl:
        out["f0_declination_st_per_s"] = float(np.median(decl))
    return out
