"""Head B spectral feature extractors (B5-T01).

All features come from one shared STFT per window (plus one extra FFT for the
group delay), so the whole extractor stays within a few milliseconds on CPU.

Returned as an ordered ``dict[str, float]`` so the classifier has a stable
feature vector *and* explanations can refer to features by name.

* LFCC            — linear-frequency cepstra (mean + std), sensitive to HF vocoder artifacts
* CQT (approx.)   — log-spaced band energies pooled from the STFT (constant-Q-like resolution)
* MGD             — modified group delay per band (phase information magnitude features miss)
* LTAS            — long-term average spectrum in octave-ish bands + spectral tilt
* roll-off        — 85% / 95% energy roll-off (mean, std)
* bicoherence     — mean quadratic phase coupling (nonlinear generation artifacts)
* low-band breath — energy below 300 Hz in unvoiced frames
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
N_FFT = 512
HOP = 160
EPS = 1e-10

_WIN = np.hanning(N_FFT).astype(np.float32)
_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
N_LFCC = 20
N_CQT = 24
N_MGD = 8
N_LTAS = 12


def _frames(x: np.ndarray, n: int = N_FFT, hop: int = HOP) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    count = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def _log_band_edges(n_bands: int, fmin: float, fmax: float) -> np.ndarray:
    edges_hz = np.geomspace(fmin, fmax, n_bands + 1)
    return np.clip(np.searchsorted(_FREQS, edges_hz), 1, len(_FREQS) - 1)


_CQT_EDGES = _log_band_edges(N_CQT, 60.0, 7900.0)
_LTAS_EDGES = _log_band_edges(N_LTAS, 50.0, 7900.0)
_MGD_EDGES = np.linspace(1, len(_FREQS) - 1, N_MGD + 1).astype(int)
_LFCC_FB_EDGES = np.linspace(0, len(_FREQS), 41).astype(int)
_DCT = np.cos(np.pi * np.arange(N_LFCC)[:, None] * (2 * np.arange(40)[None, :] + 1) / 80)


def _band_pool(spec: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """spec [T, F] -> [T, len(edges)-1] mean power per band."""
    csum = np.concatenate([np.zeros((spec.shape[0], 1)), np.cumsum(spec, axis=1)], axis=1)
    lo, hi = edges[:-1], np.maximum(edges[1:], edges[:-1] + 1)
    return (csum[:, hi] - csum[:, lo]) / (hi - lo)


@dataclass
class Spectra:
    power: np.ndarray  # [T, F]
    phase_frames: np.ndarray  # windowed frames [T, N]
    frame_db: np.ndarray  # [T]
    voiced: np.ndarray  # [T] bool


def compute_spectra(pcm: np.ndarray) -> Spectra:
    fr = _frames(pcm.astype(np.float32)) * _WIN
    power = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + EPS
    frame_db = 10 * np.log10(power.mean(axis=1))
    floor = np.percentile(frame_db, 10)
    voiced = frame_db > floor + 12.0
    return Spectra(power=power, phase_frames=fr, frame_db=frame_db, voiced=voiced)


def lfcc(s: Spectra) -> np.ndarray:
    fb = np.log(_band_pool(s.power, _LFCC_FB_EDGES))  # [T, 40]
    ceps = fb @ _DCT.T  # [T, 20]
    return np.concatenate([ceps.mean(0), ceps.std(0)])


def cqt_bands(s: Spectra) -> np.ndarray:
    frames = s.power[s.voiced] if s.voiced.sum() >= 5 else s.power
    bands = 10 * np.log10(_band_pool(frames, _CQT_EDGES).mean(0))
    return bands - bands.max()


def modified_group_delay(s: Spectra, alpha: float = 0.4, gamma: float = 0.9) -> np.ndarray:
    frames = s.phase_frames[s.voiced] if s.voiced.sum() >= 5 else s.phase_frames
    frames = frames[:: max(1, len(frames) // 40)]  # cap cost: <= ~40 frames
    n = np.arange(N_FFT, dtype=np.float32)
    x = np.fft.rfft(frames, axis=1)
    y = np.fft.rfft(frames * n, axis=1)
    mag = np.abs(x) ** 2
    # Cepstrally smoothed magnitude replaced by a cheap moving average.
    k = np.ones(9) / 9
    smooth = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, mag) + EPS
    tau = (x.real * y.real + x.imag * y.imag) / (smooth**gamma)
    mgd = np.sign(tau) * np.abs(tau) ** alpha
    return np.array(
        [np.mean(mgd[:, a:b]) for a, b in zip(_MGD_EDGES[:-1], _MGD_EDGES[1:], strict=True)]
    )


def ltas(s: Spectra) -> tuple[np.ndarray, float]:
    bands = 10 * np.log10(_band_pool(s.power, _LTAS_EDGES).mean(0))
    centres = np.log2(np.sqrt(_FREQS[_LTAS_EDGES[:-1]] * _FREQS[_LTAS_EDGES[1:]]) + 1)
    tilt = float(np.polyfit(centres, bands, 1)[0])  # dB per octave
    return bands - bands.mean(), tilt


def rolloff(s: Spectra) -> np.ndarray:
    c = np.cumsum(s.power, axis=1)
    total = c[:, -1:]
    r85 = _FREQS[np.argmax(c >= 0.85 * total, axis=1)]
    r95 = _FREQS[np.argmax(c >= 0.95 * total, axis=1)]
    v = s.voiced if s.voiced.sum() >= 5 else np.ones_like(s.voiced)
    return np.array([r85[v].mean(), r85[v].std(), r95[v].mean(), r95[v].std()]) / (SR / 2)


def bicoherence(s: Spectra, n_bins: int = 32, max_frames: int = 24) -> float:
    """Mean bicoherence over a coarse (f1, f2) grid up to 4 kHz."""
    frames = s.phase_frames[s.voiced] if s.voiced.sum() >= 5 else s.phase_frames
    frames = frames[:: max(1, len(frames) // max_frames)]
    spec = np.fft.rfft(frames, axis=1)[:, : 2 * n_bins * 4 : 4]  # every 4th bin, up to ~8 kHz
    i = np.arange(n_bins)
    f1, f2 = np.meshgrid(i, i, indexing="ij")
    mask = f1 + f2 < spec.shape[1]
    a, b = f1[mask], f2[mask]
    triple = spec[:, a] * spec[:, b] * np.conj(spec[:, a + b])
    num = np.abs(triple.mean(0)) ** 2
    den = (np.abs(spec[:, a] * spec[:, b]) ** 2).mean(0) * (np.abs(spec[:, a + b]) ** 2).mean(
        0
    ) + EPS
    return float(np.mean(num / den))


def low_band_breath_ratio(s: Spectra) -> float:
    """Energy <300 Hz relative to total, in unvoiced but non-silent frames (breaths, lip noise)."""
    floor = np.percentile(s.frame_db, 10)
    breathy = (~s.voiced) & (s.frame_db > floor + 3.0)
    if breathy.sum() < 3:
        return float("nan")
    lo = s.power[breathy][:, _FREQS < 300].sum(1)
    return float(np.median(lo / s.power[breathy].sum(1)))


def extract(pcm: np.ndarray, s: Spectra | None = None) -> dict[str, float]:
    s = s or compute_spectra(pcm)
    feats: dict[str, float] = {}
    for i, v in enumerate(lfcc(s)):
        feats[f"lfcc_{'mean' if i < N_LFCC else 'std'}_{i % N_LFCC}"] = float(v)
    for i, v in enumerate(cqt_bands(s)):
        feats[f"cqt_{i}"] = float(v)
    for i, v in enumerate(modified_group_delay(s)):
        feats[f"mgd_{i}"] = float(v)
    bands, tilt = ltas(s)
    for i, v in enumerate(bands):
        feats[f"ltas_{i}"] = float(v)
    feats["ltas_tilt_db_per_oct"] = tilt
    for name, v in zip(
        ("rolloff85_mean", "rolloff85_std", "rolloff95_mean", "rolloff95_std"),
        rolloff(s),
        strict=True,
    ):
        feats[name] = float(v)
    feats["bicoherence"] = bicoherence(s)
    feats["breath_low_band_ratio"] = low_band_breath_ratio(s)
    feats["voiced_frame_ratio"] = float(s.voiced.mean())
    return feats
