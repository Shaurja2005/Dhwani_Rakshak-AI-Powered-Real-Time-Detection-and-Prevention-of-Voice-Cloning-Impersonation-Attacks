"""Laundering and adversarial attacks against our own detector (B15-T08).

Two families:

* **Laundering** (black-box, cheap, what a fraudster does first): re-encode
  through a codec, speed perturbation, pitch shift, additive noise. Applied to
  **spoof audio only** — the attacker controls the fake, not the bona fide calls.
* **Adversarial** (white-box, worst case, needs a differentiable model):
  - ``pgd`` — L∞-bounded projected gradient attack pushing spoofs towards bona fide;
  - ``filter_attack`` — Malacopula-style: one *universal* short FIR filter learnt
    against the detector and then applied to every spoof. Unlike PGD noise it is a
    plausible "voice processing" step and transfers across utterances.

Channel sweeps (``channel_sweep``) apply the same transform to **both** classes —
that is a robustness curve, not an attack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import resample_poly

from ml.data.channel.base import ChannelRecord
from ml.data.channel.codecs import G711, CodecUnavailable, FFmpegCodec, codec_available
from ml.data.channel.noise import colored_noise, mix_at_snr

SR = 16000
Transform = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def codec_roundtrip(name: str) -> Transform:
    """Encode + decode through a telephony/VoIP codec, back to 16 kHz."""

    def run(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rec = ChannelRecord()
        codec = G711(name[-1]) if name in ("g711u", "g711a") else FFmpegCodec(name)
        y, sr = codec.apply(x.astype(np.float32), SR, rng, rec)
        if sr != SR:
            y = resample_poly(y, SR, sr).astype(np.float32)
        return _fit(y, len(x))

    return run


def available_codecs(names: list[str]) -> list[str]:
    return [n for n in names if n in ("g711u", "g711a") or codec_available(n)]


def speed(factor: float) -> Transform:
    """Play faster/slower (changes pitch and tempo together)."""
    up, down = _ratio(1 / factor)

    def run(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return resample_poly(x, up, down).astype(np.float32)

    return run


def pitch_shift(semitones: float) -> Transform:
    """Pitch shift keeping duration: resample, then overlap-add time-stretch back."""
    f = 2 ** (semitones / 12)

    def run(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        up, down = _ratio(1 / f)
        y = resample_poly(x, up, down)
        return _fit(_ola_stretch(y, len(x)), len(x))

    return run


def add_noise(snr_db: float, color: float = 1.0) -> Transform:
    def run(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return mix_at_snr(x.astype(np.float32), colored_noise(len(x), rng, color), snr_db)

    return run


def _ratio(r: float, max_den: int = 100) -> tuple[int, int]:
    from fractions import Fraction

    fr = Fraction(r).limit_denominator(max_den)
    return fr.numerator, fr.denominator


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def _ola_stretch(y: np.ndarray, n_out: int, frame: int = 1024) -> np.ndarray:
    hop_out = frame // 4
    n_frames = max(1, (n_out - frame) // hop_out + 1)
    hop_in = (len(y) - frame) / max(1, n_frames - 1) if len(y) > frame else 0
    win = np.hanning(frame)
    out = np.zeros(n_out + frame)
    norm = np.zeros(n_out + frame)
    for i in range(n_frames):
        s = int(round(i * hop_in))
        seg = y[s : s + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        out[i * hop_out : i * hop_out + frame] += seg * win
        norm[i * hop_out : i * hop_out + frame] += win
    return (out / np.maximum(norm, 1e-6))[:n_out]


def laundering_suite(codecs: list[str]) -> dict[str, Transform]:
    suite: dict[str, Transform] = {
        "speed_0.9": speed(0.9),
        "speed_1.1": speed(1.1),
        "pitch_-2st": pitch_shift(-2),
        "pitch_+2st": pitch_shift(2),
        "noise_pink_20dB": add_noise(20, 1.0),
        "noise_pink_10dB": add_noise(10, 1.0),
    }
    for c in available_codecs(codecs):
        suite[f"reencode_{c}"] = codec_roundtrip(c)
    return suite


# ---------------------------------------------------------------- white-box attacks
ScoreFn = Callable[[torch.Tensor], torch.Tensor]  # [B, T] -> bona fide score [B]


@dataclass
class AttackReport:
    name: str
    n: int
    mean_score_before: float
    mean_score_after: float
    snr_db: float  # attack perturbation energy relative to the signal


def pgd(
    score_fn: ScoreFn,
    wav: torch.Tensor,
    eps: float = 0.002,
    steps: int = 10,
    alpha: float | None = None,
) -> torch.Tensor:
    """L∞ PGD maximising the bona fide score of ``wav`` [B, T]."""
    alpha = alpha if alpha is not None else eps / 4
    x0 = wav.detach()
    x = x0.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        loss = score_fn(x).sum()
        (grad,) = torch.autograd.grad(loss, x)
        with torch.no_grad():
            x = x + alpha * grad.sign()
            x = torch.clamp(torch.min(torch.max(x, x0 - eps), x0 + eps), -1, 1)
    return x.detach()


def learn_universal_filter(
    score_fn: ScoreFn,
    spoofs: torch.Tensor,
    taps: int = 64,
    steps: int = 60,
    lr: float = 5e-3,
    max_gain_db: float = 6.0,
    batch: int = 8,
) -> torch.Tensor:
    """Malacopula-style universal FIR filter that makes spoofs score as bona fide.

    Starts at identity; the frequency response is kept within ±``max_gain_db`` of
    unity so the output still sounds like speech. Gradients are accumulated over
    ``batch``-sized chunks (same result as one big batch, but fits a laptop GPU).
    """
    h = torch.zeros(taps)
    h[0] = 1.0
    h.requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)
    limit = 10 ** (max_gain_db / 20)
    n = len(spoofs)
    for _ in range(steps):
        opt.zero_grad()
        for i in range(0, n, batch):
            (-score_fn(apply_filter(spoofs[i : i + batch], h)).sum() / n).backward()
        mag = torch.abs(torch.fft.rfft(h, 512))
        (10 * torch.relu(mag - limit).mean() + 10 * torch.relu(1 / limit - mag).mean()).backward()
        opt.step()
    return h.detach()


def apply_filter(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1] + len(h) - 1
    y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)[..., : x.shape[-1]]
    return torch.clamp(y, -1, 1)


def perturbation_snr(clean: np.ndarray, attacked: np.ndarray) -> float:
    d = attacked - clean
    return float(10 * np.log10((np.sum(clean**2) + 1e-12) / (np.sum(d**2) + 1e-12)))


__all__ = [
    "AttackReport",
    "CodecUnavailable",
    "add_noise",
    "apply_filter",
    "available_codecs",
    "codec_roundtrip",
    "laundering_suite",
    "learn_universal_filter",
    "perturbation_snr",
    "pgd",
    "pitch_shift",
    "speed",
]
