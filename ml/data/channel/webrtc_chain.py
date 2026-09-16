"""Channel destruction: WebRTC-style processing + the full simulator (B3-T07).

``ChannelSimulator.simulate`` samples a random chain from a seed derived from
the utterance id only — never the label — and applies it.

Order mirrors a real call: room → noise → client DSP (AGC, suppression) →
bandwidth/codec → network loss/jitter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import istft, stft

from ml.data.channel.base import ChannelRecord, Transform, resample
from ml.data.channel.codecs import G711, FFmpegCodec, Narrowband, codec_available
from ml.data.channel.noise import AdditiveNoise
from ml.data.channel.packet_loss import Jitter, PacketLoss
from ml.data.channel.rir import RIRConvolve


@dataclass
class AGC:
    target_dbfs: float = -20.0
    frame_ms: int = 10
    attack: float = 0.2

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        flen = max(1, sr * self.frame_ms // 1000)
        y = x.astype(np.float32).copy()
        target = 10 ** (self.target_dbfs / 20)
        gain = 1.0
        for i in range(0, len(y), flen):
            rms = float(np.sqrt(np.mean(y[i : i + flen] ** 2)) + 1e-6)
            desired = min(target / rms, 10.0)
            gain += self.attack * (desired - gain)
            y[i : i + flen] *= gain
        rec.codec_chain.append("agc")
        return np.clip(y, -1, 1), sr


@dataclass
class NoiseSuppression:
    """Spectral subtraction stand-in for RNNoise / DeepFilterNet."""

    over_subtract: float = 1.5
    floor: float = 0.05

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        nper = 512 if sr >= 16000 else 256
        _, _, z = stft(x, fs=sr, nperseg=nper)
        mag, phase = np.abs(z), np.angle(z)
        noise = np.quantile(mag, 0.1, axis=1, keepdims=True)
        clean = np.maximum(mag - self.over_subtract * noise, self.floor * mag)
        _, y = istft(clean * np.exp(1j * phase), fs=sr, nperseg=nper)
        y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
        rec.codec_chain.append("ns")
        return y.astype(np.float32), sr


@dataclass
class ChannelConfig:
    p_rir: float = 0.3
    p_noise: float = 0.6
    p_agc: float = 0.3
    p_ns: float = 0.3
    p_loss: float = 0.5
    p_jitter: float = 0.2
    loss_rates: tuple[float, ...] = (0.01, 0.03, 0.05)
    # codec name -> weight. "clean" = wideband passthrough.
    codec_weights: dict[str, float] = field(
        default_factory=lambda: {
            "clean": 1.0,
            "g711u": 2.0,
            "g711a": 1.0,
            "opus": 2.0,
            "amr_nb": 1.0,
            "gsm": 0.5,
        }
    )
    opus_kbps: tuple[float, ...] = (6, 12, 16, 24)
    output_sr: int = 16000


def seed_for(utt_id: str, base_seed: int = 0) -> int:
    """Chain seed from utterance id only (I3: never from the label)."""
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{utt_id}".encode()).digest()[:8], "big")


class ChannelSimulator:
    def __init__(
        self,
        cfg: ChannelConfig | None = None,
        noise: AdditiveNoise | None = None,
        rir: RIRConvolve | None = None,
    ) -> None:
        self.cfg = cfg or ChannelConfig()
        self.noise = noise or AdditiveNoise()
        self.rir = rir or RIRConvolve()
        # Drop ffmpeg codecs the host cannot encode, rather than failing mid-corpus.
        self.codecs = {
            k: w
            for k, w in self.cfg.codec_weights.items()
            if k in ("clean", "g711u", "g711a") or codec_available(k)
        }

    def sample_chain(self, rng: np.random.Generator) -> list[Transform]:
        c = self.cfg
        chain: list[Transform] = []
        if rng.random() < c.p_rir:
            chain.append(self.rir)
        if rng.random() < c.p_noise:
            chain.append(self.noise)
        if rng.random() < c.p_agc:
            chain.append(AGC())
        if rng.random() < c.p_ns:
            chain.append(NoiseSuppression())
        names = sorted(self.codecs)
        w = np.array([self.codecs[n] for n in names], dtype=np.float64)
        codec = names[int(rng.choice(len(names), p=w / w.sum()))]
        if codec in ("g711u", "g711a"):
            chain += [Narrowband(), G711(law=codec[-1])]
        elif codec == "opus":
            chain.append(FFmpegCodec("opus", float(rng.choice(c.opus_kbps))))
        elif codec != "clean":
            chain.append(FFmpegCodec(codec))
        if rng.random() < c.p_loss:
            chain.append(PacketLoss(loss_rate=float(rng.choice(c.loss_rates))))
        if rng.random() < c.p_jitter:
            chain.append(Jitter())
        return chain

    def simulate(
        self, x: np.ndarray, sr: int, utt_id: str, base_seed: int = 0
    ) -> tuple[np.ndarray, int, ChannelRecord]:
        rng = np.random.default_rng(seed_for(utt_id, base_seed))
        rec = ChannelRecord()
        y = np.asarray(x, dtype=np.float32)
        for t in self.sample_chain(rng):
            y, sr = t.apply(y, sr, rng, rec)
        if sr != self.cfg.output_sr:
            y = resample(y, sr, self.cfg.output_sr)
            sr = self.cfg.output_sr
        return np.clip(y, -1, 1).astype(np.float32), sr, rec
