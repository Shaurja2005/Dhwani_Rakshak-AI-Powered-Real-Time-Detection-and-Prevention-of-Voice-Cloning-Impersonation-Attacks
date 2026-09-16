"""Channel destruction: narrowband resampling and codecs (B3-T07).

* G.711 µ-law / A-law: pure numpy companding + 8-bit quantisation.
* Opus, AMR-NB, GSM, G.722: round-trip through ffmpeg over pipes.

G.729 and EVS have no open encoder in stock ffmpeg; they are left out rather
than faked. Add them via a dedicated container if needed.
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 - fixed argv, no shell
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ml.data.channel.base import ChannelRecord, resample, to_float32

MU = 255.0
A = 87.6


class CodecUnavailable(RuntimeError):  # noqa: N818
    pass


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    x = to_float32(x)
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return (np.sign(q) * np.expm1(np.abs(q) * np.log1p(MU)) / MU).astype(np.float32)


def alaw_roundtrip(x: np.ndarray) -> np.ndarray:
    x = to_float32(x)
    ax = np.abs(x)
    small = ax < 1 / A
    y = np.where(
        small, A * ax / (1 + np.log(A)), (1 + np.log(np.maximum(A * ax, 1e-12))) / (1 + np.log(A))
    )
    y = np.sign(x) * y
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    aq = np.abs(q)
    small = aq < 1 / (1 + np.log(A))
    inv = np.where(small, aq * (1 + np.log(A)) / A, np.exp(aq * (1 + np.log(A)) - 1) / A)
    return (np.sign(q) * inv).astype(np.float32)


@dataclass
class Narrowband:
    """Downsample to 8 kHz (telephony bandwidth)."""

    target_sr: int = 8000

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        rec.codec_chain.append(f"resample:{self.target_sr}")
        return resample(x, sr, self.target_sr), self.target_sr


@dataclass
class G711:
    law: str = "u"  # "u" | "a"

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        if sr != 8000:
            x, sr = resample(x, sr, 8000), 8000
            rec.codec_chain.append("resample:8000")
        rec.codec_chain.append(f"codec:g711{self.law}")
        fn = mulaw_roundtrip if self.law == "u" else alaw_roundtrip
        return fn(x), sr


# name -> (ffmpeg encoder, container, sample rate)
FFMPEG_CODECS: dict[str, tuple[str, str, int]] = {
    "opus": ("libopus", "ogg", 16000),
    "amr_nb": ("libopencore_amrnb", "amr", 8000),
    "gsm": ("libgsm", "gsm", 8000),
    "g722": ("g722", "g722", 16000),
}


@lru_cache(maxsize=1)
def _ffmpeg_encoders() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return ""
    out = subprocess.run(  # noqa: S603
        [exe, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    return out.stdout


def codec_available(name: str) -> bool:
    enc = FFMPEG_CODECS[name][0]
    return f" {enc} " in _ffmpeg_encoders()


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise CodecUnavailable("ffmpeg not found on PATH")
    p = subprocess.run(  # noqa: S603
        [exe, "-hide_banner", "-loglevel", "error", *args],
        input=data,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0:
        raise CodecUnavailable(p.stderr.decode(errors="replace").strip())
    return p.stdout


@dataclass
class FFmpegCodec:
    name: str
    bitrate_kbps: float | None = None

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator, rec: ChannelRecord
    ) -> tuple[np.ndarray, int]:
        if not codec_available(self.name):
            raise CodecUnavailable(f"ffmpeg encoder for {self.name} not available")
        encoder, fmt, codec_sr = FFMPEG_CODECS[self.name]
        if sr != codec_sr:
            x = resample(x, sr, codec_sr)
            rec.codec_chain.append(f"resample:{codec_sr}")
        pcm = (to_float32(x) * 32767).astype("<i2").tobytes()
        raw = ["-f", "s16le", "-ar", str(codec_sr), "-ac", "1"]
        enc_args = [*raw, "-i", "pipe:0", "-c:a", encoder]
        if self.name == "amr_nb":
            # AMR-NB only supports fixed modes; pick the nearest.
            enc_args += ["-b:a", f"{int((self.bitrate_kbps or 12.2) * 1000)}"]
        elif self.bitrate_kbps:
            enc_args += ["-b:a", f"{int(self.bitrate_kbps * 1000)}"]
        encoded = _ffmpeg([*enc_args, "-f", fmt, "pipe:1"], pcm)
        decoded = _ffmpeg(["-f", fmt, "-i", "pipe:0", *raw, "pipe:1"], encoded)
        y = np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768
        # Codecs add priming samples / padding; keep length stable.
        n = len(x)
        y = np.pad(y, (0, max(0, n - len(y))))[:n]
        tag = f"codec:{self.name}" + (f"@{self.bitrate_kbps:g}k" if self.bitrate_kbps else "")
        rec.codec_chain.append(tag)
        return y, codec_sr
