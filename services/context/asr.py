"""Streaming ASR with a rolling transcript buffer (B10-T01).

Backends (all lazy — nothing heavy is imported until selected):

* ``IndicConformerASR`` — AI4Bharat IndicConformer via NVIDIA NeMo, for the 12 Indic languages
* ``WhisperASR``        — Whisper large-v3 via ``faster-whisper`` (English + fallback)
* ``ScriptedASR``       — replays a known transcript by time; for tests, demos and replay

Streaming model: audio is fed in chunks; the backend re-decodes a sliding
buffer (``decode_every_s``) and emits *partial* segments; segments that end
more than ``finalize_after_s`` before the buffer end become *final*. The
``RollingTranscript`` keeps final segments for ``keep_s`` seconds plus the
current partial, and exposes word-level tokens with timestamps for B6
(disfluency) and B8 (challenge verification).

Runs asynchronously to the audio path (I10); the context service never blocks scoring.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SR = 16000


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None
    final: bool = True
    confidence: float = 1.0
    words: list[tuple[str, int, int]] = field(default_factory=list)  # (word, start_ms, end_ms)

    def tokens(self) -> list[tuple[str, int, int]]:
        if self.words:
            return self.words
        parts = self.text.split()
        if not parts:
            return []
        step = (self.end_ms - self.start_ms) / len(parts)
        return [
            (w, int(self.start_ms + i * step), int(self.start_ms + (i + 1) * step))
            for i, w in enumerate(parts)
        ]


class StreamingASR(Protocol):
    name: str

    def feed(self, pcm: np.ndarray, t_ms: int) -> list[Segment]: ...


class RollingTranscript:
    def __init__(self, keep_s: float = 120.0) -> None:
        self.keep_ms = int(keep_s * 1000)
        self.final: deque[Segment] = deque()
        self.partial: Segment | None = None

    def add(self, segments: list[Segment]) -> None:
        for s in segments:
            if s.final:
                if not self.final or s.start_ms >= self.final[-1].end_ms - 50:
                    self.final.append(s)
                self.partial = None
            else:
                self.partial = s
        if self.final:
            horizon = self.final[-1].end_ms - self.keep_ms
            while self.final and self.final[0].end_ms < horizon:
                self.final.popleft()

    def segments(self, include_partial: bool = True) -> list[Segment]:
        out = list(self.final)
        if include_partial and self.partial is not None:
            out.append(self.partial)
        return out

    def text(self, include_partial: bool = True) -> str:
        return " ".join(s.text for s in self.segments(include_partial)).strip()

    def tokens(self, since_ms: int = 0) -> list[tuple[str, int, int]]:
        return [t for s in self.segments() for t in s.tokens() if t[1] >= since_ms]


# ---------------------------------------------------------------- backends
class ScriptedASR:
    """Emits pre-defined segments once the fed audio clock passes their end time."""

    name = "scripted"

    def __init__(self, segments: list[Segment]) -> None:
        self._pending = sorted(segments, key=lambda s: s.start_ms)
        self._clock_ms = 0

    def feed(self, pcm: np.ndarray, t_ms: int) -> list[Segment]:
        self._clock_ms = max(self._clock_ms, t_ms + int(len(pcm) / SR * 1000))
        ready = [s for s in self._pending if s.end_ms <= self._clock_ms]
        self._pending = [s for s in self._pending if s.end_ms > self._clock_ms]
        return ready


class _BufferedASR:
    """Shared sliding-buffer logic for offline-decoder backends."""

    name = "buffered"

    def __init__(
        self, decode_every_s: float = 2.0, window_s: float = 20.0, finalize_after_s: float = 3.0
    ) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start_ms = 0
        self._since_decode = 0
        self._every = int(decode_every_s * SR)
        self._window = int(window_s * SR)
        self._finalize_ms = int(finalize_after_s * 1000)
        self._emitted_final_ms = 0

    def _decode(self, audio: np.ndarray, offset_ms: int) -> list[Segment]:
        raise NotImplementedError

    def feed(self, pcm: np.ndarray, t_ms: int) -> list[Segment]:
        if not len(self._buf):
            self._buf_start_ms = t_ms
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.float32)])
        self._since_decode += len(pcm)
        if len(self._buf) > self._window:
            drop = len(self._buf) - self._window
            self._buf = self._buf[drop:]
            self._buf_start_ms += int(drop / SR * 1000)
        if self._since_decode < self._every:
            return []
        self._since_decode = 0
        end_ms = self._buf_start_ms + int(len(self._buf) / SR * 1000)
        out = []
        for s in self._decode(self._buf, self._buf_start_ms):
            if s.end_ms <= self._emitted_final_ms:
                continue
            s.final = s.end_ms < end_ms - self._finalize_ms
            if s.final:
                self._emitted_final_ms = s.end_ms
            out.append(s)
        return out


class WhisperASR(_BufferedASR):
    name = "whisper-large-v3"

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        language: str | None = None,
        **kw: float,
    ) -> None:
        super().__init__(**kw)
        from faster_whisper import WhisperModel  # lazy, optional dependency

        self._model = WhisperModel(model, device=device, compute_type="int8")
        self._language = language

    def _decode(self, audio: np.ndarray, offset_ms: int) -> list[Segment]:
        segs, info = self._model.transcribe(
            audio, language=self._language, word_timestamps=True, vad_filter=True
        )
        out = []
        for s in segs:
            words = [
                (w.word.strip(), offset_ms + int(w.start * 1000), offset_ms + int(w.end * 1000))
                for w in (s.words or [])
            ]
            out.append(
                Segment(
                    offset_ms + int(s.start * 1000),
                    offset_ms + int(s.end * 1000),
                    s.text.strip(),
                    info.language,
                    confidence=float(np.exp(s.avg_logprob)),
                    words=words,
                )
            )
        return out


class IndicConformerASR(_BufferedASR):
    name = "indicconformer"

    def __init__(
        self,
        language: str,
        model: str = "ai4bharat/indicconformer_stt_multi_hybrid_rnnt_600m",
        **kw: float,
    ) -> None:
        super().__init__(**kw)
        import nemo.collections.asr as nemo_asr  # lazy, optional dependency

        self._model = nemo_asr.models.ASRModel.from_pretrained(model).eval()
        self._language = language

    def _decode(self, audio: np.ndarray, offset_ms: int) -> list[Segment]:
        text = self._model.transcribe([audio], language_id=self._language)[0]
        text = text if isinstance(text, str) else getattr(text, "text", str(text))
        dur = int(len(audio) / SR * 1000)
        return (
            [Segment(offset_ms, offset_ms + dur, text.strip(), self._language)]
            if text.strip()
            else []
        )


def build_asr(name: str, language: str | None = None) -> StreamingASR:
    if name in ("whisper", "whisper-large-v3"):
        return WhisperASR(language=language)
    if name in ("indicconformer", "ai4bharat/indicconformer"):
        return IndicConformerASR(language or "hi")
    raise ValueError(f"unknown ASR backend {name!r}")
