"""services/ingest/adapters/replay_wav.py — File/WAV replay adapter.

B1-T01: Replays a WAV file in real time with configurable jitter and
packet-loss simulation.  This is the primary development tool — nobody
should need a PBX to develop on the pipeline.

Design:
- Reads the WAV file, converts to PCM 16-bit frames of CHUNK_MS ms each.
- Publishes AudioChunk events onto the bus at wall-clock pace (optionally
  faster/slower than real-time via speed_factor).
- Jitter: each chunk sleeps for N(nominal_ms, jitter_ms) before publish.
- Packet loss: each chunk is dropped with probability loss_rate, simulating
  an unreliable transport. The seq counter still advances (gap = loss).
- Publishes CallMetadata at session start.

Usage::

    from services.ingest.adapters.replay_wav import WavReplayAdapter, WavReplayConfig

    config = WavReplayConfig(
        wav_path=Path("tests/fixtures/sample.wav"),
        tenant_id="demo",
        loss_rate=0.03,
        jitter_ms=10,
    )
    adapter = WavReplayAdapter(config)
    async for metadata, chunk in adapter.stream():
        await bus.publish(...)
"""
from __future__ import annotations

import asyncio
import random
import struct
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from packages.vg_core.logging import get_logger
from packages.vg_core.models import (
    AudioChunk,
    AudioEncoding,
    CallMetadata,
    Channel,
    ConsentBasis,
)
from services.ingest.metadata import build_call_metadata

log = get_logger(__name__)

# Size of each emitted chunk in milliseconds
CHUNK_MS: int = 20  # 20 ms = 320 samples @ 16 kHz  (standard RTP ptime)


@dataclass
class WavReplayConfig:
    """Configurable parameters for the WAV replay adapter."""

    wav_path: Path
    tenant_id: str = "demo"
    session_id: Optional[str] = None
    direction: str = "inbound"
    caller_number: Optional[str] = None
    callee_number: Optional[str] = None
    claimed_identity_id: Optional[str] = None
    language_hint: Optional[str] = None
    consent_basis: ConsentBasis = ConsentBasis.LEGITIMATE_USE
    shadow_mode: bool = True

    # Playback control
    speed_factor: float = 1.0   # 1.0 = real-time, 2.0 = 2× faster
    chunk_ms: int = CHUNK_MS

    # Network simulation
    loss_rate: float = 0.0      # [0, 1]: probability of dropping a packet
    jitter_ms: float = 0.0      # std-dev of Gaussian jitter in ms (0 = none)
    seed: Optional[int] = None  # RNG seed for reproducibility


class WavReplayAdapter:
    """Streams a WAV file as AudioChunk events, simulating real-time telephony.

    Supports:
    - Any sample rate (re-read at native SR; downstream B2 does resampling)
    - Mono and stereo (stereo is mixed to mono at byte level)
    - 8-bit µ-law, 8-bit A-law, 16-bit PCM (via Python standard wave module)
    - Configurable chunk size, speed factor, jitter, packet loss
    """

    def __init__(self, config: WavReplayConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def _make_session_id(self) -> str:
        return self.config.session_id or str(uuid.uuid4())

    def _open_wav(self) -> tuple[wave.Wave_read, int, int, int, str]:
        """Open the WAV file and return (reader, n_frames, sample_rate, n_channels, encoding)."""
        path = self.config.wav_path
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {path}")
        wf = wave.open(str(path), "rb")
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()  # bytes per sample
        n_frames = wf.getnframes()

        if sample_width == 1:
            encoding_str = "mulaw"
        elif sample_width == 2:
            encoding_str = "pcm_s16le"
        else:
            wf.close()
            raise ValueError(f"Unsupported sample width: {sample_width} bytes in {path}")

        return wf, n_frames, sample_rate, n_channels, encoding_str

    def _mix_to_mono(self, frames: bytes, n_channels: int, sample_width: int) -> bytes:
        """Mix multi-channel audio to mono by averaging channels."""
        if n_channels == 1:
            return frames
        if sample_width == 2:
            n_samples = len(frames) // (sample_width * n_channels)
            mono = bytearray()
            for i in range(n_samples):
                offset = i * sample_width * n_channels
                samples = [
                    struct.unpack_from("<h", frames, offset + c * sample_width)[0]
                    for c in range(n_channels)
                ]
                avg = int(sum(samples) / n_channels)
                mono += struct.pack("<h", avg)
            return bytes(mono)
        # 1-byte µ-law: take first channel
        n_samples = len(frames) // n_channels
        return bytes(frames[i * n_channels] for i in range(n_samples))

    def _compute_delay_s(self, chunk_duration_s: float) -> float:
        """Compute the sleep duration for this chunk, with optional Gaussian jitter."""
        nominal = chunk_duration_s / self.config.speed_factor
        if self.config.jitter_ms > 0:
            jitter = self._rng.gauss(0, self.config.jitter_ms / 1000.0)
            return max(0.0, nominal + jitter)
        return nominal

    def _should_drop(self) -> bool:
        """Return True if this packet should be simulated as lost."""
        if self.config.loss_rate <= 0.0:
            return False
        return self._rng.random() < self.config.loss_rate

    def _get_sample_rate(self) -> int:
        """Peek at the WAV header for sample rate."""
        wf = wave.open(str(self.config.wav_path), "rb")
        sr = wf.getframerate()
        wf.close()
        return sr

    async def build_metadata(self, session_id: str) -> CallMetadata:
        """Build the CallMetadata for this replay session."""
        return build_call_metadata(
            session_id=session_id,
            tenant_id=self.config.tenant_id,
            channel=Channel.FILE,
            source_sample_rate=self._get_sample_rate(),
            direction=self.config.direction,
            caller_number=self.config.caller_number,
            callee_number=self.config.callee_number,
            claimed_identity_id=self.config.claimed_identity_id,
            language_hint=self.config.language_hint,
            consent_basis=self.config.consent_basis,
            shadow_mode=self.config.shadow_mode,
            content_type="audio/wav",
        )

    async def stream(self) -> AsyncIterator[tuple[CallMetadata, AudioChunk]]:
        """Yield (CallMetadata, AudioChunk) tuples at real-time pace.

        The caller receives CallMetadata on every yield; it should publish it
        to the bus only on the first chunk (seq == 0).
        """
        session_id = self._make_session_id()
        metadata = await self.build_metadata(session_id)

        wf, n_frames, sample_rate, n_channels, encoding_str = self._open_wav()
        encoding = AudioEncoding(encoding_str)
        chunk_samples = int(self.config.chunk_ms * sample_rate / 1000)
        chunk_duration_s = chunk_samples / sample_rate

        log.info(
            "wav_replay_start",
            session_id=session_id,
            path=str(self.config.wav_path),
            sample_rate=sample_rate,
            n_channels=n_channels,
            n_frames=n_frames,
            duration_s=round(n_frames / sample_rate, 2),
            chunk_ms=self.config.chunk_ms,
            loss_rate=self.config.loss_rate,
            jitter_ms=self.config.jitter_ms,
            speed_factor=self.config.speed_factor,
        )

        seq = 0
        ts_ms = 0

        try:
            while True:
                raw = wf.readframes(chunk_samples)
                if not raw:
                    break

                payload = self._mix_to_mono(raw, n_channels, wf.getsampwidth())

                # Simulate packet loss (seq and ts still advance)
                if self._should_drop():
                    log.debug("packet_dropped", session_id=session_id, seq=seq, ts_ms=ts_ms)
                    seq += 1
                    ts_ms += self.config.chunk_ms
                    continue

                chunk = AudioChunk(
                    session_id=session_id,
                    seq=seq,
                    ts_ms=ts_ms,
                    sample_rate=sample_rate,
                    encoding=encoding,
                    payload=payload,
                )

                yield metadata, chunk

                seq += 1
                ts_ms += self.config.chunk_ms

                delay = self._compute_delay_s(chunk_duration_s)
                if delay > 0:
                    await asyncio.sleep(delay)

        finally:
            wf.close()
            log.info(
                "wav_replay_done",
                session_id=session_id,
                chunks_emitted=seq,
                duration_ms=ts_ms,
            )
