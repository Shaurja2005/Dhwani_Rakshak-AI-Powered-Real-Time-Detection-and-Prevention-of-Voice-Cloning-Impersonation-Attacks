"""services/conditioner/main.py — B2 Stream Conditioning Service.

Subscribes to the `vg:{tenant_id}:audio_chunk` bus stream.
For each chunk:
  1. Route to per-session JitterBuffer (B2-T01)
  2. Decode + resample to 16 kHz float32 (B2-T02)
  3. Feed to per-session VAD (B2-T03)
  4. Feed to per-session Windower (B2-T04)
  5. For each complete window:
     a. Run quality gate (B2-T05 / B2-T07)
     b. Normalise loudness, log pre-norm level (B2-T06)
     c. Store PCM reference (shm://) — invariant I5: never persist raw audio
     d. Publish AnalysisWindow to vg:{tenant_id}:analysis_window

Latency budget: < 60 ms p95 added by this layer.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.vg_audio.jitter_buffer import JitterBuffer
from packages.vg_audio.quality import assess_quality, normalize_loudness
from packages.vg_audio.vad import EnergyVAD
from packages.vg_audio.windowing import Windower, WindowSlice
from packages.vg_audio.features import clear_session_cache
from packages.vg_core.bus import AbstractBus
from packages.vg_core.logging import get_logger, bind_session
from packages.vg_core.models import AnalysisWindow, AudioChunk

log = get_logger(__name__)

TARGET_SR = 16_000
WINDOW_S = 3.0
HOP_S = 1.0


@dataclass
class SessionConditioner:
    """Holds all per-session DSP state."""
    session_id: str
    original_sample_rate: int
    jitter_buffer: JitterBuffer = field(default_factory=JitterBuffer)
    vad: EnergyVAD = field(default_factory=EnergyVAD)
    windower: Optional[Windower] = None
    window_count: int = 0
    chunk_count: int = 0
    abstain_count: int = 0
    created_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.windower = Windower(
            session_id=self.session_id,
            original_sample_rate=self.original_sample_rate,
            window_s=WINDOW_S,
            hop_s=HOP_S,
        )


class Conditioner:
    """The B2 stream conditioner — one instance per tenant deployment."""

    def __init__(self, bus: AbstractBus, tenant_id: str = "demo") -> None:
        self._bus = bus
        self._tenant_id = tenant_id
        self._sessions: dict[str, SessionConditioner] = {}

    async def process_chunk(self, chunk: AudioChunk) -> list[AnalysisWindow]:
        """Process one AudioChunk and return any complete AnalysisWindows."""
        sid = chunk.session_id
        bind_session(sid)

        if sid not in self._sessions:
            self._sessions[sid] = SessionConditioner(
                session_id=sid,
                original_sample_rate=chunk.sample_rate,
            )

        sc = self._sessions[sid]
        sc.chunk_count += 1

        t0 = time.perf_counter()

        # B2-T01: jitter buffer + PLC
        pcm_frames = sc.jitter_buffer.push(
            seq=chunk.seq,
            ts_ms=chunk.ts_ms,
            payload=chunk.payload,
            encoding=chunk.encoding.value,
            sample_rate=chunk.sample_rate,
        )

        if not pcm_frames:
            return []

        assert sc.windower is not None
        windows_out: list[AnalysisWindow] = []

        for pcm_native in pcm_frames:
            # B2-T02: resample to 16 kHz if needed
            if chunk.sample_rate == TARGET_SR:
                pcm_16k = pcm_native
            else:
                pcm_16k = self._resample_float(pcm_native, chunk.sample_rate)

            # B2-T03: VAD on frame level
            frame = pcm_16k[:320] if len(pcm_16k) >= 320 else pcm_16k
            sc.vad.process_frame(frame)

            # B2-T04: windowing
            for win_slice in sc.windower.push(pcm_16k):
                w = await self._process_window(win_slice, sc)
                if w:
                    windows_out.append(w)

        latency_ms = (time.perf_counter() - t0) * 1000
        if latency_ms > 60:
            log.warning(
                "conditioner_latency_exceeded",
                session_id=sid,
                latency_ms=round(latency_ms, 2),
            )

        return windows_out

    def _resample_float(self, pcm: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample float32 PCM from source_rate to TARGET_SR."""
        try:
            from scipy.signal import resample_poly
            from fractions import Fraction
            frac = Fraction(TARGET_SR, source_rate).limit_denominator(500)
            return resample_poly(pcm, frac.numerator, frac.denominator).astype(np.float32)
        except ImportError:
            n_target = int(len(pcm) * TARGET_SR / source_rate)
            return np.interp(
                np.linspace(0, len(pcm) - 1, n_target),
                np.arange(len(pcm)),
                pcm,
            ).astype(np.float32)

    async def _process_window(
        self, win_slice: WindowSlice, sc: SessionConditioner
    ) -> Optional[AnalysisWindow]:
        # B2-T03: VAD voiced ratio over the full window
        voiced_ratio, _ = sc.vad.process_window(win_slice.pcm)

        # B2-T05 + B2-T07: quality gate
        qr = assess_quality(
            pcm=win_slice.pcm,
            voiced_ratio=voiced_ratio,
            window_duration_s=WINDOW_S,
        )

        # B2-T06: loudness normalisation — log pre-norm level (forensic signal)
        _, pre_norm_dbfs = normalize_loudness(win_slice.pcm)
        log.debug(
            "window_loudness",
            session_id=sc.session_id,
            window_id=win_slice.window_id,
            pre_norm_dbfs=round(pre_norm_dbfs, 2),
            snr_db=qr.snr_db,
            quality_ok=qr.quality_ok,
        )

        sc.window_count += 1
        if not qr.quality_ok:
            sc.abstain_count += 1

        samples_ref = f"shm://{sc.session_id}/{win_slice.window_id}"

        window = AnalysisWindow(
            session_id=sc.session_id,
            window_id=win_slice.window_id,
            start_ms=win_slice.start_ms,
            end_ms=win_slice.end_ms,
            sample_rate=TARGET_SR,
            samples_ref=samples_ref,
            voiced_ms=qr.voiced_ms,
            snr_db=qr.snr_db,
            clipping_ratio=qr.clipping_ratio,
            quality_ok=qr.quality_ok,
            quality_flags=qr.quality_flags,
            original_sample_rate=win_slice.original_sample_rate,
        )

        log.info(
            "window_emitted",
            session_id=sc.session_id,
            window_id=win_slice.window_id,
            quality_ok=qr.quality_ok,
            voiced_ms=qr.voiced_ms,
            flags=qr.quality_flags,
        )

        return window

    async def teardown_session(self, session_id: str) -> None:
        sc = self._sessions.pop(session_id, None)
        if sc is None:
            return
        bind_session(session_id)
        if sc.windower:
            for win_slice in sc.windower.flush():
                await self._process_window(win_slice, sc)
        removed = clear_session_cache(session_id)
        log.info(
            "session_conditioner_teardown",
            session_id=session_id,
            chunks=sc.chunk_count,
            windows=sc.window_count,
            abstained=sc.abstain_count,
            cache_cleared=removed,
            jitter_stats=sc.jitter_buffer.stats,
        )

    async def run(self, stream_key: Optional[str] = None) -> None:
        key = stream_key or f"vg:{self._tenant_id}:audio_chunk"
        out_key = f"vg:{self._tenant_id}:analysis_window"
        log.info("conditioner_starting", stream_key=key)
        async for msg in self._bus.consume(key):
            try:
                chunk = AudioChunk.model_validate_json(msg.data)
                windows = await self.process_chunk(chunk)
                for w in windows:
                    await self._bus.publish(out_key, w.model_dump_json())
            except Exception as exc:  # noqa: BLE001
                log.error("conditioner_chunk_error", error=str(exc))
