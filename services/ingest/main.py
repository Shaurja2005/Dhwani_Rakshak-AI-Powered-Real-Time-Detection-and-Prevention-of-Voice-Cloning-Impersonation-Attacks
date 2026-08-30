"""services/ingest/main.py — Ingest service entrypoint.

Starts all configured capture adapters and the session orphan reaper.
Each adapter runs as an asyncio task.

Adapters can be toggled via environment variables:
    VG_INGEST_WAV_REPLAY=1        (dev only: replay a WAV on startup)
    VG_INGEST_WEBSOCKET=1         (default port 8765)
    VG_INGEST_TWILIO=1            (default port 8766)
    VG_INGEST_FREESWITCH=1        (default port 8767)
    VG_INGEST_AUDIOSOCKET=1       (default port 9092)

Usage::

    python -m services.ingest.main
    python -m services.ingest.main --replay tests/fixtures/sample.wav
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.vg_core.bus import get_bus, InMemoryBus, set_bus
from packages.vg_core.config import settings
from packages.vg_core.logging import configure_logging, get_logger
from services.ingest.session import SessionStore, orphan_reaper_task

log = get_logger(__name__)


async def run_replay(wav_path: Path, store: SessionStore, bus: object) -> None:
    """Run a WAV replay session (dev mode)."""
    from services.ingest.adapters.replay_wav import WavReplayAdapter, WavReplayConfig
    import json

    config = WavReplayConfig(
        wav_path=wav_path,
        tenant_id=settings.tenant_default,
        shadow_mode=settings.shadow_mode,
        speed_factor=10.0,  # 10x faster in dev
        loss_rate=0.0,
        jitter_ms=2.0,
    )
    adapter = WavReplayAdapter(config)
    first = True
    async for metadata, chunk in adapter.stream():
        if first:
            await bus.publish(  # type: ignore[attr-defined]
                f"vg:{metadata.tenant_id}:call_metadata",
                metadata.model_dump_json(),
            )
            await store.create(metadata)
            first = False
        await bus.publish(  # type: ignore[attr-defined]
            f"vg:{metadata.tenant_id}:audio_chunk",
            chunk.model_dump_json(),
        )
    await store.terminate(metadata.session_id, reason="replay_complete")  # type: ignore[possibly-undefined]
    log.info("replay_complete", wav=str(wav_path))


async def main(replay_wav: Path | None = None) -> None:
    configure_logging(
        level=os.getenv("VG_LOG_LEVEL", "INFO"),
        json_output=settings.env != "dev",
    )

    bus = await get_bus()
    store = SessionStore()

    tasks: list[asyncio.Task] = []

    # Always start the orphan reaper
    tasks.append(asyncio.create_task(orphan_reaper_task(store), name="orphan_reaper"))
    log.info("ingest_service_starting", env=settings.env)

    if replay_wav:
        log.info("replay_mode", wav=str(replay_wav))
        tasks.append(asyncio.create_task(run_replay(replay_wav, store, bus), name="wav_replay"))

    if os.getenv("VG_INGEST_WEBSOCKET", "0") == "1" or not replay_wav:
        from services.ingest.adapters.websocket_pcm import serve_ws
        port = int(os.getenv("VG_WS_PORT", "8765"))
        tasks.append(asyncio.create_task(serve_ws(bus, store, port=port), name="ws_pcm"))

    if os.getenv("VG_INGEST_TWILIO", "0") == "1":
        from services.ingest.adapters.twilio_stream import serve_twilio_ws
        port = int(os.getenv("VG_TWILIO_PORT", "8766"))
        tasks.append(asyncio.create_task(
            serve_twilio_ws(bus, store, port=port, tenant_id=settings.tenant_default),
            name="twilio_ws",
        ))

    if os.getenv("VG_INGEST_FREESWITCH", "0") == "1":
        from services.ingest.adapters.freeswitch_ws import serve_freeswitch_ws
        port = int(os.getenv("VG_FS_PORT", "8767"))
        tasks.append(asyncio.create_task(
            serve_freeswitch_ws(bus, store, port=port, tenant_id=settings.tenant_default),
            name="freeswitch_ws",
        ))

    if os.getenv("VG_INGEST_AUDIOSOCKET", "0") == "1":
        from services.ingest.adapters.asterisk_audiosocket import serve_audiosocket
        port = int(os.getenv("VG_AUDIOSOCKET_PORT", "9092"))
        tasks.append(asyncio.create_task(
            serve_audiosocket(bus, store, port=port, tenant_id=settings.tenant_default),
            name="audiosocket",
        ))

    log.info("ingest_service_ready", adapters=[t.get_name() for t in tasks])

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc:
                log.error("adapter_failed", name=task.get_name(), error=str(exc))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("ingest_service_shutting_down")
    finally:
        for task in tasks:
            task.cancel()
        await bus.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceGuard ingest service")
    parser.add_argument("--replay", type=Path, help="Replay a WAV file (dev mode)")
    args = parser.parse_args()
    asyncio.run(main(replay_wav=args.replay))
