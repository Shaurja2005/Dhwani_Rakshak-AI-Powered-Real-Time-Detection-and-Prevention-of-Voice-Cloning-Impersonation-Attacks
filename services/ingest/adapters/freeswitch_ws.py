"""services/ingest/adapters/freeswitch_ws.py — FreeSWITCH mod_audio_stream adapter.

B1-T04: Accepts audio from FreeSWITCH via mod_audio_stream (WebSocket fork).

FreeSWITCH mod_audio_stream sends:
  1. A JSON text frame describing the call on connect:
     {"uuid": "...", "direction": "recv", "caller_id_number": "...", ...}
  2. Binary frames of raw µ-law or SLIN audio at 8000 Hz.
  3. A JSON "stop" frame on hangup.

Dialplan usage::

    <action application="audio_stream" data="ws://192.168.1.100:8767|mono|8000"/>

See deploy/telephony/freeswitch/ for complete sofia.conf and dialplan configs.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

from packages.vg_core.logging import get_logger, bind_session
from packages.vg_core.models import AudioChunk, AudioEncoding, Channel, ConsentBasis
from packages.vg_core.bus import AbstractBus
from services.ingest.metadata import build_call_metadata
from services.ingest.session import SessionStore

log = get_logger(__name__)

FS_PORT = 8767
FS_SAMPLE_RATE = 8000


class FreeSwitchStreamHandler:
    """Handles a single FreeSWITCH mod_audio_stream WebSocket session."""

    def __init__(
        self,
        websocket: object,
        bus: AbstractBus,
        store: SessionStore,
        tenant_id: str = "demo",
    ) -> None:
        self._ws = websocket
        self._bus = bus
        self._store = store
        self._tenant_id = tenant_id

    async def handle(self) -> None:
        session_id: Optional[str] = None
        seq = 0
        ts_ms = 0

        try:
            async for raw_msg in self._ws:  # type: ignore[attr-defined]
                if isinstance(raw_msg, str):
                    # JSON control frame
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", msg.get("action", ""))

                    if msg_type in ("start", "session", "") and session_id is None:
                        # FreeSWITCH initial info frame
                        session_id = str(uuid.uuid4())
                        bind_session(session_id)

                        caller_number = (
                            msg.get("caller_id_number")
                            or msg.get("from")
                            or msg.get("callerIdNumber")
                        )
                        callee_number = msg.get("to") or msg.get("called_number")
                        fs_uuid = msg.get("uuid", "")

                        metadata = build_call_metadata(
                            session_id=session_id,
                            tenant_id=self._tenant_id,
                            channel=Channel.VOIP,
                            source_sample_rate=FS_SAMPLE_RATE,
                            direction=msg.get("direction", "inbound"),
                            caller_number=caller_number,
                            callee_number=callee_number,
                            consent_basis=ConsentBasis.LEGITIMATE_USE,
                            shadow_mode=True,
                            content_type="audio/x-slin16",
                            sip_headers={"X-FreeSWITCH-UUID": fs_uuid},
                        )
                        await self._store.create(metadata)
                        await self._bus.publish(
                            f"vg:{self._tenant_id}:call_metadata",
                            metadata.model_dump_json(),
                        )
                        log.info(
                            "freeswitch_session_started",
                            session_id=session_id,
                            fs_uuid=fs_uuid,
                        )

                    elif msg_type == "stop":
                        log.info("freeswitch_stop", session_id=session_id)
                        break

                elif isinstance(raw_msg, (bytes, bytearray)) and session_id:
                    # Binary audio frame
                    payload = bytes(raw_msg)
                    chunk = AudioChunk(
                        session_id=session_id,
                        seq=seq,
                        ts_ms=ts_ms,
                        sample_rate=FS_SAMPLE_RATE,
                        encoding=AudioEncoding.PCM_S16LE,
                        payload=payload,
                    )
                    await self._bus.publish(
                        f"vg:{self._tenant_id}:audio_chunk",
                        chunk.model_dump_json(),
                    )
                    await self._store.heartbeat(session_id)

                    chunk_ms = int(len(payload) / (FS_SAMPLE_RATE / 1000 * 2))
                    seq += 1
                    ts_ms += chunk_ms

        except Exception as exc:  # noqa: BLE001
            log.error("freeswitch_exception", session_id=session_id, error=str(exc))
        finally:
            if session_id:
                await self._store.terminate(session_id, reason="freeswitch_hangup")


async def serve_freeswitch_ws(
    bus: AbstractBus,
    store: SessionStore,
    host: str = "0.0.0.0",
    port: int = FS_PORT,
    tenant_id: str = "demo",
) -> None:
    try:
        import websockets  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("websockets is required") from exc

    async def _handler(websocket: object, path: str = "/") -> None:
        FreeSwitchStreamHandler(websocket, bus, store, tenant_id)
        await FreeSwitchStreamHandler(websocket, bus, store, tenant_id).handle()

    log.info("freeswitch_ws_starting", host=host, port=port)
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()
