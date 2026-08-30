"""services/ingest/adapters/websocket_pcm.py — Generic WebSocket PCM ingest.

B1-T02: Accepts raw 16-bit PCM audio frames over a WebSocket connection.

Protocol:
  1. Client opens WS connection to ws://host:PORT/ingest/ws
  2. Client sends a JSON header as the FIRST text frame:
     {
       "tenant_id": "...",
       "direction": "inbound|outbound",
       "sample_rate": 16000,
       "caller_number": "+919876543210",   // optional
       "claimed_identity_id": "...",        // optional
       "language_hint": "hi",              // optional
       "shadow_mode": true
     }
  3. Client sends binary frames of raw PCM s16le audio.
  4. Client closes the connection to signal end of session.

The server emits AudioChunk events to the internal bus as they arrive.

This adapter is intentionally protocol-agnostic — it works with browser
demos (getUserMedia → AudioWorklet → WebSocket), cloud telephony vendors
(Twilio, Plivo, SignalWire via custom relay), and SoftPhones.

The Twilio-specific base64-µ-law adapter in twilio_stream.py reuses the
session and bus logic from here.
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

WS_INGEST_PORT: int = 8765


class WebSocketPCMHandler:
    """Handles a single WebSocket session from connection to teardown.

    Instantiate one per WS connection (not shared).
    """

    def __init__(
        self,
        websocket: object,          # websockets.ServerProtocol — typed loosely to avoid hard dep
        bus: AbstractBus,
        store: SessionStore,
        tenant_id: Optional[str] = None,
    ) -> None:
        self._ws = websocket
        self._bus = bus
        self._store = store
        self._default_tenant = tenant_id or "demo"

    async def handle(self) -> None:
        """Main coroutine: parse header, then stream chunks until close."""
        session_id = str(uuid.uuid4())
        bind_session(session_id)

        try:
            # Step 1: receive JSON header (first text frame)
            header = await self._recv_header()
            if header is None:
                log.warning("ws_no_header", session_id=session_id)
                return

            tenant_id = header.get("tenant_id", self._default_tenant)
            sample_rate = int(header.get("sample_rate", 16000))
            direction = header.get("direction", "inbound")
            caller_number = header.get("caller_number")
            claimed_id = header.get("claimed_identity_id")
            language_hint = header.get("language_hint")
            shadow_mode = bool(header.get("shadow_mode", True))

            metadata = build_call_metadata(
                session_id=session_id,
                tenant_id=tenant_id,
                channel=Channel.WEBRTC,
                source_sample_rate=sample_rate,
                direction=direction,
                caller_number=caller_number,
                claimed_identity_id=claimed_id,
                language_hint=language_hint,
                consent_basis=ConsentBasis.LEGITIMATE_USE,
                shadow_mode=shadow_mode,
            )

            session = await self._store.create(metadata)
            stream_key = f"vg:{tenant_id}:audio_chunk"

            # Publish CallMetadata to the bus
            await self._bus.publish(
                f"vg:{tenant_id}:call_metadata",
                metadata.model_dump_json(),
            )
            log.info(
                "ws_session_started",
                session_id=session_id,
                tenant_id=tenant_id,
                sample_rate=sample_rate,
            )

            # Step 2: stream binary PCM frames
            seq = 0
            ts_ms = 0
            chunk_ms = 20  # default ptime
            chunk_samples = int(chunk_ms * sample_rate / 1000)
            chunk_bytes = chunk_samples * 2  # 16-bit PCM

            buffer = bytearray()

            async for message in self._iter_messages():
                if isinstance(message, str):
                    # Control frame (e.g., heartbeat ping)
                    try:
                        ctrl = json.loads(message)
                        if ctrl.get("type") == "heartbeat":
                            await self._store.heartbeat(session_id)
                    except json.JSONDecodeError:
                        pass
                    continue

                # Binary frame: accumulate into buffer
                buffer.extend(message)

                # Emit complete chunks
                while len(buffer) >= chunk_bytes:
                    payload = bytes(buffer[:chunk_bytes])
                    buffer = buffer[chunk_bytes:]

                    chunk = AudioChunk(
                        session_id=session_id,
                        seq=seq,
                        ts_ms=ts_ms,
                        sample_rate=sample_rate,
                        encoding=AudioEncoding.PCM_S16LE,
                        payload=payload,
                    )
                    await self._bus.publish(stream_key, chunk.model_dump_json())
                    session.chunk_count += 1
                    seq += 1
                    ts_ms += chunk_ms

        except Exception as exc:  # noqa: BLE001
            log.error("ws_handler_error", session_id=session_id, error=str(exc))
        finally:
            await self._store.terminate(session_id, reason="ws_close")
            log.info("ws_session_ended", session_id=session_id)

    async def _recv_header(self) -> Optional[dict]:
        """Receive and parse the first text message as the session header."""
        try:
            msg = await asyncio.wait_for(self._ws.recv(), timeout=10.0)  # type: ignore[attr-defined]
            if isinstance(msg, str):
                return json.loads(msg)
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
            pass
        return None

    async def _iter_messages(self):  # type: ignore[return]
        """Yield raw messages from the WebSocket until close."""
        try:
            async for msg in self._ws:  # type: ignore[attr-defined]
                yield msg
        except Exception:  # noqa: BLE001
            return


async def serve_ws(
    bus: AbstractBus,
    store: SessionStore,
    host: str = "0.0.0.0",
    port: int = WS_INGEST_PORT,
    tenant_id: Optional[str] = None,
) -> None:
    """Start the WebSocket PCM ingest server.

    Call from services/ingest/main.py as an asyncio task.
    """
    try:
        import websockets  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "websockets is required for WebSocket ingest. pip install websockets"
        ) from exc

    async def _handler(websocket: object, path: str = "/") -> None:
        handler = WebSocketPCMHandler(websocket, bus, store, tenant_id)
        await handler.handle()

    log.info("ws_server_starting", host=host, port=port)
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()  # run forever
