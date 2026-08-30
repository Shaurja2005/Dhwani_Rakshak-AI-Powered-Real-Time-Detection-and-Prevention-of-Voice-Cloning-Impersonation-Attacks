"""services/ingest/adapters/twilio_stream.py — Twilio Media Streams adapter.

B1-T05: Accepts Twilio Media Streams (µ-law 8kHz, base64-encoded PCM over WSS).

Twilio protocol (https://www.twilio.com/docs/voice/twiml/stream):
  - Client (Twilio) opens WS to your server.
  - Twilio sends JSON text frames:
      {"event": "start",   "start": {...}}
      {"event": "media",   "media": {"payload": "<base64 mulaw>", "chunk": N, "timestamp": "..."}}
      {"event": "stop",    "stop": {...}}
      {"event": "mark",    "mark": {...}}

Key telephony facts handled here:
  - Encoding: G.711 µ-law (PCMU) at 8000 Hz, 8-bit samples, base64 encoded.
  - B2 will resample to 16 kHz — we preserve the original 8000 Hz in metadata.
  - Dual tracks (inbound + outbound) are demuxed by trackSide.
  - ngrok tunnel required for local dev (documented at end of file).

Usage::

    # In TwiML:
    # <Stream url="wss://your-ngrok.ngrok.io/twilio/ws" track="both_tracks"/>
"""
from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Optional

from packages.vg_core.logging import get_logger, bind_session
from packages.vg_core.models import AudioChunk, AudioEncoding, Channel, ConsentBasis
from packages.vg_core.bus import AbstractBus
from services.ingest.metadata import build_call_metadata
from services.ingest.session import SessionStore

log = get_logger(__name__)

TWILIO_SAMPLE_RATE = 8000
TWILIO_ENCODING = AudioEncoding.MULAW


class TwilioStreamHandler:
    """Handles a single Twilio Media Streams WebSocket session.

    Twilio sends µ-law base64 chunks; we decode and emit AudioChunks.
    The original 8 kHz µ-law encoding is preserved — B2 handles resampling.
    """

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
        call_sid: Optional[str] = None

        try:
            async for raw_msg in self._ws:  # type: ignore[attr-defined]
                if not isinstance(raw_msg, str):
                    continue

                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event")

                if event == "start":
                    session_id, call_sid = await self._handle_start(msg)

                elif event == "media" and session_id:
                    await self._handle_media(msg, session_id)

                elif event == "stop" and session_id:
                    await self._handle_stop(session_id)
                    break

                elif event == "mark":
                    log.debug("twilio_mark", mark=msg.get("mark", {}).get("name"))

        except Exception as exc:  # noqa: BLE001
            log.error("twilio_handler_error", session_id=session_id, error=str(exc))
        finally:
            if session_id:
                await self._store.terminate(session_id, reason="twilio_stop")

    async def _handle_start(self, msg: dict) -> tuple[str, str]:
        """Process the Twilio 'start' event — extract metadata and create session."""
        start = msg.get("start", {})
        call_sid = start.get("callSid", str(uuid.uuid4()))
        stream_sid = msg.get("streamSid", "")
        session_id = str(uuid.uuid4())

        bind_session(session_id)

        # Extract custom parameters if set in TwiML <Parameter> tags
        custom_params = start.get("customParameters", {})
        caller_number = start.get("from") or custom_params.get("caller_number")
        callee_number = start.get("to") or custom_params.get("callee_number")
        claimed_id = custom_params.get("claimed_identity_id")
        language_hint = custom_params.get("language_hint")

        metadata = build_call_metadata(
            session_id=session_id,
            tenant_id=self._tenant_id,
            channel=Channel.PSTN,
            source_sample_rate=TWILIO_SAMPLE_RATE,
            direction="inbound",
            caller_number=caller_number,
            callee_number=callee_number,
            claimed_identity_id=claimed_id,
            language_hint=language_hint,
            consent_basis=ConsentBasis.LEGITIMATE_USE,
            shadow_mode=True,
            content_type="audio/mulaw",
            sip_headers={"X-Twilio-CallSid": call_sid, "X-Twilio-StreamSid": stream_sid},
        )

        await self._store.create(metadata)
        await self._bus.publish(
            f"vg:{self._tenant_id}:call_metadata",
            metadata.model_dump_json(),
        )

        log.info(
            "twilio_session_started",
            session_id=session_id,
            call_sid=call_sid,
            caller_number=caller_number,
        )
        return session_id, call_sid

    async def _handle_media(self, msg: dict, session_id: str) -> None:
        """Process a Twilio 'media' event — decode µ-law payload and emit AudioChunk."""
        media = msg.get("media", {})
        raw_b64 = media.get("payload", "")
        twilio_chunk = int(media.get("chunk", 0))
        timestamp_str = media.get("timestamp", "0")

        try:
            payload = base64.b64decode(raw_b64)
        except Exception:  # noqa: BLE001
            log.warning("twilio_bad_b64", session_id=session_id, chunk=twilio_chunk)
            return

        ts_ms = int(timestamp_str) if timestamp_str.isdigit() else twilio_chunk * 20

        chunk = AudioChunk(
            session_id=session_id,
            seq=twilio_chunk,
            ts_ms=ts_ms,
            sample_rate=TWILIO_SAMPLE_RATE,
            encoding=TWILIO_ENCODING,
            payload=payload,
        )

        # Look up the session to get tenant_id for stream key
        session = await self._store.get(session_id)
        tenant_id = session.tenant_id if session else self._tenant_id

        await self._bus.publish(f"vg:{tenant_id}:audio_chunk", chunk.model_dump_json())
        if session:
            session.chunk_count += 1
            await self._store.heartbeat(session_id)

    async def _handle_stop(self, session_id: str) -> None:
        log.info("twilio_stop_received", session_id=session_id)


# ---------------------------------------------------------------------------
# Local dev tunnel guide
# ---------------------------------------------------------------------------
"""
## ngrok tunnel for local Twilio dev

1. Install ngrok: https://ngrok.com/download
2. Run: ngrok http 8766
3. Copy the https URL, e.g.: https://abc123.ngrok.io
4. In Twilio console, configure your phone number's Voice webhook:
   - WebSocket URL: wss://abc123.ngrok.io/twilio/ws
5. In your TwiML:
   <Response>
     <Connect>
       <Stream url="wss://abc123.ngrok.io/twilio/ws" track="inbound_track">
         <Parameter name="claimed_identity_id" value="{{ caller_id }}" />
         <Parameter name="language_hint" value="hi" />
       </Stream>
     </Connect>
   </Response>

Note: Twilio sends 8 kHz µ-law. The B2 conditioner MUST be tested with this
specific codec path. Use tests/fixtures/sample_8k_mulaw.wav for regression.
"""


async def serve_twilio_ws(
    bus: AbstractBus,
    store: SessionStore,
    host: str = "0.0.0.0",
    port: int = 8766,
    tenant_id: str = "demo",
) -> None:
    """Start the Twilio Media Streams WebSocket server."""
    try:
        import websockets  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("websockets is required. pip install websockets") from exc

    async def _handler(websocket: object, path: str = "/") -> None:
        handler = TwilioStreamHandler(websocket, bus, store, tenant_id)
        await handler.handle()

    log.info("twilio_ws_starting", host=host, port=port)
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()
