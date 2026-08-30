"""services/ingest/adapters/asterisk_audiosocket.py — Asterisk AudioSocket adapter.

B1-T03: Accepts audio from Asterisk via the AudioSocket protocol.

AudioSocket protocol (https://github.com/CyCoreSystems/audiosocket):
  - TCP connection from Asterisk to this server.
  - Each message is a 3-byte header + payload:
      [type:1 byte][length:2 bytes big-endian][payload:N bytes]
  - Types:
      0x00 = UUID (16 bytes, session identifier from Asterisk)
      0x01 = HANGUP (zero-length payload)
      0x02 = SILENCE (empty payload, no audio)
      0x10 = SLIN (signed 16-bit linear PCM, 8000 Hz, mono)
      0xff = ERROR

Audio arrives as slin16 (signed 16-bit PCM) at 8000 Hz.
B2 handles upsampling to 16000 Hz; we preserve original_sample_rate=8000.

Usage (in extensions.conf)::

    [vg-hook]
    exten => _X.,1,Answer()
    same => n,Set(CHANNEL(audioreadformat)=slin16)
    same => n,AudioSocket(192.168.1.100:9092,${UNIQUEID})
    same => n,Hangup()

See deploy/telephony/asterisk/ for complete configs.
"""
from __future__ import annotations

import asyncio
import struct
import uuid
from typing import Optional

from packages.vg_core.logging import get_logger, bind_session
from packages.vg_core.models import AudioChunk, AudioEncoding, Channel, ConsentBasis
from packages.vg_core.bus import AbstractBus
from services.ingest.metadata import build_call_metadata
from services.ingest.session import SessionStore

log = get_logger(__name__)

AUDIOSOCKET_PORT = 9092
AUDIOSOCKET_SAMPLE_RATE = 8000

# Message type constants
MSG_UUID    = 0x00
MSG_HANGUP  = 0x01
MSG_SILENCE = 0x02
MSG_SLIN    = 0x10
MSG_ERROR   = 0xFF

HEADER_LEN = 3  # 1 type + 2 length


class AudioSocketSession:
    """Handles a single Asterisk AudioSocket TCP connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        bus: AbstractBus,
        store: SessionStore,
        tenant_id: str = "demo",
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._bus = bus
        self._store = store
        self._tenant_id = tenant_id

    async def handle(self) -> None:
        peer = self._writer.get_extra_info("peername", ("unknown", 0))
        session_id: Optional[str] = None
        seq = 0
        ts_ms = 0

        log.info("audiosocket_connection", peer=f"{peer[0]}:{peer[1]}")

        try:
            while True:
                # Read 3-byte header
                try:
                    header = await asyncio.wait_for(
                        self._reader.readexactly(HEADER_LEN), timeout=30.0
                    )
                except asyncio.IncompleteReadError:
                    break
                except asyncio.TimeoutError:
                    log.warning("audiosocket_read_timeout", session_id=session_id)
                    break

                msg_type = header[0]
                msg_len = struct.unpack(">H", header[1:3])[0]

                # Read payload
                payload = b""
                if msg_len > 0:
                    try:
                        payload = await asyncio.wait_for(
                            self._reader.readexactly(msg_len), timeout=5.0
                        )
                    except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                        break

                if msg_type == MSG_UUID:
                    # First message — establishes the Asterisk channel UUID
                    asterisk_uuid = payload.hex() if len(payload) == 16 else str(uuid.uuid4())
                    session_id = str(uuid.uuid4())
                    bind_session(session_id)

                    metadata = build_call_metadata(
                        session_id=session_id,
                        tenant_id=self._tenant_id,
                        channel=Channel.PSTN,
                        source_sample_rate=AUDIOSOCKET_SAMPLE_RATE,
                        direction="inbound",
                        sip_headers={"X-Asterisk-UUID": asterisk_uuid},
                        consent_basis=ConsentBasis.LEGITIMATE_USE,
                        shadow_mode=True,
                        content_type="audio/x-slin16",
                    )
                    await self._store.create(metadata)
                    await self._bus.publish(
                        f"vg:{self._tenant_id}:call_metadata",
                        metadata.model_dump_json(),
                    )
                    log.info(
                        "audiosocket_session_started",
                        session_id=session_id,
                        asterisk_uuid=asterisk_uuid,
                    )

                elif msg_type == MSG_SLIN:
                    if session_id is None:
                        log.warning("audiosocket_audio_before_uuid")
                        continue

                    chunk = AudioChunk(
                        session_id=session_id,
                        seq=seq,
                        ts_ms=ts_ms,
                        sample_rate=AUDIOSOCKET_SAMPLE_RATE,
                        encoding=AudioEncoding.PCM_S16LE,
                        payload=payload,
                    )
                    await self._bus.publish(
                        f"vg:{self._tenant_id}:audio_chunk",
                        chunk.model_dump_json(),
                    )
                    await self._store.heartbeat(session_id)

                    # Each SLIN frame from Asterisk is 20 ms = 160 samples @ 8kHz
                    chunk_ms = int(len(payload) / (AUDIOSOCKET_SAMPLE_RATE / 1000 * 2))
                    seq += 1
                    ts_ms += chunk_ms

                elif msg_type == MSG_SILENCE:
                    pass  # Quality gate handles silence downstream

                elif msg_type == MSG_HANGUP:
                    log.info("audiosocket_hangup", session_id=session_id)
                    break

                elif msg_type == MSG_ERROR:
                    log.error("audiosocket_error_msg", session_id=session_id, payload=payload.hex())
                    break

        except Exception as exc:  # noqa: BLE001
            log.error("audiosocket_exception", session_id=session_id, error=str(exc))
        finally:
            if session_id:
                await self._store.terminate(session_id, reason="audiosocket_hangup")
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass


async def serve_audiosocket(
    bus: AbstractBus,
    store: SessionStore,
    host: str = "0.0.0.0",
    port: int = AUDIOSOCKET_PORT,
    tenant_id: str = "demo",
) -> None:
    """Start the Asterisk AudioSocket TCP server."""

    async def _client_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        session = AudioSocketSession(reader, writer, bus, store, tenant_id)
        await session.handle()

    server = await asyncio.start_server(_client_handler, host, port)
    log.info("audiosocket_server_started", host=host, port=port)
    async with server:
        await server.serve_forever()
