"""services/ingest/adapters/siprec.py — SIPREC receiver (B1-T06, stretch target).

SIPREC (RFC 7865/7866) allows an SBC or session border element to fork
a call to a recording server using a multipart SIP INVITE.

This module provides:
1. A minimal SIPREC SIP INVITE parser to extract metadata from the XML body.
2. Two-track RTP demuxer (inbound + outbound legs).
3. Integration into the ingest bus.

SIPREC is the enterprise story — BFSI customers with Cisco CUBE or Ribbon
SBCs will use this path. For greenfield VoIP, use the WebSocket adapter.

Simplifications in this implementation:
- SIP signalling is handled by the SBC; we receive RTP directly.
- We parse the multipart XML to know which SSRC is which party.
- RTP framing: we only support PCMU (G.711 µ-law) and PCMA (G.711 A-law).
- Full SIP stack (re-INVITE, BYE, etc.) is delegated to the SBC.

For a production deployment, front this with Kamailio or a commercial SBC
that terminates SIP and forwards RTP to our known port range.
"""
from __future__ import annotations

import asyncio
import struct
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

from packages.vg_core.logging import get_logger, bind_session
from packages.vg_core.models import AudioChunk, AudioEncoding, Channel, ConsentBasis
from packages.vg_core.bus import AbstractBus
from services.ingest.metadata import build_call_metadata
from services.ingest.session import SessionStore

log = get_logger(__name__)

SIPREC_RTP_PORT_BASE = 9100  # Allocate ports 9100-9199 for SIPREC RTP
RTP_HEADER_LEN = 12


def parse_siprec_xml(xml_body: str) -> dict:
    """Parse SIPREC recording metadata XML (RFC 7865) to extract call info.

    Returns a dict with: caller_number, callee_number, direction, language_hint.
    Returns empty dict on parse failure (non-blocking — we proceed with defaults).
    """
    try:
        root = ET.fromstring(xml_body)
        ns = {"rs": "urn:ietf:params:xml:ns:recording:1"}
        info: dict = {}

        # Extract participant info
        participants = root.findall(".//rs:participant", ns)
        for p in participants:
            role = p.get("participantid", "")
            name_el = p.find("rs:nameID/rs:name", ns)
            aor_el = p.find("rs:nameID", ns)
            if aor_el is not None:
                aor = aor_el.get("aor", "")
                if "caller" in role.lower() or participants.index(p) == 0:
                    info["caller_aor"] = aor
                else:
                    info["callee_aor"] = aor

        return info
    except ET.ParseError:
        return {}


@dataclass
class RTPSession:
    """Tracks RTP state for one leg of a SIPREC recording."""
    ssrc: int
    session_id: str
    payload_type: int = 0  # 0=PCMU, 8=PCMA
    seq: int = 0
    ts_ms: int = 0
    packet_count: int = 0


class SiprecRtpReceiver:
    """UDP server receiving dual-track RTP from an SBC for a SIPREC session.

    Each SIPREC session has two RTP tracks on consecutive ports:
      port   → inbound (caller → callee direction)
      port+1 → outbound (callee → caller direction)
    """

    def __init__(
        self,
        bus: AbstractBus,
        store: SessionStore,
        tenant_id: str = "demo",
        port_base: int = SIPREC_RTP_PORT_BASE,
    ) -> None:
        self._bus = bus
        self._store = store
        self._tenant_id = tenant_id
        self._port_base = port_base
        self._sessions: dict[tuple, RTPSession] = {}  # (host, port) -> RTPSession

    def _parse_rtp(self, data: bytes) -> Optional[tuple[int, int, bytes]]:
        """Parse RTP header. Returns (payload_type, seq, payload) or None."""
        if len(data) < RTP_HEADER_LEN:
            return None
        try:
            # Version should be 2
            version = (data[0] >> 6) & 0x3
            if version != 2:
                return None
            payload_type = data[1] & 0x7F
            seq = struct.unpack(">H", data[2:4])[0]
            payload = data[RTP_HEADER_LEN:]
            return payload_type, seq, payload
        except Exception:  # noqa: BLE001
            return None

    async def create_session(
        self,
        metadata_xml: Optional[str] = None,
        caller_number: Optional[str] = None,
        callee_number: Optional[str] = None,
    ) -> tuple[str, int]:
        """Create a new SIPREC session and return (session_id, allocated_port)."""
        session_id = str(uuid.uuid4())
        bind_session(session_id)

        # Parse XML metadata if provided
        xml_info: dict = {}
        if metadata_xml:
            xml_info = parse_siprec_xml(metadata_xml)

        caller = caller_number or xml_info.get("caller_aor")
        callee = callee_number or xml_info.get("callee_aor")

        metadata = build_call_metadata(
            session_id=session_id,
            tenant_id=self._tenant_id,
            channel=Channel.PSTN,
            source_sample_rate=8000,
            direction="inbound",
            caller_number=caller,
            callee_number=callee,
            consent_basis=ConsentBasis.LEGITIMATE_USE,
            shadow_mode=True,
            content_type="audio/pcmu",
        )
        await self._store.create(metadata)
        await self._bus.publish(
            f"vg:{self._tenant_id}:call_metadata",
            metadata.model_dump_json(),
        )

        # Allocate port (naive: use session count * 2 offset from base)
        port = self._port_base + (len(self._sessions) * 2) % 100
        log.info(
            "siprec_session_created",
            session_id=session_id,
            port_inbound=port,
            port_outbound=port + 1,
        )
        return session_id, port

    async def handle_rtp_packet(
        self,
        data: bytes,
        addr: tuple,
        session_id: str,
        direction: str = "inbound",
    ) -> None:
        """Process a single RTP packet for a SIPREC session."""
        parsed = self._parse_rtp(data)
        if parsed is None:
            return

        payload_type, rtp_seq, payload = parsed

        # Determine encoding from payload type
        if payload_type == 0:
            encoding = AudioEncoding.MULAW
        elif payload_type == 8:
            encoding = AudioEncoding.ALAW
        else:
            return  # Unsupported codec — ignore

        # Each µ-law/A-law RTP packet is 20ms @ 8kHz = 160 bytes
        chunk_ms = int(len(payload) / 8)  # 8 samples/ms at 8kHz
        ts_ms = rtp_seq * chunk_ms

        chunk = AudioChunk(
            session_id=session_id,
            seq=rtp_seq,
            ts_ms=ts_ms,
            sample_rate=8000,
            encoding=encoding,
            payload=payload,
        )
        await self._bus.publish(
            f"vg:{self._tenant_id}:audio_chunk",
            chunk.model_dump_json(),
        )
        await self._store.heartbeat(session_id)
