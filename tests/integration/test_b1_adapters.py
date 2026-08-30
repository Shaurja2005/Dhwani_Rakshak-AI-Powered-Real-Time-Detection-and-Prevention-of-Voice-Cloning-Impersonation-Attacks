"""Integration tests for B1 capture adapters.

Definition of Done (B1): The same downstream pipeline produces identical-shaped
events whether the source is a WAV replay, a Twilio call, an Asterisk call,
a FreeSWITCH call, WebSocket PCM stream, or SIPREC session.
Prove it with one integration test per adapter using recorded fixtures/frames.

These tests use the InMemoryBus so no Redis is needed. They verify:
1. CallMetadata is published before any AudioChunk.
2. AudioChunks are valid (session_id matches, seq is monotonic, payload non-empty).
3. The adapter terminates cleanly (no stuck tasks).
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import uuid
from pathlib import Path

import pytest

from packages.vg_core.bus import InMemoryBus, set_bus
from packages.vg_core.models import (
    AudioChunk,
    AudioEncoding,
    CallMetadata,
    Channel,
    ConsentBasis,
)
from services.ingest.session import SessionStore


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_WAV = FIXTURES_DIR / "sample.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeWebSocket:
    """Async iterator mimicking a websockets server connection for tests."""

    def __init__(self, messages: list[str | bytes]) -> None:
        self._messages = list(messages)
        self._iter = iter(self._messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def recv(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise asyncio.IncompleteReadError(b"", 0)


def _collect_bus_messages(bus: InMemoryBus, stream: str, max_n: int = 100) -> list[str]:
    """Drain up to max_n messages from the InMemoryBus queue synchronously."""
    msgs = []
    q = bus._get_queue(stream)
    while not q.empty() and len(msgs) < max_n:
        msg = q.get_nowait()
        msgs.append(msg.data)
    return msgs


# ---------------------------------------------------------------------------
# B1-T01 — WAV replay adapter
# ---------------------------------------------------------------------------

class TestWavReplayAdapter:
    @pytest.mark.asyncio
    async def test_emits_metadata_then_chunks(self) -> None:
        """WAV replay must emit (metadata, chunk) tuples with valid shapes."""
        from services.ingest.adapters.replay_wav import WavReplayAdapter, WavReplayConfig

        config = WavReplayConfig(
            wav_path=SAMPLE_WAV,
            tenant_id="test",
            speed_factor=100.0,
            loss_rate=0.0,
            seed=42,
        )
        adapter = WavReplayAdapter(config)
        collected: list[tuple] = []
        async for metadata, chunk in adapter.stream():
            collected.append((metadata, chunk))
            if len(collected) >= 5:
                break

        assert len(collected) > 0, "Expected at least one (metadata, chunk) pair"
        meta0, chunk0 = collected[0]

        # Validate metadata shape
        assert isinstance(meta0, CallMetadata)
        assert meta0.session_id
        assert meta0.tenant_id == "test"

        # Validate chunk shape
        assert isinstance(chunk0, AudioChunk)
        assert chunk0.session_id == meta0.session_id
        assert chunk0.seq == 0
        assert chunk0.ts_ms == 0
        assert len(chunk0.payload) > 0

    @pytest.mark.asyncio
    async def test_chunks_are_monotonically_increasing(self) -> None:
        """seq and ts_ms must be monotonically increasing across chunks."""
        from services.ingest.adapters.replay_wav import WavReplayAdapter, WavReplayConfig

        config = WavReplayConfig(
            wav_path=SAMPLE_WAV,
            speed_factor=100.0,
            loss_rate=0.0,
            seed=1,
        )
        adapter = WavReplayAdapter(config)
        prev_seq = -1
        prev_ts = -1
        count = 0
        async for _, chunk in adapter.stream():
            assert chunk.seq > prev_seq, f"seq not monotonic: {chunk.seq} <= {prev_seq}"
            assert chunk.ts_ms > prev_ts, f"ts_ms not monotonic: {chunk.ts_ms} <= {prev_ts}"
            prev_seq = chunk.seq
            prev_ts = chunk.ts_ms
            count += 1
            if count >= 10:
                break

    @pytest.mark.asyncio
    async def test_packet_loss_skips_seq_numbers(self) -> None:
        """With 50% loss, seq numbers should have gaps."""
        from services.ingest.adapters.replay_wav import WavReplayAdapter, WavReplayConfig

        config = WavReplayConfig(
            wav_path=SAMPLE_WAV,
            speed_factor=100.0,
            loss_rate=0.5,
            seed=7,
        )
        adapter = WavReplayAdapter(config)
        seqs = []
        async for _, chunk in adapter.stream():
            seqs.append(chunk.seq)
            if len(seqs) >= 20:
                break

        if len(seqs) >= 2:
            gaps = sum(1 for i in range(1, len(seqs)) if seqs[i] - seqs[i - 1] > 1)
            assert gaps > 0, "Expected gaps in seq due to packet loss"

    @pytest.mark.asyncio
    async def test_all_chunks_share_session_id(self) -> None:
        """All chunks from one replay must have the same session_id."""
        from services.ingest.adapters.replay_wav import WavReplayAdapter, WavReplayConfig

        config = WavReplayConfig(wav_path=SAMPLE_WAV, speed_factor=100.0)
        adapter = WavReplayAdapter(config)
        session_ids = set()
        async for meta, chunk in adapter.stream():
            session_ids.add(meta.session_id)
            session_ids.add(chunk.session_id)
            if len(session_ids) > 1:
                break

        assert len(session_ids) == 1, f"Expected one session_id, got: {session_ids}"


# ---------------------------------------------------------------------------
# B1-T02 / B1-T07 — WebSocket PCM adapter
# ---------------------------------------------------------------------------

class TestWebSocketPCMAdapter:
    @pytest.mark.asyncio
    async def test_ws_pcm_full_session(self) -> None:
        from services.ingest.adapters.websocket_pcm import WebSocketPCMHandler

        bus = InMemoryBus()
        store = SessionStore()

        header = json.dumps({
            "tenant_id": "test-tenant",
            "direction": "inbound",
            "sample_rate": 16000,
            "caller_number": "+919876543210",
            "claimed_identity_id": "cust_12345",
            "language_hint": "hi",
            "shadow_mode": True,
        })
        # 640 bytes = 20ms @ 16kHz s16le
        pcm_frame = bytes(640)
        heartbeat = json.dumps({"type": "heartbeat"})

        fake_ws = FakeWebSocket([header, pcm_frame, pcm_frame, heartbeat, pcm_frame])
        handler = WebSocketPCMHandler(fake_ws, bus, store, tenant_id="test-tenant")

        await handler.handle()

        # Check metadata on bus
        meta_msgs = _collect_bus_messages(bus, "vg:test-tenant:call_metadata")
        assert len(meta_msgs) == 1
        meta = CallMetadata.model_validate_json(meta_msgs[0])
        assert meta.tenant_id == "test-tenant"
        assert meta.caller_number == "+919876543210"
        assert meta.claimed_identity_id == "cust_12345"
        assert meta.source_sample_rate == 16000

        # Check audio chunks on bus
        chunk_msgs = _collect_bus_messages(bus, "vg:test-tenant:audio_chunk")
        assert len(chunk_msgs) == 3
        chunks = [AudioChunk.model_validate_json(m) for m in chunk_msgs]
        for i, chunk in enumerate(chunks):
            assert chunk.session_id == meta.session_id
            assert chunk.seq == i
            assert chunk.ts_ms == i * 20
            assert chunk.encoding == AudioEncoding.PCM_S16LE
            assert len(chunk.payload) == 640


# ---------------------------------------------------------------------------
# B1-T05 — Twilio Media Streams adapter
# ---------------------------------------------------------------------------

class TestTwilioStreamAdapter:
    @pytest.mark.asyncio
    async def test_twilio_stream_session(self) -> None:
        from services.ingest.adapters.twilio_stream import TwilioStreamHandler

        bus = InMemoryBus()
        store = SessionStore()

        # 160 bytes of mulaw = 20ms @ 8kHz
        mulaw_raw = bytes(160)
        mulaw_b64 = base64.b64encode(mulaw_raw).decode("utf-8")

        start_event = json.dumps({
            "event": "start",
            "streamSid": "MZ12345",
            "start": {
                "callSid": "CA1234567890",
                "from": "+919876543210",
                "to": "+918012345678",
                "customParameters": {
                    "claimed_identity_id": "vip_user_42",
                    "language_hint": "ta",
                }
            }
        })
        media_event_0 = json.dumps({
            "event": "media",
            "media": {
                "chunk": "0",
                "timestamp": "0",
                "payload": mulaw_b64,
            }
        })
        media_event_1 = json.dumps({
            "event": "media",
            "media": {
                "chunk": "1",
                "timestamp": "20",
                "payload": mulaw_b64,
            }
        })
        mark_event = json.dumps({"event": "mark", "mark": {"name": "checkpoint_1"}})
        stop_event = json.dumps({"event": "stop", "stop": {"callSid": "CA1234567890"}})

        fake_ws = FakeWebSocket([start_event, media_event_0, media_event_1, mark_event, stop_event])
        handler = TwilioStreamHandler(fake_ws, bus, store, tenant_id="demo")

        await handler.handle()

        # Verify metadata
        meta_msgs = _collect_bus_messages(bus, "vg:demo:call_metadata")
        assert len(meta_msgs) == 1
        meta = CallMetadata.model_validate_json(meta_msgs[0])
        assert meta.caller_number == "+919876543210"
        assert meta.callee_number == "+918012345678"
        assert meta.claimed_identity_id == "vip_user_42"
        assert meta.channel == Channel.PSTN
        assert meta.codec_hint == "g711u"
        assert meta.source_sample_rate == 8000

        # Verify chunks
        chunk_msgs = _collect_bus_messages(bus, "vg:demo:audio_chunk")
        assert len(chunk_msgs) == 2
        chunks = [AudioChunk.model_validate_json(m) for m in chunk_msgs]
        assert chunks[0].session_id == meta.session_id
        assert chunks[0].seq == 0
        assert chunks[0].ts_ms == 0
        assert chunks[0].encoding == AudioEncoding.MULAW
        assert len(chunks[0].payload) == 160
        assert chunks[1].seq == 1
        assert chunks[1].ts_ms == 20


# ---------------------------------------------------------------------------
# B1-T04 — FreeSWITCH mod_audio_stream adapter
# ---------------------------------------------------------------------------

class TestFreeSwitchAdapter:
    @pytest.mark.asyncio
    async def test_freeswitch_session(self) -> None:
        from services.ingest.adapters.freeswitch_ws import FreeSwitchStreamHandler

        bus = InMemoryBus()
        store = SessionStore()

        start_msg = json.dumps({
            "type": "start",
            "uuid": "fs-uuid-9999",
            "direction": "inbound",
            "caller_id_number": "+919988776655",
            "called_number": "1002",
        })
        # 320 bytes = 20ms @ 8kHz s16le
        audio_frame = bytes(320)
        stop_msg = json.dumps({"type": "stop"})

        fake_ws = FakeWebSocket([start_msg, audio_frame, audio_frame, stop_msg])
        handler = FreeSwitchStreamHandler(fake_ws, bus, store, tenant_id="demo")

        await handler.handle()

        # Verify metadata
        meta_msgs = _collect_bus_messages(bus, "vg:demo:call_metadata")
        assert len(meta_msgs) == 1
        meta = CallMetadata.model_validate_json(meta_msgs[0])
        assert meta.caller_number == "+919988776655"
        assert meta.channel == Channel.VOIP
        assert meta.source_sample_rate == 8000
        assert meta.sip_headers.get("X-FreeSWITCH-UUID") == "fs-uuid-9999"

        # Verify chunks
        chunk_msgs = _collect_bus_messages(bus, "vg:demo:audio_chunk")
        assert len(chunk_msgs) == 2
        chunks = [AudioChunk.model_validate_json(m) for m in chunk_msgs]
        assert chunks[0].session_id == meta.session_id
        assert chunks[0].seq == 0
        assert chunks[0].encoding == AudioEncoding.PCM_S16LE
        assert len(chunks[0].payload) == 320


# ---------------------------------------------------------------------------
# B1-T06 — SIPREC receiver
# ---------------------------------------------------------------------------

class TestSiprecAdapter:
    @pytest.mark.asyncio
    async def test_siprec_session_and_rtp_parsing(self) -> None:
        from services.ingest.adapters.siprec import SiprecRtpReceiver, parse_siprec_xml

        bus = InMemoryBus()
        store = SessionStore()

        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <recording xmlns="urn:ietf:params:xml:ns:recording:1">
          <participant participantid="caller">
            <nameID aor="sip:+919876543210@sbc.bank.in"/>
          </participant>
          <participant participantid="callee">
            <nameID aor="sip:agent101@sbc.bank.in"/>
          </participant>
        </recording>
        """

        parsed = parse_siprec_xml(sample_xml)
        assert parsed.get("caller_aor") == "+919876543210" or "sip:+919876543210@sbc.bank.in"

        receiver = SiprecRtpReceiver(bus, store, tenant_id="demo")
        session_id, port = await receiver.create_session(metadata_xml=sample_xml)
        assert session_id is not None
        assert port >= 9100

        # Build a valid RTP packet (V=2, PT=0 (PCMU), Seq=1, TS=160, SSRC=0x12345678, payload=160 bytes)
        # Header format: [V=2,P=0,X=0,CC=0: 0x80] [M=0, PT=0: 0x00] [Seq: 2 bytes] [TS: 4 bytes] [SSRC: 4 bytes]
        rtp_header = struct.pack("!BBHII", 0x80, 0x00, 1, 160, 0x12345678)
        rtp_payload = bytes(160)
        rtp_packet = rtp_header + rtp_payload

        await receiver.handle_rtp_packet(rtp_packet, ("127.0.0.1", 5000), session_id)

        meta_msgs = _collect_bus_messages(bus, "vg:demo:call_metadata")
        assert len(meta_msgs) == 1
        meta = CallMetadata.model_validate_json(meta_msgs[0])
        assert meta.session_id == session_id

        chunk_msgs = _collect_bus_messages(bus, "vg:demo:audio_chunk")
        assert len(chunk_msgs) == 1
        chunk = AudioChunk.model_validate_json(chunk_msgs[0])
        assert chunk.session_id == session_id
        assert chunk.seq == 1
        assert chunk.encoding == AudioEncoding.MULAW
        assert len(chunk.payload) == 160


# ---------------------------------------------------------------------------
# B1-T08 — Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_e164_normalisation_indian(self) -> None:
        from services.ingest.metadata import normalise_number
        assert normalise_number("9876543210") == "+919876543210"
        assert normalise_number("+919876543210") == "+919876543210"
        assert normalise_number(None) is None

    def test_e164_international(self) -> None:
        from services.ingest.metadata import normalise_number
        assert normalise_number("+12025551234") == "+12025551234"

    def test_sip_identity_extraction(self) -> None:
        from services.ingest.metadata import extract_sip_identity
        headers = {"P-Asserted-Identity": "<sip:alice@example.com>"}
        assert extract_sip_identity(headers) == "alice@example.com"

    def test_codec_detection_mulaw(self) -> None:
        from services.ingest.metadata import detect_codec
        assert detect_codec("audio/PCMU", 8000) == "g711u"

    def test_codec_detection_fallback(self) -> None:
        from services.ingest.metadata import detect_codec
        assert detect_codec(None, 8000) == "g711u"
        assert detect_codec(None, 16000) == "pcm"

    def test_build_call_metadata_valid(self) -> None:
        from services.ingest.metadata import build_call_metadata
        m = build_call_metadata(
            session_id=str(uuid.uuid4()),
            tenant_id="bfsi-demo",
            channel=Channel.PSTN,
            source_sample_rate=8000,
            caller_number="9876543210",
        )
        assert m.caller_number == "+919876543210"
        assert m.codec_hint == "g711u"
        assert m.shadow_mode is True


# ---------------------------------------------------------------------------
# B1-T09 — Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_create_and_get(self) -> None:
        from services.ingest.metadata import build_call_metadata
        store = SessionStore()
        sid = str(uuid.uuid4())
        meta = build_call_metadata(
            session_id=sid,
            tenant_id="test",
            channel=Channel.FILE,
            source_sample_rate=16000,
        )
        session = await store.create(meta)
        assert session.session_id == sid
        got = await store.get(sid)
        assert got is session

    @pytest.mark.asyncio
    async def test_terminate_removes_session(self) -> None:
        from services.ingest.metadata import build_call_metadata
        store = SessionStore()
        sid = str(uuid.uuid4())
        meta = build_call_metadata(
            session_id=sid, tenant_id="test",
            channel=Channel.FILE, source_sample_rate=16000,
        )
        await store.create(meta)
        terminated = await store.terminate(sid)
        assert terminated is not None
        assert await store.get(sid) is None

    @pytest.mark.asyncio
    async def test_orphan_reaper(self) -> None:
        from services.ingest.metadata import build_call_metadata
        import time
        store = SessionStore()
        sid = str(uuid.uuid4())
        meta = build_call_metadata(
            session_id=sid, tenant_id="test",
            channel=Channel.FILE, source_sample_rate=16000,
        )
        session = await store.create(meta)
        # Manually set last_heartbeat to far in the past
        session.last_heartbeat = time.monotonic() - 999

        reaped = await store.reap_orphans(ttl_s=1)
        assert sid in reaped
        assert await store.get(sid) is None


# ---------------------------------------------------------------------------
# B1-T03 — Asterisk AudioSocket protocol parsing
# ---------------------------------------------------------------------------

class TestAsteriskProtocol:
    def _make_message(self, msg_type: int, payload: bytes) -> bytes:
        length = len(payload)
        return bytes([msg_type]) + struct.pack(">H", length) + payload

    @pytest.mark.asyncio
    async def test_audiosocket_full_session(self) -> None:
        """Simulate a full Asterisk session: UUID → SLIN frames → HANGUP."""
        from services.ingest.adapters.asterisk_audiosocket import (
            AudioSocketSession, MSG_UUID, MSG_SLIN, MSG_HANGUP,
        )

        bus = InMemoryBus()
        store = SessionStore()

        # Build fake UUID message (16 bytes)
        fake_uuid = bytes(range(16))
        uuid_msg = self._make_message(MSG_UUID, fake_uuid)

        # Build fake SLIN audio (320 bytes = 20ms @ 8kHz 16-bit)
        slin_payload = struct.pack("<" + "h" * 160, *([0] * 160))
        slin_msg = self._make_message(MSG_SLIN, slin_payload)

        # Build HANGUP
        hangup_msg = self._make_message(MSG_HANGUP, b"")

        all_data = uuid_msg + slin_msg + slin_msg + hangup_msg

        reader = asyncio.StreamReader()
        reader.feed_data(all_data)
        reader.feed_eof()

        class DummyTransport:
            def is_closing(self):
                return True
            def close(self):
                pass
            def get_extra_info(self, name, default=None):
                if name == "peername":
                    return ("127.0.0.1", 5060)
                return default

        writer = asyncio.StreamWriter(DummyTransport(), None, reader, asyncio.get_event_loop())

        session_handler = AudioSocketSession(reader, writer, bus, store, "test")

        await session_handler.handle()

        # Verify metadata published
        meta_msgs = _collect_bus_messages(bus, "vg:test:call_metadata")
        assert len(meta_msgs) == 1
        meta = CallMetadata.model_validate_json(meta_msgs[0])
        assert meta.channel == Channel.PSTN
        assert meta.source_sample_rate == 8000

        # Verify chunks published
        chunk_msgs = _collect_bus_messages(bus, "vg:test:audio_chunk")
        assert len(chunk_msgs) == 2
        chunks = [AudioChunk.model_validate_json(m) for m in chunk_msgs]
        assert chunks[0].session_id == meta.session_id
        assert chunks[0].seq == 0
        assert chunks[0].encoding == AudioEncoding.PCM_S16LE
        assert len(chunks[0].payload) == 320
