"""Reference connectors: telephony media → VoiceGuard streaming API (B12-T09).

Each bridge turns one provider's media protocol into (call_metadata, audio chunk
bytes, encoding, sample rate) and forwards it to the gateway through the Python
SDK's streaming client — the same path any integrator uses. Protocol parsing
mirrors the B1 ingest adapters.

* ``TwilioMediaStreamsBridge`` — Twilio ``<Stream>`` WebSocket JSON (8 kHz µ-law, base64)
* ``AsteriskAudioSocketBridge`` — Asterisk AudioSocket TCP frames (8 kHz s16le)
* ``FreeSwitchAudioStreamBridge`` — mod_audio_stream WebSocket (JSON header + binary L16)

Collaboration platforms (Teams / Meet / Zoom web): a browser extension captures
tab or microphone audio with an AudioWorklet and streams it with the JS SDK
(``vg.stream`` + ``floatTo16BitPCM``) to the same ``/v1/stream`` endpoint — see the
example in ``sdks/js/README.md``. The extension itself is not built yet.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import struct
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

# AudioSocket message types (Asterisk res_audiosocket)
AS_HANGUP, AS_UUID, AS_AUDIO, AS_ERROR = 0x00, 0x01, 0x10, 0xFF


@dataclass
class BridgeFrame:
    kind: str  # "start" | "audio" | "stop"
    call_metadata: dict[str, Any] | None = None
    payload: bytes = b""
    encoding: str = "pcm_s16le"
    sample_rate: int = 16000


def _meta(
    tenant_id: str, session_id: str, channel: str, codec: str, rate: int, **extra: Any
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "tenant_id": tenant_id,
        "direction": "inbound",
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "channel": channel,
        "codec_hint": codec,
        "source_sample_rate": rate,
        "consent_basis": "legitimate_use",
        **extra,
    }


class TwilioMediaStreamsBridge:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def frames(self, messages: Iterable[str]) -> Iterator[BridgeFrame]:
        for raw in messages:
            msg = json.loads(raw)
            event = msg.get("event")
            if event == "start":
                start = msg.get("start", {})
                params = start.get("customParameters", {})
                yield BridgeFrame(
                    "start",
                    _meta(
                        self.tenant_id,
                        str(uuid.uuid4()),
                        "pstn",
                        "g711u",
                        8000,
                        caller_number=params.get("from"),
                        claimed_identity_id=params.get("claimed_identity_id"),
                        sip_headers={"X-Twilio-CallSid": start.get("callSid", "")},
                    ),
                )
            elif event == "media":
                yield BridgeFrame(
                    "audio",
                    payload=base64.b64decode(msg["media"]["payload"]),
                    encoding="mulaw",
                    sample_rate=8000,
                )
            elif event == "stop":
                yield BridgeFrame("stop")


class AsteriskAudioSocketBridge:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def frames(self, data: bytes) -> Iterator[BridgeFrame]:
        pos = 0
        while pos + 3 <= len(data):
            kind, length = data[pos], struct.unpack(">H", data[pos + 1 : pos + 3])[0]
            payload = data[pos + 3 : pos + 3 + length]
            pos += 3 + length
            if kind == AS_UUID:
                yield BridgeFrame(
                    "start",
                    _meta(
                        self.tenant_id,
                        str(uuid.uuid4()),
                        "pstn",
                        "pcm",
                        8000,
                        sip_headers={"X-Asterisk-UUID": payload.hex()},
                    ),
                )
            elif kind == AS_AUDIO:
                yield BridgeFrame("audio", payload=payload, encoding="pcm_s16le", sample_rate=8000)
            elif kind in (AS_HANGUP, AS_ERROR):
                yield BridgeFrame("stop")


class FreeSwitchAudioStreamBridge:
    def __init__(self, tenant_id: str, sample_rate: int = 16000) -> None:
        self.tenant_id = tenant_id
        self.sample_rate = sample_rate

    def frames(self, messages: Iterable[str | bytes]) -> Iterator[BridgeFrame]:
        for m in messages:
            if isinstance(m, bytes):
                yield BridgeFrame(
                    "audio", payload=m, encoding="pcm_s16le", sample_rate=self.sample_rate
                )
                continue
            hdr = json.loads(m)
            if hdr.get("event", "start") == "stop":
                yield BridgeFrame("stop")
                continue
            self.sample_rate = int(hdr.get("sampleRate", self.sample_rate))
            yield BridgeFrame(
                "start",
                _meta(
                    self.tenant_id,
                    str(uuid.uuid4()),
                    "voip",
                    "pcm",
                    self.sample_rate,
                    caller_number=hdr.get("caller"),
                    sip_headers={"X-FS-UUID": hdr.get("uuid", "")},
                ),
            )


def forward(frames: Iterable[BridgeFrame], stream: Any) -> list[dict[str, Any]]:
    """Drive an SDK stream (``VoiceGuardClient.stream()``) from bridge frames; returns all events."""
    events: list[dict[str, Any]] = []
    started = False
    for f in frames:
        if f.kind == "start" and f.call_metadata is not None:
            events += stream.start(
                f.call_metadata,
                encoding="mulaw" if f.call_metadata["codec_hint"] == "g711u" else "pcm_s16le",
                sample_rate=f.call_metadata["source_sample_rate"],
            )
            started = True
        elif f.kind == "audio" and started:
            events += stream.send_audio(f.payload)
        elif f.kind == "stop" and started:
            events += stream.close()
            started = False
    if started:
        events += stream.close()
    return events
