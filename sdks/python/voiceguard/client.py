"""VoiceGuard Python SDK (B12-T05).

    from voiceguard import VoiceGuardClient

    vg = VoiceGuardClient("https://voiceguard.bank.internal", api_key="vg_live_...")
    result = vg.analyze_file("call.wav", tenant_id="bank-a")
    print(result["session_risk"]["state"], result["agent_prompt"])

    with vg.stream(tenant_id="bank-a", claimed_identity_id="cust-42") as s:
        for chunk in pcm_chunks:            # 16 kHz s16le bytes (or mulaw at 8 kHz)
            for event in s.send_audio(chunk):
                if event["type"] == "policy_decision":
                    show_banner(event["data"])

Detection is advisory: show results to a human; never auto-decline a customer on them.
"""

from __future__ import annotations

import base64
import datetime as dt
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from services.policy.notify import verify_signature as _verify_signature


class VoiceGuardError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def call_metadata(
    tenant_id: str,
    session_id: str | None = None,
    channel: str = "file",
    codec_hint: str = "pcm",
    source_sample_rate: int = 16000,
    direction: str = "inbound",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "direction": direction,
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "channel": channel,
        "codec_hint": codec_hint,
        "source_sample_rate": source_sample_rate,
        "consent_basis": extra.pop("consent_basis", "legitimate_use"),
        **extra,
    }


class StreamSession:
    """A live session over the gateway's session REST API (works behind any HTTP proxy)."""

    def __init__(self, client: VoiceGuardClient) -> None:
        self._c = client
        self.session_id: str | None = None
        self.encoding = "pcm_s16le"
        self.sample_rate = 16000
        self.events: list[dict[str, Any]] = []

    def start(
        self, metadata: dict[str, Any], encoding: str = "pcm_s16le", sample_rate: int = 16000
    ) -> list[dict[str, Any]]:
        self.encoding, self.sample_rate = encoding, sample_rate
        body = self._c._request("POST", "/v1/sessions", json={"call_metadata": metadata})
        self.session_id = body["session_id"]
        return []

    def send_audio(self, chunk: bytes) -> list[dict[str, Any]]:
        body = self._c._request(
            "POST",
            f"/v1/sessions/{self.session_id}/audio",
            json={
                "payload_base64": base64.b64encode(chunk).decode(),
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            },
        )
        self.events += body["events"]
        return list(body["events"])

    def send_transcript(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        body = self._c._request("POST", f"/v1/sessions/{self.session_id}/transcript", json=segments)
        self.events += body["events"]
        return list(body["events"])

    def risk(self) -> dict[str, Any]:
        return self._c._request("GET", f"/v1/sessions/{self.session_id}/risk")

    def close(self) -> list[dict[str, Any]]:
        if self.session_id is None:
            return []
        body = self._c._request("POST", f"/v1/sessions/{self.session_id}/close")
        self.events += body["events"]
        sid, self.session_id = self.session_id, None
        del sid
        return list(body["events"])

    def __enter__(self) -> StreamSession:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.session_id is not None:
            self.close()


class VoiceGuardClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
        self._http.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise VoiceGuardError(r.status_code, str(detail))
        return r.json() if r.content else None

    # ------------------------------------------------------------------ analysis
    def analyze_file(
        self, audio: str | Path | bytes, tenant_id: str, **metadata: Any
    ) -> dict[str, Any]:
        data = Path(audio).read_bytes() if isinstance(audio, str | Path) else audio
        meta = call_metadata(tenant_id, **metadata)
        return self._request(
            "POST",
            "/v1/analyze/file",
            json={
                "call_metadata": meta,
                "audio_base64": base64.b64encode(data).decode(),
                "filename": str(audio) if isinstance(audio, str | Path) else None,
            },
        )

    def stream(
        self,
        tenant_id: str | None = None,
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        **metadata: Any,
    ) -> StreamSession:
        s = StreamSession(self)
        if tenant_id is not None:
            codec = "g711u" if encoding == "mulaw" else "g711a" if encoding == "alaw" else "pcm"
            s.start(
                call_metadata(
                    tenant_id, codec_hint=codec, source_sample_rate=sample_rate, **metadata
                ),
                encoding=encoding,
                sample_rate=sample_rate,
            )
        return s

    # ------------------------------------------------------------------ enrollment
    def enroll(
        self,
        tenant_id: str,
        speaker_id: str,
        audio: str | Path | bytes,
        consent_ref: str,
        session_ref: str,
        replace: bool = False,
    ) -> dict[str, Any]:
        data = Path(audio).read_bytes() if isinstance(audio, str | Path) else audio
        return self._request(
            "POST",
            f"/v1/tenants/{tenant_id}/speakers/{speaker_id}/enrollments",
            json={
                "audio_base64": base64.b64encode(data).decode(),
                "encoding": "wav",
                "consent_ref": consent_ref,
                "session_ref": session_ref,
                "replace": replace,
            },
        )

    def delete_speaker(self, tenant_id: str, speaker_id: str) -> None:
        self._request("DELETE", f"/v1/tenants/{tenant_id}/speakers/{speaker_id}")

    # ------------------------------------------------------------------ forensics
    def evidence(self, bundle_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/evidence/{bundle_id}")

    def session_evidence(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/v1/sessions/{session_id}/evidence")["bundles"])

    def feedback(self, bundle_id: str, label: str, analyst: str, note: str = "") -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/evidence/{bundle_id}/feedback",
            json={"label": label, "analyst": analyst, "note": note},
        )

    # ------------------------------------------------------------------ webhooks
    def add_webhook(
        self, tenant_id: str, url: str, events: Iterable[str] = ("policy_decision",)
    ) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/tenants/{tenant_id}/webhooks", json={"url": url, "events": list(events)}
        )

    @staticmethod
    def verify_webhook(
        secret: str, body: bytes, signature_header: str, tolerance_s: int = 300
    ) -> bool:
        return _verify_signature(secret, body, signature_header, tolerance_s)


def grpc_stream(
    target: str,
    api_key: str,
    metadata: dict[str, Any],
    chunks: Iterable[bytes],
    encoding: str = "pcm_s16le",
    sample_rate: int = 16000,
    secure: bool = False,
) -> Iterator[Any]:
    """Bidirectional gRPC AnalyzeStream. Yields proto RiskEvent messages."""
    import grpc

    from packages.vg_core.models import CallMetadata
    from services.api_gateway.proto_codec import load, to_proto

    pb2, pb2_grpc = load()
    enc = {"pcm_s16le": 1, "mulaw": 2, "alaw": 3}[encoding]
    channel = (
        grpc.secure_channel(target, grpc.ssl_channel_credentials())
        if secure
        else grpc.insecure_channel(target)
    )
    stub = pb2_grpc.VoiceIntegrityStub(channel)

    def requests() -> Iterator[Any]:
        yield pb2.StreamRequest(
            call_metadata=to_proto(CallMetadata.model_validate(metadata), pb2.CallMetadata)
        )
        for i, c in enumerate(chunks):
            yield pb2.StreamRequest(
                audio_chunk=pb2.AudioChunk(
                    session_id=metadata["session_id"],
                    seq=i,
                    sample_rate=sample_rate,
                    encoding=enc,
                    payload=c,
                )
            )

    yield from stub.AnalyzeStream(requests(), metadata=[("authorization", f"Bearer {api_key}")])
