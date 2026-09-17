"""gRPC server: bidirectional AnalyzeStream, AnalyzeFile, Enroll (B12-T01).

Auth: every call must carry ``authorization: Bearer <key>`` (or ``x-api-key``)
metadata; scopes and tenant isolation are checked per RPC. The first
``StreamRequest`` of ``AnalyzeStream`` must be ``call_metadata``; its
``tenant_id`` must match the key's tenant.

mTLS: pass ``server_credentials(cert, key, ca, require_client_auth=True)``.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Iterator
from concurrent import futures
from typing import Any

import grpc
import numpy as np

from packages.vg_core.models import CallMetadata
from services.api_gateway.auth import ApiKey, AuthError, KeyStore
from services.api_gateway.pipeline import Event, Services, SessionPipeline
from services.api_gateway.proto_codec import load, to_proto

pb2, pb2_grpc = load()

_EVENT_FIELD = {
    "window_score": ("window_score", "FusedWindowScore"),
    "session_risk": ("session_risk", "SessionRisk"),
    "policy_decision": ("policy_decision", "PolicyDecision"),
    "context_signals": ("context_signals", "ContextSignals"),
}
_ENCODINGS = {1: "pcm_s16le", 2: "mulaw", 3: "alaw"}


def event_to_proto(ev: Event) -> Any:
    field, cls = _EVENT_FIELD[ev.type]
    return pb2.RiskEvent(**{field: to_proto(ev.data, getattr(pb2, cls))})


def decode_file(audio: bytes) -> tuple[np.ndarray, int]:
    """WAV/FLAC/OGG via soundfile; otherwise assume 16 kHz s16le PCM."""
    import soundfile as sf

    try:
        pcm, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        return pcm, int(sr)
    except Exception:  # noqa: BLE001 - not a container format: raw PCM
        return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0, 16000


def run_file(pipe: SessionPipeline, pcm: np.ndarray, sr: int) -> list[Event]:
    from ml.data.channel.base import resample

    x = resample(pcm, sr, 16000) if sr != 16000 else pcm.astype(np.float32)
    events = []
    for i in range(0, len(x), 16000):
        events += pipe.push_pcm16k(x[i : i + 16000])
    return events + pipe.close()


class VoiceIntegrityServicer(pb2_grpc.VoiceIntegrityServicer):  # type: ignore[misc,name-defined]
    def __init__(self, services: Services, keys: KeyStore, enrollment: Any = None) -> None:
        self.services = services
        self.keys = keys
        self.enrollment = enrollment

    def _auth(self, context: grpc.ServicerContext, scope: str) -> ApiKey:
        md = dict(context.invocation_metadata())
        raw = md.get("x-api-key") or (
            md.get("authorization", "")[7:]
            if md.get("authorization", "").lower().startswith("bearer ")
            else None
        )
        try:
            key = self.keys.authenticate(raw)
            if scope not in key.scopes and "admin" not in key.scopes:
                raise AuthError(403, f"API key lacks scope '{scope}'")
            self.keys.charge_request(key)
            return key
        except AuthError as e:
            code = {
                401: grpc.StatusCode.UNAUTHENTICATED,
                403: grpc.StatusCode.PERMISSION_DENIED,
                429: grpc.StatusCode.RESOURCE_EXHAUSTED,
            }[e.status]
            context.abort(code, e.message)
            raise  # unreachable; abort raises

    def _meta(self, proto_meta: Any, key: ApiKey, context: grpc.ServicerContext) -> CallMetadata:
        d = from_proto_dict(proto_meta)
        d.setdefault("started_at", dt.datetime.now(dt.UTC).isoformat())
        d.setdefault("tenant_id", key.tenant_id)
        if d["tenant_id"] != key.tenant_id:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "call_metadata.tenant_id does not match API key"
            )
        try:
            return CallMetadata.model_validate(d)
        except Exception as exc:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid call_metadata: {exc}")
            raise

    def AnalyzeStream(
        self, request_iterator: Iterator[Any], context: grpc.ServicerContext
    ) -> Iterator[Any]:  # noqa: N802
        key = self._auth(context, "stream")
        pipe: SessionPipeline | None = None
        for req in request_iterator:
            which = req.WhichOneof("payload")
            if which == "call_metadata":
                if pipe is not None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "call_metadata sent twice")
                meta = self._meta(req.call_metadata, key, context)
                pipe = SessionPipeline(
                    meta, self.services, on_audio_seconds=lambda s: self.keys.charge_audio(key, s)
                )
                yield pb2.RiskEvent(call_metadata=to_proto(meta, pb2.CallMetadata))
            elif which == "audio_chunk":
                if pipe is None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION, "first message must be call_metadata"
                    )
                    return
                ch = req.audio_chunk
                try:
                    events = pipe.push_chunk(
                        ch.payload,
                        _ENCODINGS.get(ch.encoding, "pcm_s16le"),
                        ch.sample_rate or 16000,
                    )
                except AuthError as e:
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, e.message)
                    return
                for ev in events:
                    yield event_to_proto(ev)
        if pipe is not None:
            for ev in pipe.close():
                yield event_to_proto(ev)

    def AnalyzeFile(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        key = self._auth(context, "analyze")
        meta = self._meta(request.metadata, key, context)
        pcm, sr = decode_file(request.audio)
        self.keys.charge_audio(key, len(pcm) / sr)
        pipe = SessionPipeline(meta, self.services)
        events = run_file(pipe, pcm, sr)
        risk = pipe.last_risk
        resp = pb2.AnalyzeFileResponse(
            windows=[
                to_proto(e.data, pb2.FusedWindowScore) for e in events if e.type == "window_score"
            ]
        )
        if risk is not None:
            resp.session_risk.CopyFrom(to_proto(risk, pb2.SessionRisk))
        if pipe.decisions:
            resp.policy.CopyFrom(to_proto(pipe.decisions[-1].decision, pb2.PolicyDecision))
        if pipe.context is not None:
            resp.context.CopyFrom(to_proto(pipe.context.report().signals, pb2.ContextSignals))
        return resp

    def Enroll(
        self, request_iterator: Iterator[Any], context: grpc.ServicerContext
    ) -> Any:  # noqa: N802
        key = self._auth(context, "enroll")
        if self.enrollment is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "enrollment service not configured")
        parts: list[np.ndarray] = []
        speaker = tenant = session_ref = ""
        for req in request_iterator:
            speaker, tenant, session_ref = (
                req.speaker_id,
                req.tenant_id or key.tenant_id,
                req.session_id,
            )
            if tenant != key.tenant_id:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "tenant mismatch")
            if req.chunk.payload:
                from packages.vg_audio.resample import resample_chunk

                parts.append(
                    resample_chunk(
                        req.chunk.payload,
                        _ENCODINGS.get(req.chunk.encoding, "pcm_s16le"),
                        req.chunk.sample_rate or 16000,
                    )
                )
            if req.finalize:
                break
        md = dict(context.invocation_metadata())
        consent = md.get("x-consent-ref", "")
        pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        res = self.enrollment.enroll(tenant, speaker, pcm, 16000, consent, session_ref or "grpc")
        return pb2.EnrollResponse(
            success=res.accepted,
            speaker_id=speaker,
            message="; ".join(res.reasons) or "enrolled",
            quality_score=float(min(max(res.snr_db / 40, 0), 1)),
        )


def from_proto_dict(msg: Any) -> dict[str, Any]:
    from google.protobuf import json_format

    from services.api_gateway.proto_codec import _from_proto_enums

    d = json_format.MessageToDict(msg, preserving_proto_field_name=True)
    return {k: v for k, v in _from_proto_enums(d).items() if v not in ("", None)}


def server_credentials(
    cert_pem: bytes, key_pem: bytes, ca_pem: bytes | None = None, require_client_auth: bool = False
) -> grpc.ServerCredentials:
    return grpc.ssl_server_credentials(
        [(key_pem, cert_pem)], root_certificates=ca_pem, require_client_auth=require_client_auth
    )


def serve(
    services: Services,
    keys: KeyStore,
    address: str = "[::]:50051",
    enrollment: Any = None,
    credentials: grpc.ServerCredentials | None = None,
    max_workers: int = 16,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_VoiceIntegrityServicer_to_server(
        VoiceIntegrityServicer(services, keys, enrollment), server
    )
    port = (
        server.add_secure_port(address, credentials)
        if credentials is not None
        else server.add_insecure_port(address)
    )
    server.start()
    server.bound_port = port  # type: ignore[attr-defined]  # useful when binding to port 0
    return server
