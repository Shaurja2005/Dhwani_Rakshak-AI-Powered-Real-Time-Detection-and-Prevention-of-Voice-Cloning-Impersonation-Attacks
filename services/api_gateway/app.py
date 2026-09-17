"""REST + WebSocket API gateway (B12-T02, T03, T04).

All routes authenticate with an API key (``Authorization: Bearer <key>``,
``X-API-Key``, or ``?api_key=`` for browser WebSockets), are scope-checked and
tenant-isolated (``auth.py``), rate-limited and quota-metered.

Streaming (browser / JS SDK / reference connectors):
    WS  /v1/stream
        → {"call_metadata": {...}, "encoding": "pcm_s16le", "sample_rate": 16000}
        → binary audio frames
        → {"type": "transcript", "segments": [...]}   (optional)
        → {"type": "close"}
        ← {"type": "window_score"|"session_risk"|"context_signals"|"policy_decision", "data": {...}}

Session REST (for integrators that cannot hold a socket):
    POST /v1/sessions  ·  POST /v1/sessions/{id}/audio  ·  POST /v1/sessions/{id}/transcript
    POST /v1/sessions/{id}/close  ·  GET /v1/sessions/{id}/risk

Post-call and platform:
    POST /v1/analyze/file
    GET  /v1/sessions/{id}/timeline · GET /v1/sessions/{id}/evidence · GET /v1/evidence/{bundle_id}
    POST /v1/evidence/{bundle_id}/feedback
    /v1/tenants/{t}/speakers…   (enrollment, B7)       /v1/sessions/{id}/challenges… (liveness, B8)
    GET|PUT /v1/tenants/{t}/profile · POST /v1/tenants/{t}/shadow
    POST|GET|DELETE /v1/tenants/{t}/webhooks · POST /v1/tenants/{t}/keys
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import queue
import threading
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from packages.vg_core.logging import get_logger
from packages.vg_core.models import CallMetadata
from services.api_gateway.auth import (
    SCOPES,
    ApiKey,
    AuthError,
    KeyStore,
    authorize,
    extract_key,
    required_scope,
)
from services.api_gateway.grpc_server import decode_file, run_file
from services.api_gateway.pipeline import Event, Services, SessionPipeline
from services.api_gateway.webhooks import WebhookRegistry
from services.context.asr import Segment
from services.policy.evidence import verify_bundle
from services.policy.feedback import FeedbackStore
from services.policy.profiles import TenantProfile

log = get_logger(__name__)


# ---------------------------------------------------------------- request bodies
class SessionCreate(BaseModel):
    call_metadata: CallMetadata


class AudioIn(BaseModel):
    payload_base64: str
    encoding: Literal["pcm_s16le", "mulaw", "alaw"] = "pcm_s16le"
    sample_rate: int = Field(16000, ge=8000, le=48000)


class SegmentIn(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None
    final: bool = True


class FileIn(BaseModel):
    call_metadata: CallMetadata
    audio_base64: str
    filename: str | None = None


class FeedbackIn(BaseModel):
    label: Literal["true_positive", "false_positive", "unsure"]
    analyst: str
    note: str = ""


class WebhookIn(BaseModel):
    url: str
    events: list[str] = ["policy_decision"]


class KeyIn(BaseModel):
    scopes: list[str]
    requests_per_minute: int = 600
    audio_minutes_per_day: float = 10_000.0


class ShadowIn(BaseModel):
    shadow_mode: bool


def create_app(
    services: Services | None = None,
    keys: KeyStore | None = None,
    webhooks: WebhookRegistry | None = None,
    enrollment: Any = None,
    challenges: Any = None,
) -> FastAPI:
    services = services or Services()
    keys = keys or KeyStore()
    webhooks = webhooks or WebhookRegistry()
    feedback = FeedbackStore(services.evidence)
    sessions: dict[str, SessionPipeline] = {}
    owners: dict[str, str] = {}  # session_id -> tenant_id (kept after close for forensics)
    lock = threading.Lock()

    app = FastAPI(
        title="VoiceGuard API",
        version="0.1.0",
        description="Real-time voice-clone impersonation risk. Detection is advisory, never authoritative.",
    )
    app.state.services, app.state.keys, app.state.webhooks, app.state.sessions = (
        services,
        keys,
        webhooks,
        sessions,
    )

    @app.middleware("http")
    async def auth_mw(request: Request, call_next: Any) -> Any:
        try:
            raw = extract_key(
                {k.lower(): v for k, v in request.headers.items()}, dict(request.query_params)
            )
            if required_scope(request.method, request.url.path) is None:  # health, docs, static UI
                return await call_next(request)
            key = keys.authenticate(raw)
            authorize(key, request.method, request.url.path)
            keys.charge_request(key)
            request.state.key = key
        except AuthError as e:
            return JSONResponse({"detail": e.message}, status_code=e.status)
        return await call_next(request)

    def key_of(request: Request) -> ApiKey:
        return request.state.key  # type: ignore[no-any-return]

    def own_session(session_id: str, key: ApiKey) -> None:
        tenant = owners.get(session_id)
        if tenant is None:
            raise HTTPException(404, "unknown session")
        if tenant != key.tenant_id:
            raise HTTPException(404, "unknown session")  # do not reveal other tenants' sessions

    watchers: dict[str, list[queue.Queue[dict[str, Any]]]] = defaultdict(list)

    def emit(tenant_id: str, events: list[Event]) -> list[dict[str, Any]]:
        out = []
        for ev in events:
            j = ev.to_json()
            j["session_id"] = getattr(ev.data, "session_id", None)
            webhooks.publish(tenant_id, ev.type, j["data"])
            with lock:
                targets = list(watchers.get(j["session_id"] or "", []))
            for q in targets:  # agent UIs watching this call (B13)
                with contextlib.suppress(queue.Full):
                    q.put_nowait(j)
            out.append(j)
        return out

    def new_pipeline(meta: CallMetadata, key: ApiKey) -> SessionPipeline:
        if meta.tenant_id != key.tenant_id:
            raise HTTPException(403, "call_metadata.tenant_id does not match API key")
        with lock:
            if meta.session_id in owners:
                raise HTTPException(409, "session already exists")
            pipe = SessionPipeline(
                meta, services, on_audio_seconds=lambda s: keys.charge_audio(key, s)
            )
            sessions[meta.session_id] = pipe
            owners[meta.session_id] = meta.tenant_id
        return pipe

    def live(session_id: str, key: ApiKey) -> SessionPipeline:
        own_session(session_id, key)
        pipe = sessions.get(session_id)
        if pipe is None:
            raise HTTPException(409, "session is closed")
        return pipe

    @app.get("/", include_in_schema=False)
    def root() -> Any:
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/ui/")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # ---------------------------------------------------------------- sessions
    @app.post("/v1/sessions", status_code=201)
    def create_session(body: SessionCreate, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        pipe = new_pipeline(body.call_metadata, key)
        return {
            "session_id": pipe.meta.session_id,
            "shadow_mode": services.profiles.get(key.tenant_id).shadow_mode,
        }

    @app.post("/v1/sessions/{session_id}/audio")
    def push_audio(session_id: str, body: AudioIn, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        pipe = live(session_id, key)
        try:
            payload = base64.b64decode(body.payload_base64, validate=True)
        except Exception as exc:
            raise HTTPException(400, "payload_base64 is not valid base64") from exc
        try:
            return {
                "events": emit(
                    key.tenant_id, pipe.push_chunk(payload, body.encoding, body.sample_rate)
                )
            }
        except AuthError as e:
            raise HTTPException(e.status, e.message) from e

    @app.post("/v1/sessions/{session_id}/transcript")
    def push_transcript(
        session_id: str, segments: list[SegmentIn], key: ApiKey = Depends(key_of)
    ) -> dict[str, Any]:
        pipe = live(session_id, key)
        segs = [Segment(s.start_ms, s.end_ms, s.text, s.language, s.final) for s in segments]
        return {"events": emit(key.tenant_id, pipe.add_transcript(segs))}

    @app.post("/v1/sessions/{session_id}/close")
    def close_session(session_id: str, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        pipe = live(session_id, key)
        events = emit(key.tenant_id, pipe.close())
        with lock:
            sessions.pop(session_id, None)
        return {"events": events}

    @app.get("/v1/sessions/{session_id}/risk")
    def session_risk(session_id: str, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        pipe = live(session_id, key)
        risk = pipe.last_risk or pipe.engine.session_risk()
        return risk.model_dump(mode="json")

    # ---------------------------------------------------------------- file analysis
    @app.post("/v1/analyze/file")
    def analyze_file(body: FileIn, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        try:
            audio = base64.b64decode(body.audio_base64, validate=True)
        except Exception as exc:
            raise HTTPException(400, "audio_base64 is not valid base64") from exc
        pcm, sr = decode_file(audio)
        try:
            keys.charge_audio(key, len(pcm) / sr)
        except AuthError as e:
            raise HTTPException(e.status, e.message) from e
        pipe = new_pipeline(body.call_metadata, key)
        events = run_file(pipe, pcm, sr)
        with lock:
            sessions.pop(pipe.meta.session_id, None)
        emit(key.tenant_id, [e for e in events if e.type == "policy_decision"])
        last = pipe.decisions[-1] if pipe.decisions else None
        return {
            "session_risk": (pipe.last_risk.model_dump(mode="json") if pipe.last_risk else None),
            "policy_decision": last.decision.model_dump(mode="json") if last else None,
            "agent_prompt": last.agent_prompt if last else "",
            "context_signals": (
                pipe.context.report().signals.model_dump(mode="json") if pipe.context else None
            ),
            "windows": [e.data.model_dump(mode="json") for e in events if e.type == "window_score"],
        }

    # ---------------------------------------------------------------- streaming
    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        try:
            key = keys.authenticate(
                extract_key({k.lower(): v for k, v in ws.headers.items()}, dict(ws.query_params))
            )
            authorize(key, "GET", "/v1/stream")
            keys.charge_request(key)
        except AuthError as e:
            await ws.close(code=4001 if e.status == 401 else 4003, reason=e.message)
            return
        await ws.accept()
        pipe: SessionPipeline | None = None
        encoding, rate = "pcm_s16le", 16000
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if pipe is None:
                        await ws.send_json({"type": "error", "detail": "send call_metadata first"})
                        continue
                    for ev in emit(key.tenant_id, pipe.push_chunk(msg["bytes"], encoding, rate)):
                        await ws.send_json(ev)
                    continue
                data = json.loads(msg.get("text") or "{}")
                if "call_metadata" in data:
                    meta = CallMetadata.model_validate(data["call_metadata"])
                    encoding, rate = data.get("encoding", encoding), int(
                        data.get("sample_rate", rate)
                    )
                    pipe = new_pipeline(meta, key)
                    await ws.send_json(
                        {"type": "session_started", "data": {"session_id": meta.session_id}}
                    )
                elif data.get("type") == "transcript" and pipe is not None:
                    segs = [
                        Segment(**{k: s[k] for k in ("start_ms", "end_ms", "text") if k in s})
                        for s in data["segments"]
                    ]
                    for ev in emit(key.tenant_id, pipe.add_transcript(segs)):
                        await ws.send_json(ev)
                elif data.get("type") == "close":
                    break
        except (WebSocketDisconnect, HTTPException, AuthError, ValueError) as exc:
            log.warning("stream_ended", error=str(exc))
            detail = getattr(exc, "detail", None) or getattr(exc, "message", None) or str(exc)
            try:
                await ws.send_json({"type": "error", "detail": detail})
            except Exception:  # noqa: BLE001, S110 - socket already gone
                pass
        finally:
            if pipe is not None and not pipe.closed:
                final = emit(key.tenant_id, pipe.close())
                with lock:
                    sessions.pop(pipe.meta.session_id, None)
                try:
                    for ev in final:
                        await ws.send_json(ev)
                    await ws.close()
                except Exception:  # noqa: BLE001, S110 - client disconnected first
                    pass

    # ---------------------------------------------------------------- forensics
    @app.get("/v1/sessions/{session_id}/timeline")
    def timeline(session_id: str, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        own_session(session_id, key)
        if services.timeline is None:
            raise HTTPException(501, "timeline store not configured")
        return {
            "session_id": session_id,
            "windows": [asdict(r) for r in services.timeline.session(session_id)],
        }

    @app.get("/v1/sessions/{session_id}/evidence")
    def session_evidence(session_id: str, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        own_session(session_id, key)
        return {"bundles": services.evidence.for_session(session_id)}

    def bundle_for(bundle_id: str, key: ApiKey) -> dict[str, Any]:
        b = services.evidence.get(bundle_id)
        if b is None or b.get("threshold_profile_snapshot", {}).get("tenant_id") != key.tenant_id:
            raise HTTPException(404, "no such bundle")
        return b

    @app.get("/v1/evidence/{bundle_id}")
    def evidence(bundle_id: str, key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        b = bundle_for(bundle_id, key)
        return {"bundle": b, "verified": verify_bundle(b)}

    @app.post("/v1/evidence/{bundle_id}/feedback", status_code=201)
    def post_feedback(
        bundle_id: str, body: FeedbackIn, key: ApiKey = Depends(key_of)
    ) -> dict[str, Any]:
        bundle_for(bundle_id, key)
        return feedback.record(bundle_id, body.label, body.analyst, body.note)

    # ---------------------------------------------------------------- tenant admin
    @app.get("/v1/tenants/{tenant_id}/profile")
    def get_profile(tenant_id: str) -> dict[str, Any]:
        return asdict(services.profiles.get(tenant_id))

    @app.put("/v1/tenants/{tenant_id}/profile")
    def put_profile(tenant_id: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            prof = TenantProfile(**{**body, "tenant_id": tenant_id})
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        services.profiles.put(prof)
        return asdict(prof)

    @app.post("/v1/tenants/{tenant_id}/shadow")
    def set_shadow(tenant_id: str, body: ShadowIn) -> dict[str, Any]:
        return asdict(services.profiles.set_shadow(tenant_id, body.shadow_mode))

    @app.post("/v1/tenants/{tenant_id}/keys", status_code=201)
    def create_key(tenant_id: str, body: KeyIn) -> dict[str, Any]:
        if set(body.scopes) - SCOPES:
            raise HTTPException(422, f"unknown scopes {set(body.scopes) - SCOPES}")
        raw, k = keys.create(
            tenant_id,
            set(body.scopes),
            requests_per_minute=body.requests_per_minute,
            audio_minutes_per_day=body.audio_minutes_per_day,
        )
        return {
            "api_key": raw,
            "key_id": k.key_id,
            "scopes": sorted(k.scopes),
            "note": "shown once; store it securely",
        }

    @app.post("/v1/tenants/{tenant_id}/webhooks", status_code=201)
    def add_webhook(tenant_id: str, body: WebhookIn) -> dict[str, Any]:
        try:
            sub = webhooks.subscribe(tenant_id, body.url, set(body.events))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "id": sub.sub_id,
            "url": sub.url,
            "events": sorted(sub.events),
            "secret": sub.secret,
            "note": "secret shown once; verify X-VoiceGuard-Signature with it",
        }

    @app.get("/v1/tenants/{tenant_id}/webhooks")
    def list_webhooks(tenant_id: str) -> dict[str, Any]:
        return {
            "webhooks": [
                {"id": s.sub_id, "url": s.url, "events": sorted(s.events)}
                for s in webhooks.list(tenant_id)
            ]
        }

    @app.delete("/v1/tenants/{tenant_id}/webhooks/{sub_id}", status_code=204)
    def delete_webhook(tenant_id: str, sub_id: str) -> None:
        if not webhooks.remove(tenant_id, sub_id):
            raise HTTPException(404, "no such webhook")

    # ---------------------------------------------------------------- agent UI support (B13)
    @app.get("/v1/sessions")
    def list_sessions(key: ApiKey = Depends(key_of)) -> dict[str, Any]:
        with lock:
            active = [p for p in sessions.values() if p.meta.tenant_id == key.tenant_id]
        return {
            "sessions": [
                {
                    "session_id": p.meta.session_id,
                    "started_at": p.meta.started_at.isoformat(),
                    "caller_number": p.meta.caller_number,
                    "claimed_identity_id": p.meta.claimed_identity_id,
                    "channel": p.meta.channel.value,
                    "state": p.last_risk.state.value if p.last_risk else "ABSTAIN",
                    "risk_score": p.last_risk.risk_score if p.last_risk else 0,
                }
                for p in active
            ]
        }

    @app.websocket("/v1/sessions/{session_id}/watch")
    async def watch(ws: WebSocket, session_id: str) -> None:
        try:
            key = keys.authenticate(
                extract_key({k.lower(): v for k, v in ws.headers.items()}, dict(ws.query_params))
            )
            authorize(key, "GET", f"/v1/sessions/{session_id}/timeline")
            keys.charge_request(key)
            own_session(session_id, key)
        except (AuthError, HTTPException) as e:
            await ws.close(
                code=4003, reason=getattr(e, "message", None) or str(getattr(e, "detail", e))
            )
            return
        await ws.accept()
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2000)
        with lock:
            watchers[session_id].append(q)
            pipe = sessions.get(session_id)
        try:
            if pipe is not None and pipe.last_risk is not None:  # catch-up snapshot
                await ws.send_json(
                    {
                        "type": "session_risk",
                        "session_id": session_id,
                        "data": pipe.last_risk.model_dump(mode="json"),
                    }
                )
            while True:
                try:
                    msg = await asyncio.to_thread(q.get, True, 0.5)
                except queue.Empty:
                    if session_id not in sessions:
                        break
                    continue
                await ws.send_json(msg)
        except WebSocketDisconnect:
            pass
        finally:
            with lock:
                if q in watchers[session_id]:
                    watchers[session_id].remove(q)
            with contextlib.suppress(Exception):
                await ws.close()

    ui_dir = Path(__file__).resolve().parents[1] / "ui" / "public"
    if ui_dir.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    # ---------------------------------------------------------------- mounted blocks
    if enrollment is not None:
        from services.enrollment.api import create_app as enrollment_app

        app.include_router(enrollment_app(enrollment).router)
    if challenges is not None:
        from fastapi import APIRouter

        from packages.vg_models.heads.head_e_liveness.api import create_router

        def guard(session_id: str, request: Request) -> None:
            own_session(session_id, key_of(request))

        guarded = APIRouter(dependencies=[Depends(guard)])
        guarded.include_router(create_router(challenges))
        app.include_router(guarded)

    return app
