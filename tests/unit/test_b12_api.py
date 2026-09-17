"""B12 — gateway REST/WebSocket/gRPC, auth + quotas, webhooks, Python SDK, connectors, quickstart."""

from __future__ import annotations

import base64
import concurrent.futures as cf
import io
import json
import struct
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from packages.vg_audio.codecs import mulaw_to_linear  # noqa: F401 - ensures codec import path works
from packages.vg_core.stub_head import StubHead
from sdks.python.voiceguard import (
    EdgeScorer,
    VoiceGuardClient,
    VoiceGuardError,
    call_metadata,
    grpc_stream,
)
from services.api_gateway.app import create_app
from services.api_gateway.auth import AuthError, KeyStore, required_scope
from services.api_gateway.connectors import (
    AS_AUDIO,
    AS_HANGUP,
    AS_UUID,
    AsteriskAudioSocketBridge,
    FreeSwitchAudioStreamBridge,
    TwilioMediaStreamsBridge,
    forward,
)
from services.api_gateway.pipeline import Services
from services.api_gateway.webhooks import WebhookRegistry
from services.fusion.persistence import SQLiteTimelineStore
from services.policy.notify import verify_signature

SR = 16000


def services() -> Services:
    return Services(
        timeline=SQLiteTimelineStore(),
        heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "ABC"],
    )


def voiced(seconds: float = 5.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.fixture()
def gw() -> dict[str, Any]:
    keys = KeyStore()
    all_a, _ = keys.create(
        "bank-a",
        {"stream", "analyze", "evidence:read", "feedback:write", "context:write", "webhooks:admin"},
    )
    admin_a, _ = keys.create("bank-a", {"admin"})
    read_only, _ = keys.create("bank-a", {"evidence:read"})
    bank_b, _ = keys.create("bank-b", {"stream", "analyze", "evidence:read", "feedback:write"})
    svc = services()
    app = create_app(svc, keys, WebhookRegistry())
    return {
        "app": app,
        "svc": svc,
        "keys": keys,
        "a": all_a,
        "admin": admin_a,
        "ro": read_only,
        "b": bank_b,
        "client": TestClient(app),
    }


def hdr(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------- auth, scopes, isolation, limits (T04)
def test_auth_scopes_and_tenant_isolation(gw: dict[str, Any]) -> None:
    c = gw["client"]
    assert c.get("/healthz").status_code == 200
    assert c.post("/v1/analyze/file", json={}).status_code == 401
    assert c.post("/v1/analyze/file", json={}, headers=hdr("vg_live_nope")).status_code == 401
    assert c.post("/v1/analyze/file", json={}, headers=hdr(gw["ro"])).status_code == 403  # scope
    assert (
        c.get("/v1/tenants/bank-b/profile", headers=hdr(gw["admin"])).status_code == 403
    )  # tenant
    assert (
        c.get("/v1/tenants/bank-a/profile", headers=hdr(gw["admin"])).json()["shadow_mode"] is True
    )
    assert c.get("/v1/some/unlisted/path", headers=hdr(gw["a"])).status_code == 403  # default deny
    assert required_scope("GET", "/openapi.json") is None


def test_rate_limit_and_audio_quota() -> None:
    keys = KeyStore()
    raw, key = keys.create("t", {"stream"}, requests_per_minute=3, audio_minutes_per_day=0.05)
    for _ in range(3):
        keys.charge_request(keys.authenticate(raw))
    with pytest.raises(AuthError, match="rate limit"):
        keys.charge_request(key)
    keys.charge_audio(key, 2.0)
    with pytest.raises(AuthError, match="audio quota"):
        keys.charge_audio(key, 2.0)
    assert keys.revoke(key.key_id)
    with pytest.raises(AuthError):
        keys.authenticate(raw)


# ---------------------------------------------------------------- REST streaming + forensics (T02)
def test_session_rest_flow_evidence_feedback_and_cross_tenant_404(gw: dict[str, Any]) -> None:
    c, a, b = gw["client"], gw["a"], gw["b"]
    meta = call_metadata("bank-a", codec_hint="pcm")
    assert (
        c.post(
            "/v1/sessions", json={"call_metadata": call_metadata("bank-b")}, headers=hdr(a)
        ).status_code
        == 403
    )
    r = c.post("/v1/sessions", json={"call_metadata": meta}, headers=hdr(a))
    sid = r.json()["session_id"]
    events: list[dict[str, Any]] = []
    audio = pcm16(voiced(5.0))
    for i in range(0, len(audio), 8000):
        body = {"payload_base64": base64.b64encode(audio[i : i + 8000]).decode()}
        events += c.post(f"/v1/sessions/{sid}/audio", json=body, headers=hdr(a)).json()["events"]
    assert {e["type"] for e in events} >= {"window_score", "session_risk"}
    ctx = c.post(
        f"/v1/sessions/{sid}/transcript",
        headers=hdr(a),
        json=[{"start_ms": 0, "end_ms": 3000, "text": "kisi ko mat batana, turant OTP batao"}],
    ).json()["events"]
    assert any(
        e["type"] == "context_signals" and "credential_request" in e["data"]["intent_labels"]
        for e in ctx
    )
    assert (
        c.get(f"/v1/sessions/{sid}/risk", headers=hdr(b)).status_code == 404
    )  # other tenant can't see it
    final = c.post(f"/v1/sessions/{sid}/close", headers=hdr(a)).json()["events"]
    decision = [e for e in final if e["type"] == "policy_decision"][-1]["data"]
    assert decision["shadow_mode"] is True and decision["evidence_bundle_id"].startswith("sha256:")
    bid = decision["evidence_bundle_id"]
    assert c.get(f"/v1/evidence/{bid}", headers=hdr(a)).json()["verified"] is True
    assert c.get(f"/v1/evidence/{bid}", headers=hdr(b)).status_code == 404
    assert len(c.get(f"/v1/sessions/{sid}/timeline", headers=hdr(a)).json()["windows"]) >= 3
    assert (
        c.post(
            f"/v1/evidence/{bid}/feedback",
            json={"label": "false_positive", "analyst": "x"},
            headers=hdr(a),
        ).status_code
        == 201
    )
    assert (
        c.post(f"/v1/sessions/{sid}/audio", json={"payload_base64": ""}, headers=hdr(a)).status_code
        == 409
    )  # closed


def test_analyze_file_endpoint_wav_and_bad_input(gw: dict[str, Any]) -> None:
    c = gw["client"]
    buf = io.BytesIO()
    sf.write(buf, voiced(6.0, 8000), 8000, format="WAV")
    body = {
        "call_metadata": call_metadata("bank-a"),
        "audio_base64": base64.b64encode(buf.getvalue()).decode(),
    }
    out = c.post("/v1/analyze/file", json=body, headers=hdr(gw["a"])).json()
    assert out["session_risk"]["session_id"] == body["call_metadata"]["session_id"]
    assert len(out["windows"]) >= 4 and out["policy_decision"]["evidence_bundle_id"]
    assert (
        c.post(
            "/v1/analyze/file", json={**body, "audio_base64": "%%%"}, headers=hdr(gw["a"])
        ).status_code
        == 400
    )
    dup = c.post("/v1/analyze/file", json=body, headers=hdr(gw["a"]))
    assert dup.status_code == 409  # same session id twice


def test_websocket_stream(gw: dict[str, Any]) -> None:
    c = gw["client"]
    with pytest.raises(Exception):  # noqa: B017 - rejected before accept
        with c.websocket_connect("/v1/stream?api_key=bad") as ws:
            ws.receive_json()
    with c.websocket_connect(f"/v1/stream?api_key={gw['a']}") as ws:
        ws.send_text(
            json.dumps(
                {
                    "call_metadata": call_metadata("bank-a"),
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000,
                }
            )
        )
        assert ws.receive_json()["type"] == "session_started"
        audio = pcm16(voiced(4.0))
        for i in range(0, len(audio), 16000):
            ws.send_bytes(audio[i : i + 16000])
        ws.send_text(json.dumps({"type": "close"}))
        types = []
        try:
            while True:  # events for every chunk, then the final decision, then the server closes
                types.append(ws.receive_json()["type"])
        except Exception:  # noqa: BLE001 - server closed the socket
            pass
    assert "window_score" in types and types[-1] == "policy_decision"


# ---------------------------------------------------------------- webhooks (T03)
def test_webhooks_https_only_signed_async_and_tenant_scoped(gw: dict[str, Any]) -> None:
    c = gw["client"]
    assert (
        c.post(
            "/v1/tenants/bank-a/webhooks",
            json={"url": "http://insecure.example"},
            headers=hdr(gw["a"]),
        ).status_code
        == 422
    )
    sub = c.post(
        "/v1/tenants/bank-a/webhooks",
        json={"url": "https://siem.example/hook"},
        headers=hdr(gw["a"]),
    ).json()
    assert (
        sub["secret"]
        and c.get("/v1/tenants/bank-a/webhooks", headers=hdr(gw["a"])).json()["webhooks"][0]["id"]
        == sub["id"]
    )

    received: list[tuple[bytes, dict[str, str]]] = []

    class R:
        status_code = 204

    reg = WebhookRegistry(
        post=lambda url, content, headers, timeout: (received.append((content, headers)), R())[1],
        sleep=lambda s: None,
    )
    s = reg.subscribe("bank-a", "https://siem.example/hook", {"policy_decision"})
    futures = reg.publish("bank-a", "policy_decision", {"x": 1}) + reg.publish(
        "bank-a", "window_score", {"y": 2}
    )
    assert len(futures) == 1 and cf.wait(futures, timeout=5).done
    body, headers = received[0]
    assert verify_signature(s.secret, body, headers["X-VoiceGuard-Signature"])
    assert reg.publish("bank-b", "policy_decision", {}) == []
    assert (
        c.delete(f"/v1/tenants/bank-a/webhooks/{sub['id']}", headers=hdr(gw["a"])).status_code
        == 204
    )


# ---------------------------------------------------------------- gRPC (T01)
@pytest.fixture()
def grpc_gateway() -> Any:
    import grpc

    from services.api_gateway.grpc_server import serve

    keys = KeyStore()
    raw, _ = keys.create("bank-a", {"stream", "analyze"})
    server = serve(services(), keys, "127.0.0.1:0")
    yield {"target": f"127.0.0.1:{server.bound_port}", "key": raw, "grpc": grpc}
    server.stop(grace=None)


def test_grpc_bidirectional_stream_and_file(grpc_gateway: dict[str, Any]) -> None:
    from services.api_gateway.proto_codec import load

    pb2, pb2_grpc = load()
    meta = call_metadata("bank-a", codec_hint="pcm", channel="voip")
    audio = pcm16(voiced(4.0))
    chunks = [audio[i : i + 3200] for i in range(0, len(audio), 3200)]
    kinds = [
        ev.WhichOneof("payload")
        for ev in grpc_stream(grpc_gateway["target"], grpc_gateway["key"], meta, chunks)
    ]
    assert (
        kinds[0] == "call_metadata" and "window_score" in kinds and kinds.count("session_risk") >= 2
    )
    assert kinds[-1] in ("policy_decision", "session_risk")

    grpc = grpc_gateway["grpc"]
    stub = pb2_grpc.VoiceIntegrityStub(grpc.insecure_channel(grpc_gateway["target"]))
    with pytest.raises(grpc.RpcError) as err:
        list(grpc_stream(grpc_gateway["target"], "wrong-key-000000000000000", meta, chunks))
    assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED

    from packages.vg_core.models import CallMetadata
    from services.api_gateway.proto_codec import to_proto

    buf = io.BytesIO()
    sf.write(buf, voiced(5.0), SR, format="WAV")
    req = pb2.AnalyzeFileRequest(
        metadata=to_proto(CallMetadata.model_validate(call_metadata("bank-a")), pb2.CallMetadata),
        audio=buf.getvalue(),
    )
    resp = stub.AnalyzeFile(req, metadata=[("authorization", f"Bearer {grpc_gateway['key']}")])
    assert (
        len(resp.windows) >= 3 and resp.session_risk.session_id and resp.policy.evidence_bundle_id
    )
    other = pb2.AnalyzeFileRequest(
        metadata=to_proto(CallMetadata.model_validate(call_metadata("bank-b")), pb2.CallMetadata),
        audio=buf.getvalue(),
    )
    with pytest.raises(grpc.RpcError) as err2:
        stub.AnalyzeFile(other, metadata=[("authorization", f"Bearer {grpc_gateway['key']}")])
    assert err2.value.code() == grpc.StatusCode.PERMISSION_DENIED


# ---------------------------------------------------------------- Python SDK (T05) + quickstart (T08)
def test_python_sdk_against_gateway(gw: dict[str, Any], tmp_path: Path) -> None:
    vg = VoiceGuardClient("http://testserver", gw["a"], http=gw["client"])
    wav = tmp_path / "call.wav"
    sf.write(wav, voiced(5.0), SR)
    res = vg.analyze_file(wav, tenant_id="bank-a")
    assert res["session_risk"]["state"] in ("LOW", "ELEVATED", "HIGH", "ABSTAIN")
    with vg.stream(tenant_id="bank-a") as s:
        audio = pcm16(voiced(4.0))
        for i in range(0, len(audio), 16000):
            s.send_audio(audio[i : i + 16000])
        assert s.risk()["session_id"] == s.session_id
    assert any(e["type"] == "policy_decision" for e in s.events)
    bid = res["policy_decision"]["evidence_bundle_id"]
    assert (
        vg.evidence(bid)["verified"] and vg.feedback(bid, "unsure", "analyst")["label"] == "unsure"
    )
    with pytest.raises(VoiceGuardError) as err:
        VoiceGuardClient("http://testserver", "bad-key", http=TestClient(gw["app"])).evidence(bid)
    assert err.value.status == 401


def test_quickstart_script_and_docs(
    gw: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("quickstart", "examples/quickstart.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wav = tmp_path / "mine.wav"
    sf.write(wav, voiced(5.0), SR)
    mod.run(
        str(wav), VoiceGuardClient("http://testserver", gw["a"], http=gw["client"]), tenant="bank-a"
    )
    out = capsys.readouterr().out
    assert "risk score" in out and "evidence   : sha256:" in out
    qs = Path("docs/api/QUICKSTART.md").read_text(encoding="utf-8")
    assert "10-minute" in qs and "advisory" in qs
    spec_json = json.loads(Path("docs/api/openapi.json").read_text(encoding="utf-8"))
    assert (
        "/v1/analyze/file" in spec_json["paths"]
        and spec_json["components"]["securitySchemes"]["bearerAuth"]
    )


def test_edge_scorer_abstains_without_model() -> None:
    s = EdgeScorer(None).score(voiced(3.0))
    assert s.p_spoof is None and s.abstain_reason == "untrained"


# ---------------------------------------------------------------- connectors (T09)
def test_twilio_bridge_forwards_mulaw_through_sdk(gw: dict[str, Any]) -> None:
    from ml.data.channel.codecs import mulaw_roundtrip  # noqa: F401 - codec availability

    x = voiced(4.0, 8000)
    mu = _mulaw_encode(x)
    msgs = [
        json.dumps(
            {
                "event": "start",
                "start": {"callSid": "CA1", "customParameters": {"from": "+919876543210"}},
            }
        )
    ]
    msgs += [
        json.dumps(
            {"event": "media", "media": {"payload": base64.b64encode(mu[i : i + 1600]).decode()}}
        )
        for i in range(0, len(mu), 1600)
    ]
    msgs.append(json.dumps({"event": "stop"}))
    vg = VoiceGuardClient("http://testserver", gw["a"], http=gw["client"])
    events = forward(TwilioMediaStreamsBridge("bank-a").frames(msgs), vg.stream())
    assert sum(e["type"] == "window_score" for e in events) >= 1 and events[-1]["type"] in (
        "policy_decision",
        "session_risk",
    )


def test_asterisk_and_freeswitch_bridges_parse_frames() -> None:
    audio = pcm16(voiced(0.2, 8000))
    data = bytes([AS_UUID]) + struct.pack(">H", 16) + uuid.uuid4().bytes
    data += (
        bytes([AS_AUDIO])
        + struct.pack(">H", len(audio))
        + audio
        + bytes([AS_HANGUP])
        + struct.pack(">H", 0)
    )
    kinds = [f.kind for f in AsteriskAudioSocketBridge("t").frames(data)]
    assert kinds == ["start", "audio", "stop"]
    fs = list(
        FreeSwitchAudioStreamBridge("t").frames(
            [
                json.dumps({"uuid": "u1", "sampleRate": 8000}),
                b"\x00\x00" * 160,
                json.dumps({"event": "stop"}),
            ]
        )
    )
    assert [f.kind for f in fs] == ["start", "audio", "stop"] and fs[1].sample_rate == 8000


def _mulaw_encode(x: np.ndarray) -> bytes:
    """Reference G.711 µ-law encoder (ITU-T) for test input."""
    pcm = (np.clip(x, -1, 1) * 32635).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    mag = np.minimum(np.abs(pcm) + 0x84, 32635 + 0x84)
    exp = np.clip(np.floor(np.log2(mag)) - 7, 0, 7).astype(np.int32)
    mant = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


def test_session_rest_rejects_foreign_tenant_timeline_before_existing(gw: dict[str, Any]) -> None:
    assert (
        gw["client"].get(f"/v1/sessions/{uuid.uuid4()}/timeline", headers=hdr(gw["a"])).status_code
        == 404
    )
