"""B11 — policy rules, tiered thresholds, actions, evidence bundles, notifications, shadow mode, feedback."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.vg_core.models import (
    ContextSignals,
    HeadID,
    HeadScore,
    PolicyDecision,
    RiskDriver,
    RiskState,
    SessionRisk,
)
from services.context.engine import ContextReport
from services.policy.actions import ABSTAIN_TEXT, LIBRARY, agent_prompt
from services.policy.engine import decide
from services.policy.evidence import (
    EvidenceIntegrityError,
    EvidenceStore,
    build_bundle,
    verify_bundle,
)
from services.policy.feedback import FeedbackStore
from services.policy.main import create_app
from services.policy.notify import (
    Broadcaster,
    Notifier,
    RecordingProvider,
    WebhookTarget,
    deliver_webhook,
    sign,
    verify_signature,
)
from services.policy.profiles import (
    ProfileStore,
    TenantProfile,
    bfsi_default,
    retail_helpline,
    wealth_desk,
)

SID = "01900b1a-0000-7000-8000-0000000b11b1"


def risk(score: int, state: RiskState = RiskState.ELEVATED) -> SessionRisk:
    return SessionRisk(
        session_id=SID,
        updated_at=dt.datetime.now(dt.UTC),
        risk_score=score,
        state=state,
        p_spoof_session_max=score / 100,
        p_spoof_session_mean=score / 200,
        abstain_ratio=0.1,
        drivers=[
            RiskDriver(
                factor="synthetic_speech_ssl",
                weight=0.6,
                detail="head A: net evidence towards synthetic",
            )
        ],
        model_versions={"A": "A@xlsr300m-nes2net-v0.3.1", "fusion": "fuse-lr-v0.1"},
    )


def ctx(
    intent: float = 0.0, labels: list[str] | None = None, tier: str = "high", tx: float = 0.6
) -> ContextReport:
    sig = ContextSignals(
        session_id=SID,
        as_of_ms=30000,
        metadata_risk=0.1,
        transaction_risk=tx,
        intent_risk=intent,
        intent_labels=labels or [],
        transcript_snippets=[],
        language_detected="en",
        code_switched=False,
    )
    return ContextReport(
        sig,
        {
            "transaction": {
                "new_beneficiary": "Money is going to a beneficiary this customer has never paid."
            },
            "intent": [f"{x}: “...” (rules)" for x in labels or []],
        },
        tier,
    )


def live(p: TenantProfile) -> TenantProfile:
    p.shadow_mode = False
    return p


# ---------------------------------------------------------------- profiles + tiers (T01, T03, T07)
def test_new_tenants_default_to_shadow_mode(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    assert (
        store.get("new-bank").shadow_mode is True
        and store.get("new-bank").name == "bfsi-default-v1"
    )
    store.set_shadow("new-bank", False)
    assert ProfileStore(tmp_path).get("new-bank").shadow_mode is False  # persisted


def test_profile_validation() -> None:
    with pytest.raises(ValueError, match="unknown actions"):
        TenantProfile(
            "x",
            "t",
            bands={"medium": {"elevated": 0.4, "high": 0.7}},
            actions={"medium": {"HIGH": ["launch_missiles"]}},
        )
    with pytest.raises(ValueError, match="elevated < high"):
        TenantProfile("x", "t", bands={"medium": {"elevated": 0.8, "high": 0.7}})


def test_tiered_thresholds_same_risk_different_band() -> None:
    p = bfsi_default("t")
    assert p.band_for(0.65, "low") == "ELEVATED"  # balance inquiry: needs 0.95 for HIGH
    assert p.band_for(0.65, "high") == "HIGH"  # high-value wire: HIGH from 0.60
    assert p.band_for(0.35, "high") == "ELEVATED" and p.band_for(0.35, "low") == "LOW"


# ---------------------------------------------------------------- engine + DoD (T01, T02, T04)
def test_dod_same_call_two_profiles_two_action_sets_both_explained() -> None:
    store = EvidenceStore()
    voice = risk(45)
    context = ctx(intent=0.4, labels=["credential_request"], tier="high", tx=0.6)
    retail = decide(live(retail_helpline("t1")), voice, context, store=store)
    wealth = decide(live(wealth_desk("t2")), voice, context, store=store)
    assert set(retail.decision.actions) != set(wealth.decision.actions)
    assert (
        "dual_approval" in wealth.decision.actions
        and "dual_approval" not in retail.decision.actions
    )
    for res, name in ((retail, "retail-helpline-v1"), (wealth, "bfsi-wealth-v2")):
        b = store.get(res.decision.evidence_bundle_id)
        assert b is not None and verify_bundle(b)
        assert b["threshold_profile_snapshot"]["name"] == name
        rationale = b["policy_decision"]["rationale"]
        assert f"profile {name}" in rationale and "final_risk=" in rationale
        assert (
            "new_beneficiary" not in rationale and "never paid" in rationale
        )  # human-readable context
        assert "synthetic_speech_ssl" in rationale and "advisory only" in rationale
        assert b["model_versions"]["A"] == "A@xlsr300m-nes2net-v0.3.1"
    assert store.verify_chain()


def test_abstain_band_never_implies_safety() -> None:
    res = decide(live(bfsi_default("t")), risk(90, RiskState.ABSTAIN), None)
    assert res.band == "ABSTAIN" and res.decision.actions == [] and res.agent_prompt == ABSTAIN_TEXT
    assert "not an indication of safety" in res.decision.rationale
    with_ctx = decide(
        live(bfsi_default("t")),
        risk(90, RiskState.ABSTAIN),
        ctx(intent=0.95, labels=["secrecy_demand"]),
    )
    assert (
        with_ctx.band != "ABSTAIN" and "agent_banner" in with_ctx.decision.actions
    )  # context still counts


def test_label_actions_and_watermark_force_callback() -> None:
    low = decide(
        live(bfsi_default("t")),
        risk(5, RiskState.LOW),
        ctx(intent=0.3, labels=["credential_request"], tier="low", tx=0.0),
    )
    assert low.band == "LOW" and low.decision.actions == ["agent_banner"]
    wm = HeadScore(
        session_id=SID,
        window_id=3,
        head_id=HeadID.F,
        raw_score=50.0,
        p_spoof=0.99,
        abstain=False,
        confidence=0.95,
        latency_ms=2,
        model_version="F@watermark-probe-v0.1.0",
        calibration_version="detector-native",
        evidence={"watermark": "present"},
    )
    res = decide(live(bfsi_default("t")), risk(5, RiskState.LOW), None, head_scores=[wm])
    assert "force_callback" in res.decision.actions and "watermark" in res.decision.rationale


def test_agent_prompt_is_plain_and_advisory() -> None:
    p = agent_prompt(
        "HIGH",
        ["agent_banner", "force_callback", "hold_transaction"],
        ["credential_request"],
        "high",
    )
    assert "call back on the number registered" in p and "one-time password" in p
    assert "fraud" not in p.lower() and "fake" not in p.lower()
    assert agent_prompt("LOW", [], [], "low") == ""
    assert set(LIBRARY) >= {"force_callback", "dual_approval", "siem_webhook"}


# ---------------------------------------------------------------- evidence (T04)
def test_evidence_bundle_hash_schema_and_tamper_detection(tmp_path: Path) -> None:
    decision = PolicyDecision(
        session_id=SID,
        decided_at=dt.datetime.now(dt.UTC),
        threshold_profile="p",
        actions=["agent_banner"],
        rationale="r",
        evidence_bundle_id="pending",
        shadow_mode=True,
        suppressed=True,
    )
    b = build_bundle(decision, risk(50), {"name": "p"}, ctx().signals)
    assert (
        b["bundle_id"].startswith("sha256:")
        and b["policy_decision"]["evidence_bundle_id"] == b["bundle_id"]
    )
    assert verify_bundle(b)
    tampered = json.loads(json.dumps(b))
    tampered["session_risk"]["risk_score"] = 1
    assert not verify_bundle(tampered)
    db = tmp_path / "ev.db"
    store = EvidenceStore(db)
    with pytest.raises(EvidenceIntegrityError):
        store.append(tampered)
    store.append(b)
    store.append(build_bundle(decision, risk(60), {"name": "p"}))
    assert store.verify_chain()
    con = sqlite3.connect(db)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE evidence SET session_id='x'")
    con.execute("DELETE FROM evidence WHERE seq=1")  # deletion is detected by the hash chain
    con.commit()
    assert not EvidenceStore(db).verify_chain()
    bad = json.loads(json.dumps(b))
    bad["unexpected"] = 1
    with pytest.raises(EvidenceIntegrityError):
        from services.policy.evidence import validate

        validate(bad)


# ---------------------------------------------------------------- notifications + shadow (T05, T07)
def test_webhook_signature_and_retry_with_backoff() -> None:
    body = b'{"a":1}'
    header = sign("s3cret", body, timestamp=1_700_000_000)
    assert verify_signature("s3cret", body, header, now=1_700_000_100)
    assert not verify_signature("wrong", body, header, now=1_700_000_100)
    assert not verify_signature("s3cret", body, header, now=1_700_009_999)  # replay window

    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    class R:
        def __init__(self, code: int) -> None:
            self.status_code = code

    codes = iter([503, 500, 200])

    def post(url: str, content: bytes, headers: dict[str, str], timeout: float) -> R:
        calls.append(headers)
        assert verify_signature("k", content, headers["X-VoiceGuard-Signature"])
        return R(next(codes))

    res = deliver_webhook(
        "https://siem.example/hook", {"x": 1}, "k", post=post, sleep=sleeps.append
    )
    assert res.delivered and res.attempts == 3 and sleeps == [0.5, 1.0]
    res4xx = deliver_webhook(
        "https://siem.example/hook", {"x": 1}, "k", post=lambda *a, **k: R(400), sleep=sleeps.append
    )
    assert not res4xx.delivered and res4xx.attempts == 1


def test_shadow_mode_records_but_never_dispatches() -> None:
    broadcaster, provider = Broadcaster(), RecordingProvider()
    q = broadcaster.subscribe("t")
    notifier = Notifier(broadcaster, provider)
    store = EvidenceStore()
    shadow = decide(
        bfsi_default("t"),
        risk(95, RiskState.HIGH),
        ctx(intent=0.9, labels=["secrecy_demand"]),
        store=store,
        notifier=notifier,
    )
    assert shadow.decision.shadow_mode and shadow.decision.suppressed and shadow.decision.actions
    assert shadow.dispatch == {"dispatched": False, "reason": "shadow mode"} and q.empty()
    assert store.get(shadow.decision.evidence_bundle_id) is not None  # still recorded
    live_res = decide(
        live(bfsi_default("t")),
        risk(95, RiskState.HIGH),
        ctx(intent=0.9, labels=["secrecy_demand"]),
        store=store,
        notifier=notifier,
    )
    msg = q.get_nowait()
    assert live_res.dispatch["websocket"] == 1 and msg["band"] == "HIGH" and msg["agent_prompt"]
    assert live_res.dispatch["siem_webhook"] == "not configured"


def test_notifier_delivers_to_configured_siem() -> None:
    class R:
        status_code = 202

    notifier = Notifier(
        webhooks={"t": WebhookTarget("https://siem.example/hook", "k")}, post=lambda *a, **k: R()
    )
    rep = notifier.dispatch("t", ["siem_webhook", "send_email"], {"x": 1}, shadow=False)
    assert rep["siem_webhook"]["delivered"] and rep["send_email"] == "queued"


# ---------------------------------------------------------------- feedback (T08)
def test_feedback_loop_exports(tmp_path: Path) -> None:
    store = EvidenceStore()
    fb = FeedbackStore(store)
    a = decide(live(bfsi_default("t")), risk(92, RiskState.HIGH), None, store=store)
    b = decide(live(bfsi_default("t")), risk(55), None, store=store)
    fb.record(a.decision.evidence_bundle_id, "true_positive", "analyst-1")
    fb.record(b.decision.evidence_bundle_id, "true_positive", "analyst-1")
    fb.record(
        b.decision.evidence_bundle_id,
        "false_positive",
        "analyst-2",
        "customer confirmed via branch",
    )
    rows = fb.export_eval_rows()
    assert {r["is_spoof_or_fraud"] for r in rows} == {True, False} and len(rows) == 2
    assert fb.export_replay_candidates(tmp_path / "replay.jsonl") == 1
    with pytest.raises(KeyError):
        fb.record("sha256:nope", "unsure", "x")
    with pytest.raises(ValueError):
        fb.record(a.decision.evidence_bundle_id, "maybe", "x")  # type: ignore[arg-type]


# ---------------------------------------------------------------- service API
def test_policy_service_api() -> None:
    c = TestClient(create_app())
    assert c.get("/v1/tenants/t9/profile").json()["shadow_mode"] is True
    assert (
        c.post("/v1/tenants/t9/shadow", json={"shadow_mode": False}).json()["shadow_mode"] is False
    )
    body = {
        "tenant_id": "t9",
        "session_risk": json.loads(risk(88, RiskState.HIGH).model_dump_json()),
        "context_signals": json.loads(ctx(0.8, ["callback_resistance"]).signals.model_dump_json()),
        "transaction_tier": "high",
    }
    r = c.post("/v1/policy/decide", json=body).json()
    assert r["band"] == "HIGH" and "hold_transaction" in r["decision"]["actions"]
    bid = r["decision"]["evidence_bundle_id"]
    got = c.get(f"/v1/evidence/{bid}").json()
    assert (
        got["verified"] is True
        and len(c.get(f"/v1/sessions/{SID}/evidence").json()["bundles"]) == 1
    )
    assert (
        c.post(
            f"/v1/evidence/{bid}/feedback", json={"label": "true_positive", "analyst": "a"}
        ).status_code
        == 201
    )
    prof = asdict(retail_helpline("t9"))
    assert c.put("/v1/tenants/t9/profile", json=prof).json()["name"] == "retail-helpline-v1"
    prof["actions"] = {"medium": {"HIGH": ["nope"]}}
    assert c.put("/v1/tenants/t9/profile", json=prof).status_code == 422
