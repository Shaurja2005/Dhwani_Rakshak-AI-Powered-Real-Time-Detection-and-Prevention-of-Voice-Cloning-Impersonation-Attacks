"""policy service entrypoint (B11).

REST (mounted behind the API gateway, B12):
    POST /v1/policy/decide                      decide for a session (risk + optional context)
    GET  /v1/evidence/{bundle_id}               evidence bundle (verifiable)
    GET  /v1/sessions/{session_id}/evidence     all bundles for a session
    POST /v1/evidence/{bundle_id}/feedback      analyst label (B13-T06)
    GET  /v1/tenants/{tenant_id}/profile        active profile
    PUT  /v1/tenants/{tenant_id}/profile        replace profile (admin, B13-T05)
    POST /v1/tenants/{tenant_id}/shadow         {"shadow_mode": bool}
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from packages.vg_core.models import ContextSignals, HeadScore, SessionRisk
from services.context.engine import ContextReport
from services.policy.engine import decide
from services.policy.evidence import EvidenceStore, verify_bundle
from services.policy.feedback import FeedbackStore
from services.policy.notify import Notifier
from services.policy.profiles import ProfileStore, TenantProfile


class DecideRequest(BaseModel):
    tenant_id: str
    session_risk: SessionRisk
    context_signals: ContextSignals | None = None
    context_explanations: dict[str, Any] = {}
    transaction_tier: Literal["low", "medium", "high"] = "low"
    speaker_mismatch: float | None = None
    head_scores: list[HeadScore] = []


class FeedbackIn(BaseModel):
    label: Literal["true_positive", "false_positive", "unsure"]
    analyst: str
    note: str = ""


class ShadowIn(BaseModel):
    shadow_mode: bool


def create_app(
    profiles: ProfileStore | None = None,
    evidence: EvidenceStore | None = None,
    notifier: Notifier | None = None,
) -> FastAPI:
    profiles = profiles or ProfileStore()
    evidence = evidence or EvidenceStore()
    notifier = notifier or Notifier()
    feedback = FeedbackStore(evidence)
    app = FastAPI(title="VoiceGuard Policy", version="0.1.0")
    app.state.profiles, app.state.evidence, app.state.notifier, app.state.feedback = (
        profiles,
        evidence,
        notifier,
        feedback,
    )

    @app.post("/v1/policy/decide")
    def post_decide(req: DecideRequest) -> dict[str, Any]:
        ctx = None
        if req.context_signals is not None:
            ctx = ContextReport(req.context_signals, req.context_explanations, req.transaction_tier)
        res = decide(
            profiles.get(req.tenant_id),
            req.session_risk,
            ctx,
            req.speaker_mismatch,
            req.head_scores or None,
            store=evidence,
            notifier=notifier,
        )
        return {
            "decision": res.decision.model_dump(mode="json"),
            "band": res.band,
            "tier": res.tier,
            "risk": res.risk,
            "agent_prompt": res.agent_prompt,
            "dispatch": res.dispatch,
        }

    @app.get("/v1/evidence/{bundle_id}")
    def get_bundle(bundle_id: str) -> dict[str, Any]:
        b = evidence.get(bundle_id)
        if b is None:
            raise HTTPException(404, "no such bundle")
        return {"bundle": b, "verified": verify_bundle(b)}

    @app.get("/v1/sessions/{session_id}/evidence")
    def session_bundles(session_id: str) -> dict[str, Any]:
        return {"bundles": evidence.for_session(session_id)}

    @app.post("/v1/evidence/{bundle_id}/feedback", status_code=201)
    def post_feedback(bundle_id: str, body: FeedbackIn) -> dict[str, Any]:
        try:
            return feedback.record(bundle_id, body.label, body.analyst, body.note)
        except KeyError as exc:
            raise HTTPException(404, "no such bundle") from exc

    @app.get("/v1/tenants/{tenant_id}/profile")
    def get_profile(tenant_id: str) -> dict[str, Any]:
        return asdict(profiles.get(tenant_id))

    @app.put("/v1/tenants/{tenant_id}/profile")
    def put_profile(tenant_id: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            prof = TenantProfile(**{**body, "tenant_id": tenant_id})
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        profiles.put(prof)
        return asdict(prof)

    @app.post("/v1/tenants/{tenant_id}/shadow")
    def shadow(tenant_id: str, body: ShadowIn) -> dict[str, Any]:
        return asdict(profiles.set_shadow(tenant_id, body.shadow_mode))

    return app


def main() -> None:
    import uvicorn

    data = Path(os.getenv("VG_POLICY_DATA", "data/policy"))
    data.mkdir(parents=True, exist_ok=True)
    app = create_app(ProfileStore(data / "profiles"), EvidenceStore(data / "evidence.db"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VG_POLICY_PORT", "8040")))  # noqa: S104


if __name__ == "__main__":
    main()
