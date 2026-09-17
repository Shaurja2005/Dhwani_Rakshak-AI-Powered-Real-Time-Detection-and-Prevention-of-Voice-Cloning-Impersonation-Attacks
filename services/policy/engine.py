"""Rules engine: risk + context + tenant profile → PolicyDecision + evidence (B11-T01…T07).

Decision procedure (all data-driven by the tenant profile):
1. ``final_risk`` (B10) combines voice, speaker-mismatch, intent, metadata and
   transaction evidence; the transaction sensitivity tier comes from B10.
2. If the voice session state is ABSTAIN and there is no contextual evidence,
   the band is ABSTAIN — shown as "insufficient audio", never as safe.
3. Otherwise the band comes from the tier-specific thresholds; actions are the
   profile's actions for (tier, band) plus label-triggered minimum actions.
4. A positive watermark (Head F present) always forces at least a call-back.
5. The rationale lists every driver and threshold so the bundle is self-explaining.
6. Shadow mode: actions are computed and recorded, ``suppressed`` is true and
   nothing is dispatched.

Detection is advisory (I1): actions add friction and route to a human; the
system never approves or declines a transaction on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from packages.vg_core.models import (
    CallMetadata,
    FusedWindowScore,
    HeadScore,
    PolicyDecision,
    RiskState,
    SessionRisk,
)
from services.context.engine import ContextReport, final_risk
from services.policy.actions import agent_prompt
from services.policy.evidence import EvidenceStore, build_bundle
from services.policy.notify import Notifier
from services.policy.profiles import TenantProfile


@dataclass
class DecisionResult:
    decision: PolicyDecision
    band: str
    tier: str
    risk: dict[str, float]
    agent_prompt: str
    bundle: dict[str, Any]
    dispatch: dict[str, Any]


def _watermark_present(head_scores: list[HeadScore] | None) -> bool:
    return any(
        h.head_id.value == "F" and h.evidence.get("watermark") == "present"
        for h in head_scores or []
    )


def decide(
    profile: TenantProfile,
    risk: SessionRisk,
    context: ContextReport | None = None,
    speaker_mismatch: float | None = None,
    head_scores: list[HeadScore] | None = None,
    window_scores: list[FusedWindowScore] | None = None,
    call_metadata: CallMetadata | None = None,
    store: EvidenceStore | None = None,
    notifier: Notifier | None = None,
) -> DecisionResult:
    breakdown = final_risk(risk, context, speaker_mismatch)
    tier = context.transaction_tier if context else "low"
    labels = context.signals.intent_labels if context else []
    contextual = (
        breakdown["intent"] > 0
        or breakdown["metadata"] > 0
        or breakdown["transaction"] > 0
        or (speaker_mismatch or 0) > 0
    )

    if risk.state == RiskState.ABSTAIN and not contextual:
        band = "ABSTAIN"
        actions: list[str] = []
    else:
        band = profile.band_for(breakdown["final"], tier)  # type: ignore[arg-type]
        actions = list(profile.actions.get(tier, {}).get(band, []))
    for label in labels:
        actions += profile.label_actions.get(label, [])
    if _watermark_present(head_scores):
        actions += ["agent_banner", "force_callback"]
    actions = list(dict.fromkeys(actions))

    b = profile.bands.get(tier) or profile.bands["medium"]
    lines = [
        f"band={band} tier={tier} final_risk={breakdown['final']:.3f} "
        f"(elevated>={b.elevated:.2f}, high>={b.high:.2f}; profile {profile.name} {profile.version})",
        f"voice: state={risk.state.value} risk_score={risk.risk_score} "
        f"max={risk.p_spoof_session_max:.2f} mean={risk.p_spoof_session_mean:.2f} abstain_ratio={risk.abstain_ratio:.2f}",
    ]
    lines += [f"voice driver: {d.factor} ({d.weight:.2f}) — {d.detail}" for d in risk.drivers]
    lines.append("factors: " + ", ".join(f"{k}={v}" for k, v in breakdown.items() if k != "final"))
    if context:
        for group, items in context.explanations.items():
            if isinstance(items, dict):
                lines += [f"{group}: {v}" for v in items.values()]
            else:
                lines += [f"{group}: {v}" for v in items]
    if _watermark_present(head_scores):
        lines.append("watermark: synthetic-speech watermark detected (forces call-back)")
    if band == "ABSTAIN":
        lines.append(
            "insufficient audio quality to assess the voice; this is not an indication of safety"
        )
    lines.append(
        "advisory only: a human decides; the system never approves or declines a transaction (I1)"
    )
    if profile.shadow_mode:
        lines.append("shadow mode: actions recorded, not dispatched")

    decision = PolicyDecision(
        session_id=risk.session_id,
        decided_at=datetime.now(tz=UTC),
        threshold_profile=profile.name,
        actions=actions,
        rationale="\n".join(lines),
        evidence_bundle_id="pending",
        shadow_mode=profile.shadow_mode,
        suppressed=profile.shadow_mode and bool(actions),
    )
    bundle = build_bundle(
        decision,
        risk,
        profile.snapshot(),
        context.signals if context else None,
        window_scores,
        head_scores,
        call_metadata,
    )
    decision = decision.model_copy(update={"evidence_bundle_id": bundle["bundle_id"]})
    if store is not None:
        store.append(bundle)

    prompt = agent_prompt(band, actions, labels, tier)
    dispatch: dict[str, Any] = {"dispatched": False, "reason": "no actions"}
    if notifier is not None and actions:
        message = {
            "type": "policy_decision",
            "session_id": risk.session_id,
            "band": band,
            "actions": actions,
            "agent_prompt": prompt,
            "evidence_bundle_id": decision.evidence_bundle_id,
            "risk_score": int(round(100 * breakdown["final"])),
        }
        dispatch = notifier.dispatch(
            profile.tenant_id,
            profile.notify_channels
            + [
                a
                for a in actions
                if a in ("send_sms", "send_email", "push_notification", "siem_webhook")
            ],
            message,
            shadow=profile.shadow_mode,
        )
    return DecisionResult(decision, band, tier, breakdown, prompt, bundle, dispatch)
