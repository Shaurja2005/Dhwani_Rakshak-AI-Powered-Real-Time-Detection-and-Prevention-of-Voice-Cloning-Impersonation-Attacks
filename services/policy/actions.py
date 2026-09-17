"""Action library and pre-transaction warning prompts (B11-T02, B11-T06).

Every action has an audience and, for agent-facing actions, wording written
for a *non-technical frontline agent*. Copy is advisory, never authoritative
(invariant I1): it says what to verify, never "this caller is a fraudster".
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.vg_core.models import VALID_ACTIONS


@dataclass(frozen=True)
class ActionSpec:
    name: str
    audience: str  # agent | customer | supervisor | soc | system
    agent_text: str | None
    blocking: bool  # adds friction to the transaction (still advisory: a human decides)


LIBRARY: dict[str, ActionSpec] = {
    "agent_banner": ActionSpec(
        "agent_banner",
        "agent",
        "Take extra care: verify the caller before acting on any request.",
        False,
    ),
    "force_callback": ActionSpec(
        "force_callback",
        "agent",
        "End the call politely and call back on the number registered to the account. "
        "Do not use a number the caller gives you.",
        True,
    ),
    "hold_transaction": ActionSpec(
        "hold_transaction",
        "system",
        "Do not process this transaction until the caller has been verified another way.",
        True,
    ),
    "supervisor_escalate": ActionSpec(
        "supervisor_escalate", "supervisor", "Bring in your supervisor before continuing.", False
    ),
    "soc_ticket": ActionSpec("soc_ticket", "soc", None, False),
    "flag_recording": ActionSpec("flag_recording", "system", None, False),
    "send_sms": ActionSpec("send_sms", "customer", None, False),
    "send_email": ActionSpec("send_email", "customer", None, False),
    "push_notification": ActionSpec("push_notification", "customer", None, False),
    "siem_webhook": ActionSpec("siem_webhook", "soc", None, False),
    "step_up_mfa": ActionSpec(
        "step_up_mfa",
        "agent",
        "Ask the caller to approve a verification request in their banking app before continuing.",
        True,
    ),
    "dual_approval": ActionSpec(
        "dual_approval",
        "agent",
        "This request needs a second approver before it can go ahead.",
        True,
    ),
}
assert set(LIBRARY) == VALID_ACTIONS, "action library must cover the contract's action list"

LABEL_HINTS = {
    "credential_request": "The caller asked for a one-time password or PIN. Staff never need these.",
    "secrecy_demand": "The caller asked for secrecy — a common pressure tactic.",
    "manufactured_urgency": "The caller is pushing for speed.",
    "authority_pressure": "The caller is invoking seniority or an authority.",
    "callback_resistance": "The caller is avoiding a call back.",
    "unusual_beneficiary": "Money is going somewhere new.",
    "unusual_channel": "The caller wants to move to another app or channel.",
    "payment_request": "The caller is asking for a payment.",
}

ABSTAIN_TEXT = (
    "Audio quality is too poor to check this voice. This is NOT a sign the caller is genuine — "
    "follow your standard verification steps."
)


def agent_prompt(band: str, actions: list[str], intent_labels: list[str], tier: str) -> str:
    """1–3 plain sentences for the agent's screen (B11-T06)."""
    if band == "ABSTAIN":
        return ABSTAIN_TEXT
    if band == "LOW" and not actions:
        return ""
    lead = {
        "HIGH": "Strong signs this call may not be who it claims to be.",
        "ELEVATED": "Some signs this call may not be who it claims to be.",
        "LOW": "A risk cue was noticed on this call.",
    }[band]
    # Most important concrete step first: a call-back beats a generic banner.
    order = [
        "force_callback",
        "hold_transaction",
        "dual_approval",
        "step_up_mfa",
        "supervisor_escalate",
        "agent_banner",
    ]
    ranked = sorted(
        (a for a in actions if LIBRARY[a].agent_text),
        key=lambda a: order.index(a) if a in order else 99,
    )
    steps = [LIBRARY[a].agent_text for a in ranked]
    hint = next(
        (
            LABEL_HINTS[x]
            for x in (
                "credential_request",
                "secrecy_demand",
                "callback_resistance",
                "unusual_beneficiary",
                "authority_pressure",
                "manufactured_urgency",
            )
            if x in intent_labels
        ),
        None,
    )
    parts = [lead]
    if hint:
        parts.append(hint)
    if steps:
        parts.append(steps[0])
    return " ".join(parts[:3])
