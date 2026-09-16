"""Watermark asymmetry rule, enforced in code (B8-T06, invariant I4, ADR 0007).

* Watermark **present** → strong positive evidence of synthesis.
* Watermark **absent**  → no evidence whatsoever. Represented as a *neutral*
  HeadScore: ``abstain=True``, ``abstain_reason=None``,
  ``evidence["watermark"] == "absent"``, ``evidence["neutral"] is True``.

Fusion (B9) must pass head scores through ``fusion_inputs`` and should assert
``not is_exonerating_misuse(score)`` in its tests, so that an absent watermark
can never lower (or otherwise change) a fused score.
"""

from __future__ import annotations

from collections.abc import Iterable

from packages.vg_core.models import HeadID, HeadScore

PRESENT_P_SPOOF_FLOOR = 0.95


def is_neutral(score: HeadScore) -> bool:
    return score.head_id == HeadID.F and score.evidence.get("watermark") != "present"


def fusion_inputs(scores: Iterable[HeadScore]) -> list[HeadScore]:
    """Scores fusion may use: drops every Head F score that is not a positive detection."""
    out = []
    for s in scores:
        if s.head_id == HeadID.F:
            if (
                s.abstain
                or s.evidence.get("watermark") != "present"
                or (s.p_spoof or 0) < PRESENT_P_SPOOF_FLOOR
            ):
                continue
        out.append(s)
    return out


def is_exonerating_misuse(score: HeadScore) -> bool:
    """True if a Head F score would lower risk — which must never happen (I4)."""
    return (
        score.head_id == HeadID.F
        and not score.abstain
        and (score.p_spoof or 0) < PRESENT_P_SPOOF_FLOOR
    )
