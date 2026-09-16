"""head_e_liveness — Head E: active liveness challenge (B8).

Implements vg_core.head_api.DetectionHead. Never raises (I2).

* No challenge issued for the session → abstain ``no_challenge``.
* Challenge issued, no answer yet (or expired unanswered) → abstain
  ``no_challenge`` with a note; an unanswered challenge is for the agent to
  follow up, not evidence of synthesis.
* Answer received → emit the verification's ``p_spoof`` on every subsequent
  window of the session, so fusion keeps seeing it.

Rule-based by design (no trained weights); ``calibration_version`` marks the
thresholds as heuristic until tuned in shadow mode.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from packages.vg_core.head_api import BaseDetectionHead
from packages.vg_core.logging import get_logger
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore, SessionContext
from packages.vg_models.heads.head_e_liveness.challenge import ChallengeRegistry
from packages.vg_models.heads.head_e_liveness.verifier import HEURISTIC_VERSION, verify

log = get_logger(__name__)


class HeadE(BaseDetectionHead):
    head_id: ClassVar[str] = "E"
    model_version: ClassVar[str] = "E@challenge-rules-v0.1.0"
    budget_ms: ClassVar[int] = 10

    def __init__(self, registry: ChallengeRegistry) -> None:
        self._registry = registry

    def _score(
        self,
        w: AnalysisWindow,
        t0: float,
        reason: AbstainReason | None,
        p: float | None = None,
        ev: dict[str, Any] | None = None,
    ) -> HeadScore:
        return HeadScore(
            session_id=w.session_id,
            window_id=w.window_id,
            head_id=HeadID.E,
            raw_score=0.0 if p is None else p,
            p_spoof=p,
            abstain=p is None,
            abstain_reason=reason,
            confidence=0.0 if p is None else min(1.0, abs(p - 0.5) * 2),
            latency_ms=int((time.monotonic() - t0) * 1000),
            model_version=self.model_version,
            calibration_version=HEURISTIC_VERSION,
            evidence=ev or {},
        )

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()
        try:
            ch = self._registry.active(window.session_id)
            if ch is None:
                return self._score(window, t0, AbstainReason.NO_CHALLENGE)
            resp = self._registry.response(ch.challenge_id)
            if resp is None:
                note = "challenge expired without an answer" if ch.expired else "awaiting answer"
                return self._score(
                    window,
                    t0,
                    AbstainReason.NO_CHALLENGE,
                    ev={"challenge_id": ch.challenge_id, "note": note},
                )
            v = verify(ch, resp.tokens)
            return self._score(
                window,
                t0,
                None,
                p=v.p_spoof,
                ev={
                    "challenge_id": ch.challenge_id,
                    "kind": ch.kind,
                    "content_match": round(v.content_match, 3),
                    "latency_ms": v.latency_ms,
                    "rate_tokens_per_s": (
                        None if v.rate_tokens_per_s is None else round(v.rate_tokens_per_s, 2)
                    ),
                    "reasons": v.reasons,
                },
            )
        except Exception as exc:  # noqa: BLE001 - heads must never raise (I2)
            log.error("head_e_error", error=str(exc), session_id=window.session_id)
            return self._score(window, t0, AbstainReason.TIMEOUT, ev={"error": type(exc).__name__})
