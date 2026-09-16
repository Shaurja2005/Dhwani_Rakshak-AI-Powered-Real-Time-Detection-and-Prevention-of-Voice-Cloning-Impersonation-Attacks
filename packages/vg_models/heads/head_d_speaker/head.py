"""head_d_speaker — Head D: speaker verification against an enrolled voiceprint (B7).

Implements vg_core.head_api.DetectionHead. Never raises; abstains instead of
guessing (I2).

Abstains with ``no_enrollment`` (B7-T06) when the call has no
``claimed_identity_id``, when that identity has no voiceprint for this tenant,
or when the voiceprint has expired — most inbound callers are not enrolled,
and a fabricated score would be worse than none.

``p_spoof`` for this head means *probability the speaker does not match the
claimed identity* (the fusion contract has one probability per head); the raw
cosine and AS-norm score are in ``evidence``.

If only the classical baseline embedder is available, the head abstains with
``untrained`` unless ``allow_baseline=True`` (tests / demos only).
"""

from __future__ import annotations

import os
import time
from typing import Any, ClassVar

import numpy as np

from packages.vg_core.head_api import BaseDetectionHead
from packages.vg_core.logging import get_logger
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore, SessionContext
from packages.vg_core.sample_store import get_samples
from packages.vg_models.heads.head_d_speaker.embedders import Embedder, build_embedder
from packages.vg_models.heads.head_d_speaker.scoring import Calibration, as_norm
from packages.vg_models.heads.head_d_speaker.vault import VoiceprintVault

log = get_logger(__name__)

MIN_VOICED_MS = 1500


def channel_condition(original_sample_rate: int | None, detected_codec: str | None = None) -> str:
    """Map window metadata to the enrolment condition used for channel compensation (B7-T05)."""
    if original_sample_rate is not None and original_sample_rate <= 8000:
        return "narrowband"
    return "wideband"


class HeadD(BaseDetectionHead):
    head_id: ClassVar[str] = "D"
    model_version: ClassVar[str] = "D@speaker-unset-v0.0.0"
    budget_ms: ClassVar[int] = 30

    def __init__(
        self,
        vault: VoiceprintVault,
        embedder: Embedder | None = None,
        calibration: Calibration | None = None,
        allow_baseline: bool = False,
        budget_ms: int | None = None,
    ) -> None:
        self._vault = vault
        self._embedder = embedder
        self._cal = calibration or Calibration()
        self._allow_baseline = allow_baseline
        self._budget = budget_ms or int(os.getenv("VG_HEAD_D_BUDGET_MS", "30"))
        self._version = "D@speaker-unset-v0.0.0"

    @property
    def model_version(self) -> str:  # type: ignore[override]  # noqa: F811
        return self._version

    @property
    def budget_ms(self) -> int:  # type: ignore[override]  # noqa: F811
        return self._budget

    def warmup(self) -> None:
        if self._embedder is None:
            name = os.getenv("VG_HEAD_D_EMBEDDER", "ecapa")
            try:
                self._embedder = build_embedder(name)
            except Exception as exc:  # noqa: BLE001 - missing optional dependency / weights
                log.warning(
                    "head_d_embedder_unavailable",
                    embedder=name,
                    error=str(exc),
                    fallback="mfcc_stats",
                )
                self._embedder = build_embedder("mfcc_stats")
        self._embedder.embed(np.zeros(16000, dtype=np.float32))
        self._version = f"D@{self._embedder.name.replace('_', '')}-cosine-v0.1.0"

    def _score(
        self,
        w: AnalysisWindow,
        reason: AbstainReason | None,
        t0: float,
        p: float | None = None,
        raw: float = 0.0,
        conf: float = 0.0,
        ev: dict[str, Any] | None = None,
    ) -> HeadScore:
        latency = int((time.monotonic() - t0) * 1000)
        evidence = dict(ev or {})
        if latency > self._budget:
            evidence["over_budget_ms"] = latency - self._budget
        return HeadScore(
            session_id=w.session_id,
            window_id=w.window_id,
            head_id=HeadID.D,
            raw_score=raw,
            p_spoof=p,
            abstain=p is None,
            abstain_reason=reason,
            confidence=conf,
            latency_ms=latency,
            model_version=self._version,
            calibration_version=self._cal.version,
            evidence=evidence,
        )

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()
        try:
            claimed = ctx.claimed_identity_id or (
                ctx.call_metadata.claimed_identity_id if ctx.call_metadata else None
            )
            if not claimed:
                return self._score(
                    window, AbstainReason.NO_ENROLLMENT, t0, ev={"note": "no claimed identity"}
                )
            vp = self._vault.get(ctx.tenant_id, claimed)
            if vp is None or vp.expired():
                note = "voiceprint expired" if vp is not None else "identity not enrolled"
                return self._score(window, AbstainReason.NO_ENROLLMENT, t0, ev={"note": note})
            if not window.quality_ok:
                return self._score(
                    window, AbstainReason.QUALITY_GATE, t0, ev={"flags": window.quality_flags}
                )
            if window.voiced_ms < MIN_VOICED_MS:
                return self._score(
                    window,
                    AbstainReason.INSUFFICIENT_SPEECH,
                    t0,
                    ev={"voiced_ms": window.voiced_ms},
                )
            pcm = get_samples(window.samples_ref)
            if pcm is None:
                return self._score(
                    window,
                    AbstainReason.INSUFFICIENT_SPEECH,
                    t0,
                    ev={"note": "samples unavailable"},
                )
            if self._embedder is None:
                self.warmup()
            assert self._embedder is not None
            if vp.embedder != self._embedder.name:
                return self._score(
                    window,
                    AbstainReason.NO_ENROLLMENT,
                    t0,
                    ev={
                        "note": f"voiceprint made with {vp.embedder}, "
                        f"head uses {self._embedder.name}"
                    },
                )
            if self._embedder.is_baseline and not self._allow_baseline:
                return self._score(
                    window,
                    AbstainReason.UNTRAINED,
                    t0,
                    ev={
                        "note": "only the classical baseline embedder is available",
                        "untrained": True,
                    },
                )

            condition = channel_condition(window.original_sample_rate, window.detected_codec)
            enroll = vp.centroid(condition)
            if enroll is None:
                return self._score(
                    window, AbstainReason.NO_ENROLLMENT, t0, ev={"note": "empty voiceprint"}
                )
            test = self._embedder.embed(pcm)
            cohort = self._vault.get_cohort(self._embedder.name)
            raw_cos, snorm = as_norm(enroll, test, cohort)
            score = snorm if (self._cal.uses_asnorm and cohort is not None) else raw_cos
            p_mismatch = self._cal.p_mismatch(score)
            return self._score(
                window,
                None,
                t0,
                p=p_mismatch,
                raw=score,
                conf=min(1.0, abs(p_mismatch - 0.5) * 2),
                ev={
                    "claimed_identity_id": claimed,
                    "cosine": round(raw_cos, 4),
                    "as_norm": round(snorm, 4) if cohort is not None else None,
                    "enrolment_condition": condition if condition in vp.embeddings else "all",
                    "enrolment_sessions": len(vp.sessions),
                    "embedder": self._embedder.name,
                    "baseline_embedder": self._embedder.is_baseline,
                    "meaning": "p_spoof = probability the voice does not match the claimed "
                    "identity",
                },
            )
        except Exception as exc:  # noqa: BLE001 - heads must never raise (I2)
            log.error("head_d_error", error=str(exc), session_id=window.session_id)
            return self._score(window, AbstainReason.TIMEOUT, t0, ev={"error": type(exc).__name__})
