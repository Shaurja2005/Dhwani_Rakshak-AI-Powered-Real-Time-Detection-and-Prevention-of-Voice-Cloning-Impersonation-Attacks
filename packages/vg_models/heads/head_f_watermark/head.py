"""head_f_watermark — Head F: watermark probe (B8).

Implements vg_core.head_api.DetectionHead. Never raises (I2).

* Any detector finds a watermark → ``p_spoof >= 0.95``, high confidence.
* No detector finds one → neutral abstain (ADR 0007, I4): absence is not
  evidence of authenticity and must change the fused score by exactly zero.
* No detector available at all → abstain ``untrained`` (ADR 0006): we did not
  look, which is different from "looked and found nothing".
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
from packages.vg_models.heads.head_f_watermark.asymmetry import PRESENT_P_SPOOF_FLOOR
from packages.vg_models.heads.head_f_watermark.detectors import (
    AudioSealDetector,
    SpreadSpectrumDetector,
    WatermarkDetector,
)

log = get_logger(__name__)


def default_detectors() -> list[WatermarkDetector]:
    dets: list[WatermarkDetector] = []
    try:
        dets.append(AudioSealDetector())
    except Exception as exc:  # noqa: BLE001 - optional dependency / checkpoint
        log.warning("head_f_audioseal_unavailable", error=str(exc))
    keys = os.getenv("VG_WATERMARK_KEYS")  # "id1:secret1,id2:secret2"
    if keys:
        pairs = dict(kv.split(":", 1) for kv in keys.split(",") if ":" in kv)
        dets.append(SpreadSpectrumDetector(pairs))
    return dets


class HeadF(BaseDetectionHead):
    head_id: ClassVar[str] = "F"
    model_version: ClassVar[str] = "F@watermark-probe-v0.1.0"
    budget_ms: ClassVar[int] = 5

    def __init__(self, detectors: list[WatermarkDetector] | None = None) -> None:
        self._detectors = detectors
        self._budget = int(os.getenv("VG_HEAD_F_BUDGET_MS", "5"))

    @property
    def budget_ms(self) -> int:  # type: ignore[override]  # noqa: F811
        return self._budget

    def warmup(self) -> None:
        if self._detectors is None:
            self._detectors = default_detectors()
        for d in self._detectors:
            d.detect(np.zeros(16000, dtype=np.float32))

    def _score(
        self,
        w: AnalysisWindow,
        t0: float,
        reason: AbstainReason | None,
        p: float | None,
        ev: dict[str, Any],
    ) -> HeadScore:
        latency = int((time.monotonic() - t0) * 1000)
        if latency > self._budget:
            ev["over_budget_ms"] = latency - self._budget
        return HeadScore(
            session_id=w.session_id,
            window_id=w.window_id,
            head_id=HeadID.F,
            raw_score=ev.get("score", 0.0),
            p_spoof=p,
            abstain=p is None,
            abstain_reason=reason,
            confidence=0.0 if p is None else 0.95,
            latency_ms=latency,
            model_version=self.model_version,
            calibration_version="detector-native",
            evidence=ev,
        )

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()
        try:
            if self._detectors is None:
                self.warmup()
            assert self._detectors is not None
            if not self._detectors:
                return self._score(
                    window,
                    t0,
                    AbstainReason.UNTRAINED,
                    None,
                    {
                        "note": "no watermark detector installed",
                        "watermark": "not_checked",
                        "neutral": True,
                    },
                )
            pcm = get_samples(window.samples_ref)
            if pcm is None:
                return self._score(
                    window,
                    t0,
                    AbstainReason.INSUFFICIENT_SPEECH,
                    None,
                    {"note": "samples unavailable", "watermark": "not_checked", "neutral": True},
                )
            results = [d.detect(pcm, window.sample_rate) for d in self._detectors]
            hits = [r for r in results if r.present]
            checked = [r.detector for r in results]
            if hits:
                best = max(hits, key=lambda r: r.probability)
                p = max(PRESENT_P_SPOOF_FLOOR, min(0.999, best.probability))
                return self._score(
                    window,
                    t0,
                    None,
                    p,
                    {
                        "watermark": "present",
                        "detector": best.detector,
                        "score": round(best.score, 3),
                        "detail": best.detail,
                        "checked": checked,
                        "reasons": [
                            f"A synthetic-speech watermark was detected ({best.detector})."
                        ],
                    },
                )
            # I4: absence is neutral — abstain with no reason so fusion ignores it.
            return self._score(
                window,
                t0,
                None,
                None,
                {
                    "watermark": "absent",
                    "neutral": True,
                    "checked": checked,
                    "note": "no watermark found; this is not evidence the voice is genuine",
                },
            )
        except Exception as exc:  # noqa: BLE001 - heads must never raise (I2)
            log.error("head_f_error", error=str(exc), session_id=window.session_id)
            return self._score(
                window,
                t0,
                AbstainReason.TIMEOUT,
                None,
                {"error": type(exc).__name__, "watermark": "not_checked", "neutral": True},
            )
