"""head_b_dsp — Head B: DSP artifact + acoustic scene consistency (B5).

Implements vg_core.head_api.DetectionHead. Never raises; abstains instead of
guessing (I2).

* With a trained GBDT (``VG_HEAD_B_MODEL`` or ``model_path=``): emits a
  calibrated ``p_spoof`` plus explanations.
* Without one: abstains with ``untrained`` (ADR 0006) but still attaches the
  interpretable detector outputs and explanations to ``evidence`` — the scene
  and codec analysis are useful to an analyst even before training.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, ClassVar

import numpy as np

from packages.vg_core.head_api import BaseDetectionHead
from packages.vg_core.logging import get_logger
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore, SessionContext
from packages.vg_core.sample_store import get_samples
from packages.vg_models.heads.head_b_dsp.explain import reasons
from packages.vg_models.heads.head_b_dsp.gbdt import GBDT
from packages.vg_models.heads.head_b_dsp.pipeline import analyse

log = get_logger(__name__)

MIN_VOICED_MS = 1500
UNTRAINED_VERSION = "B@dsp-untrained-v0.0.0"


class HeadB(BaseDetectionHead):
    head_id: ClassVar[str] = "B"
    model_version: ClassVar[str] = UNTRAINED_VERSION
    budget_ms: ClassVar[int] = 15

    def __init__(self, model_path: str | None = None, budget_ms: int | None = None) -> None:
        self._path = model_path or os.getenv("VG_HEAD_B_MODEL")
        self._budget = budget_ms or int(os.getenv("VG_HEAD_B_BUDGET_MS", "15"))
        self._model: GBDT | None = None
        self._meta: dict[str, Any] = {}
        self._version = UNTRAINED_VERSION
        self._cal_version = "uncalibrated"
        self._ready = False

    @property
    def model_version(self) -> str:  # type: ignore[override]  # noqa: F811
        return self._version

    @property
    def budget_ms(self) -> int:  # type: ignore[override]  # noqa: F811
        return self._budget

    def warmup(self) -> None:
        if self._path:
            self._model, self._meta = GBDT.load(self._path)
            self._version = str(self._meta.get("model_version", "B@dsp-gbdt-v0.1.0"))
            self._cal_version = str(
                self._meta.get("calibration", {}).get("version", "uncalibrated")
            )
            log.info("head_b_loaded", path=self._path, lineage=self._meta.get("lineage"))
        else:
            log.warning(
                "head_b_untrained_mode", note="abstains with evidence until a model is trained"
            )
        analyse(np.zeros(48000, dtype=np.float32), 16000)  # warm numpy paths
        self._ready = True

    def _score(
        self,
        window: AnalysisWindow,
        reason: AbstainReason | None,
        t0: float,
        p_spoof: float | None = None,
        raw: float = 0.0,
        confidence: float = 0.0,
        evidence: dict[str, Any] | None = None,
    ) -> HeadScore:
        latency = int((time.monotonic() - t0) * 1000)
        ev = dict(evidence or {})
        if latency > self._budget:
            ev["over_budget_ms"] = latency - self._budget
        return HeadScore(
            session_id=window.session_id,
            window_id=window.window_id,
            head_id=HeadID.B,
            raw_score=raw,
            p_spoof=p_spoof,
            abstain=p_spoof is None,
            abstain_reason=reason,
            confidence=confidence,
            latency_ms=latency,
            model_version=self._version,
            calibration_version=self._cal_version,
            evidence=ev,
        )

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()
        try:
            if not window.quality_ok:
                return self._score(
                    window, AbstainReason.QUALITY_GATE, t0, evidence={"flags": window.quality_flags}
                )
            if window.voiced_ms < MIN_VOICED_MS:
                return self._score(
                    window,
                    AbstainReason.INSUFFICIENT_SPEECH,
                    t0,
                    evidence={"voiced_ms": window.voiced_ms},
                )
            pcm = get_samples(window.samples_ref)
            if pcm is None or len(pcm) < MIN_VOICED_MS * 16:
                return self._score(
                    window,
                    AbstainReason.INSUFFICIENT_SPEECH,
                    t0,
                    evidence={"note": "samples unavailable"},
                )
            if not self._ready:
                self.warmup()

            a = analyse(pcm, window.original_sample_rate)
            reference = {k: tuple(v) for k, v in self._meta.get("bona_fide_reference", {}).items()}
            evidence: dict[str, Any] = {
                "reasons": reasons(a.scene, a.vocoder, a.codec, a.features, reference or None),  # type: ignore[arg-type]
                "codec_estimate": a.codec.label,
                "bandwidth_hz": round(a.codec.bandwidth_hz),
                "scene": {
                    "rt60_voice_s": a.scene.rt60_voice_s,
                    "rt60_background_s": a.scene.rt60_background_s,
                    "inconsistency": round(a.scene.inconsistency, 3),
                },
                "vocoder": {
                    "upsampling_tones": round(a.vocoder.upsampling_tones, 3),
                    "band_edge_sharpness": round(a.vocoder.band_edge_sharpness, 3),
                },
            }
            if self._model is None:
                return self._score(
                    window, AbstainReason.UNTRAINED, t0, evidence={**evidence, "untrained": True}
                )

            x = np.array([[a.features.get(n, float("nan")) for n in self._model.feature_names]])
            raw = float(self._model.decision(x)[0])  # log-odds spoof
            cal = self._meta.get("calibration", {"scale": 1.0, "bias": 0.0})
            z = float(cal["scale"]) * raw + float(cal["bias"])
            p_spoof = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            return self._score(
                window,
                None,
                t0,
                p_spoof=p_spoof,
                raw=raw,
                confidence=min(1.0, abs(p_spoof - 0.5) * 2),
                evidence=evidence,
            )
        except Exception as exc:  # noqa: BLE001 - heads must never raise (I2)
            log.error("head_b_error", error=str(exc), session_id=window.session_id)
            return self._score(
                window, AbstainReason.TIMEOUT, t0, evidence={"error": type(exc).__name__}
            )
