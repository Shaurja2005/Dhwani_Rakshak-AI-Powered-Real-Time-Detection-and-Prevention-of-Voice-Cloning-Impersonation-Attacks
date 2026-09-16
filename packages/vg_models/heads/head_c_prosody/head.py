"""head_c_prosody — Head C: prosody and behavioural analysis (B6).

Implements vg_core.head_api.DetectionHead. Never raises; abstains instead of
guessing (I2). Abstains with ``insufficient_speech`` below 2.0 s of voiced
speech (SOURCE_OF_TRUTH §6) and with ``untrained`` (ADR 0006) when no model is
loaded — still attaching a human-readable prosody summary to ``evidence``.

Language comes from ``ctx.call_metadata.language_hint`` and drives Indic-aware
measurement and normalisation (B6-T06).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, ClassVar

import numpy as np
import torch

from packages.vg_core.head_api import BaseDetectionHead
from packages.vg_core.logging import get_logger
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore, SessionContext
from packages.vg_core.sample_store import get_samples
from packages.vg_models.heads.head_c_prosody.model import (
    ProsodyAnalysis,
    analyse,
    global_vector,
    load,
)

log = get_logger(__name__)

MIN_VOICED_MS = 2000
UNTRAINED_VERSION = "C@prosody-untrained-v0.0.0"


def summarise(a: ProsodyAnalysis) -> list[str]:
    """Plain-English prosody observations (not verdicts)."""
    f = a.features
    out: list[str] = []
    rng = f.get("f0_range_st", math.nan)
    if not math.isnan(rng):
        out.append(
            f"Pitch range {rng:.1f} semitones"
            + (
                " — unusually flat."
                if rng < 3
                else " — within a natural range." if rng < 16 else " — very wide."
            )
        )
    reg = f.get("over_regularity", math.nan)
    if not math.isnan(reg) and reg > 0.7:
        out.append("Pauses and speaking rate are unusually regular (machine-like rhythm).")
    if f.get("breath_count", 0) == 0 and f.get("pause_count", 0) >= 2:
        out.append("No audible breaths before phrases despite several pauses.")
    if f.get("filled_pause_count", 0) > 0:
        out.append("Contains hesitation sounds (e.g. 'uhh'), common in spontaneous human speech.")
    return out[:3]


class HeadC(BaseDetectionHead):
    head_id: ClassVar[str] = "C"
    model_version: ClassVar[str] = UNTRAINED_VERSION
    budget_ms: ClassVar[int] = 25

    def __init__(self, model_path: str | None = None, budget_ms: int | None = None) -> None:
        self._path = model_path or os.getenv("VG_HEAD_C_MODEL")
        self._budget = budget_ms or int(os.getenv("VG_HEAD_C_BUDGET_MS", "25"))
        self._version = UNTRAINED_VERSION
        self._cal_version = "uncalibrated"
        self._loaded: tuple[Any, list[str], Any, dict[str, Any]] | None = None
        self._ready = False

    @property
    def model_version(self) -> str:  # type: ignore[override]  # noqa: F811
        return self._version

    @property
    def budget_ms(self) -> int:  # type: ignore[override]  # noqa: F811
        return self._budget

    def warmup(self) -> None:
        torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
        if self._path:
            self._loaded = load(self._path)
            meta = self._loaded[3]
            self._version = str(meta.get("model_version", "C@prosody-bigru-v0.1.0"))
            self._cal_version = str(meta.get("calibration", {}).get("version", "uncalibrated"))
            log.info("head_c_loaded", path=self._path, lineage=meta.get("lineage"))
        else:
            log.warning("head_c_untrained_mode", note="abstains with prosody summary until trained")
        analyse(np.zeros(48000, dtype=np.float32))
        self._ready = True

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
            head_id=HeadID.C,
            raw_score=raw,
            p_spoof=p,
            abstain=p is None,
            abstain_reason=reason,
            confidence=conf,
            latency_ms=latency,
            model_version=self._version,
            calibration_version=self._cal_version,
            evidence=evidence,
        )

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()
        try:
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
            if pcm is None or len(pcm) < MIN_VOICED_MS * 16:
                return self._score(
                    window,
                    AbstainReason.INSUFFICIENT_SPEECH,
                    t0,
                    ev={"note": "samples unavailable"},
                )
            if not self._ready:
                self.warmup()

            language = ctx.call_metadata.language_hint if ctx.call_metadata else None
            a = analyse(pcm, language)
            evidence: dict[str, Any] = {
                "reasons": summarise(a),
                "language": language,
                "prosody": {
                    k: (None if math.isnan(v) else round(v, 4)) for k, v in a.features.items()
                },
            }
            if self._loaded is None:
                return self._score(
                    window, AbstainReason.UNTRAINED, t0, ev={**evidence, "untrained": True}
                )

            model, names, norm, meta = self._loaded
            g = global_vector(a.features, names, norm, language)
            _, norm_source = norm.apply(a.features, language)
            with torch.inference_mode():
                logit = float(model(torch.from_numpy(a.frames)[None], torch.from_numpy(g)[None])[0])
            cal = meta.get("calibration", {"scale": 1.0, "bias": 0.0})
            z = max(min(float(cal["scale"]) * logit + float(cal["bias"]), 30.0), -30.0)
            p_spoof = 1.0 / (1.0 + math.exp(-z))
            evidence["normalised_with"] = norm_source
            return self._score(
                window,
                None,
                t0,
                p=p_spoof,
                raw=logit,
                conf=min(1.0, abs(p_spoof - 0.5) * 2),
                ev=evidence,
            )
        except Exception as exc:  # noqa: BLE001 - heads must never raise (I2)
            log.error("head_c_error", error=str(exc), session_id=window.session_id)
            return self._score(window, AbstainReason.TIMEOUT, t0, ev={"error": type(exc).__name__})
