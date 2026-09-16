"""head_a_ssl — Head A real-time inference wrapper (B4-T09).

Implements vg_core.head_api.DetectionHead. Never raises; abstains instead of
guessing (I2) and on budget overrun (I10).

Modes
-----
* checkpoint: ``VG_HEAD_A_CHECKPOINT`` (or ``checkpoint=``) points to a trained
  checkpoint from ml/training/train_head_a.py.
* untrained: no checkpoint -> random-init tiny model. Inference still runs (so
  latency and plumbing are exercised) but the head ABSTAINS with
  ``evidence.untrained = True`` and the raw score recorded — a random model has
  no basis to score (I2). ``emit_untrained_scores=True`` (or
  ``VG_HEAD_A_EMIT_UNTRAINED=1``) emits them anyway, for debugging only.
"""

from __future__ import annotations

import concurrent.futures as cf
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
from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel, load_checkpoint

log = get_logger(__name__)

MIN_VOICED_MS = 1500
UNCALIBRATED = "uncalibrated"


class HeadA(BaseDetectionHead):
    head_id: ClassVar[str] = "A"
    model_version: ClassVar[str] = "A@tiny-nes2net-v0.0.0"
    budget_ms: ClassVar[int] = int(os.getenv("VG_HEAD_A_BUDGET_MS", "400"))  # CPU budget (SoT §6)

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        budget_ms: int | None = None,
        # Temperature-scaling calibration (B9 will replace with fitted params).
        cal_scale: float = 8.0,
        cal_bias: float = 0.0,
        calibration_version: str = UNCALIBRATED,
        emit_untrained_scores: bool | None = None,
    ) -> None:
        self._emit_untrained = (
            emit_untrained_scores
            if emit_untrained_scores is not None
            else os.getenv("VG_HEAD_A_EMIT_UNTRAINED") == "1"
        )
        self._checkpoint = checkpoint or os.getenv("VG_HEAD_A_CHECKPOINT")
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._budget_ms = budget_ms or type(self).budget_ms
        self._cal = (cal_scale, cal_bias)
        self._cal_version = calibration_version
        self._model: HeadAModel | None = None
        self._untrained = True
        self._version = type(self).model_version
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="head-a")

    @property
    def model_version(self) -> str:  # type: ignore[override]  # noqa: F811
        return self._version

    @property
    def budget_ms(self) -> int:  # type: ignore[override]  # noqa: F811
        return self._budget_ms

    # ------------------------------------------------------------------ lifecycle
    def warmup(self) -> None:
        if self._checkpoint:
            model, meta = load_checkpoint(self._checkpoint)
            self._untrained = False
            if "calibration" in meta:
                cal = meta["calibration"]
                self._cal = (float(cal["scale"]), float(cal["bias"]))
                self._cal_version = str(cal["version"])
            log.info("head_a_loaded", checkpoint=self._checkpoint, lineage=meta.get("lineage"))
        else:
            torch.manual_seed(0)
            model = HeadAModel(HeadAConfig(frontend="tiny", version="0.0.0"))
            log.warning(
                "head_a_untrained_mode", note="scores are meaningless until a checkpoint is set"
            )
        self._model = model.to(self._device).eval()
        self._version = model.model_version
        with torch.inference_mode():  # first call compiles kernels / allocates
            self._model(torch.zeros(1, 16000, device=self._device))

    # ------------------------------------------------------------------ scoring
    def _abstain(
        self,
        window: AnalysisWindow,
        reason: AbstainReason | None,
        t0: float,
        **evidence: Any,  # noqa: ANN401
    ) -> HeadScore:
        return HeadScore(
            session_id=window.session_id,
            window_id=window.window_id,
            head_id=HeadID.A,
            raw_score=0.0,
            p_spoof=None,
            abstain=True,
            abstain_reason=reason,
            confidence=0.0,
            latency_ms=int((time.monotonic() - t0) * 1000),
            model_version=self._version,
            calibration_version=self._cal_version,
            evidence=evidence,
        )

    def _infer(self, pcm: np.ndarray) -> float:
        assert self._model is not None
        with torch.inference_mode():
            wav = torch.tensor(pcm, dtype=torch.float32).unsqueeze(
                0
            )  # copy: store arrays are read-only
            _, score = self._model(wav.to(self._device))
        return float(score.item())

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()
        try:
            if not window.quality_ok:
                return self._abstain(
                    window, AbstainReason.QUALITY_GATE, t0, flags=window.quality_flags
                )
            if window.voiced_ms < MIN_VOICED_MS:
                return self._abstain(
                    window, AbstainReason.INSUFFICIENT_SPEECH, t0, voiced_ms=window.voiced_ms
                )
            if self._model is None:
                self.warmup()
            pcm = get_samples(window.samples_ref)
            if pcm is None or len(pcm) < MIN_VOICED_MS * 16:
                return self._abstain(
                    window, AbstainReason.INSUFFICIENT_SPEECH, t0, note="samples unavailable"
                )

            remaining = self._budget_ms / 1000 - (time.monotonic() - t0)
            fut = self._pool.submit(self._infer, pcm)
            try:
                raw = fut.result(timeout=max(remaining, 0.0))
            except cf.TimeoutError:
                fut.cancel()
                return self._abstain(window, AbstainReason.TIMEOUT, t0, budget_ms=self._budget_ms)

            if self._untrained and not self._emit_untrained:
                # No AbstainReason fits yet; adding one is a contract change (ADR needed).
                return self._abstain(window, None, t0, untrained=True, raw_score=round(raw, 4))

            scale, bias = self._cal
            p_spoof = 1.0 / (1.0 + math.exp(scale * raw + bias))  # higher raw = more bona fide
            confidence = 0.0 if self._untrained else min(1.0, abs(p_spoof - 0.5) * 2)
            return HeadScore(
                session_id=window.session_id,
                window_id=window.window_id,
                head_id=HeadID.A,
                raw_score=raw,
                p_spoof=p_spoof,
                abstain=False,
                confidence=confidence,
                latency_ms=int((time.monotonic() - t0) * 1000),
                model_version=self._version,
                calibration_version=self._cal_version,
                evidence={
                    "untrained": self._untrained,
                    "cosine_to_bona_fide_centre": round(raw, 4),
                    "layer_weights": (
                        [round(w, 3) for w in self._model.layer_sum.weights()]
                        if self._model is not None
                        else []
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - heads must never raise (I2)
            log.error("head_a_error", error=str(exc), session_id=window.session_id)
            return self._abstain(window, AbstainReason.TIMEOUT, t0, error=type(exc).__name__)
