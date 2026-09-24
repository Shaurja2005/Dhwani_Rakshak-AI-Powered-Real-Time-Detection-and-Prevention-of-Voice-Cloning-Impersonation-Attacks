"""Backpressure and graceful degradation (B14-T06, invariant I10).

One ``LoadController`` per gateway process watches per-window processing
latency (EMA) and in-flight windows. When overloaded it enters DEGRADED mode:

* expensive heads (default: Head A) are **shed** — they return an abstain with
  reason ``timeout`` and ``evidence.load_shed = True`` instead of being queued;
* cheap heads (B, C, D, …) still run, but their confidence is multiplied by
  ``confidence_factor`` and marked ``evidence.degraded = True``, so fusion and
  the UI know the verdict rests on fewer signals.

Hysteresis: enter when EMA > ``enter_ms``, in-flight > ``max_inflight`` or the
batcher reports overload; leave only after ``exit_after`` consecutive windows
under ``exit_ms`` **and** at least ``min_degraded_s`` in degraded mode.
Thresholds are operational knobs, not detection thresholds.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore


@dataclass
class DegradeConfig:
    expensive_heads: tuple[str, ...] = ("A",)
    enter_ms: float = 600.0  # CPU target is p95 < 700 ms per window
    exit_ms: float = 300.0
    exit_after: int = 10
    min_degraded_s: float = 5.0  # windows arrive from all calls, so count alone flaps under load
    ema_alpha: float = 0.3
    max_inflight: int | None = None
    confidence_factor: float = 0.8


class LoadController:
    def __init__(self, cfg: DegradeConfig | None = None) -> None:
        self.cfg = cfg or DegradeConfig()
        self._lock = threading.Lock()
        self.ema_ms = 0.0
        self.inflight = 0
        self.degraded = False
        self._calm = 0
        self.transitions: list[tuple[float, bool]] = []
        self.shed_count = 0
        self._since = 0.0

    # ------------------------------------------------------------------ signals
    @contextmanager
    def track(self) -> Iterator[None]:
        with self._lock:
            self.inflight += 1
            if self.cfg.max_inflight is not None and self.inflight > self.cfg.max_inflight:
                self._set(True)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe((time.perf_counter() - t0) * 1000, _done=True)

    def observe(self, elapsed_ms: float, _done: bool = False) -> None:
        with self._lock:
            if _done:
                self.inflight -= 1
            a = self.cfg.ema_alpha
            self.ema_ms = elapsed_ms if self.ema_ms == 0 else (1 - a) * self.ema_ms + a * elapsed_ms
            if not self.degraded and self.ema_ms > self.cfg.enter_ms:
                self._set(True)
            elif self.degraded:
                self._calm = self._calm + 1 if self.ema_ms < self.cfg.exit_ms else 0
                held = time.monotonic() - self._since >= self.cfg.min_degraded_s
                if self._calm >= self.cfg.exit_after and held:
                    self._set(False)

    def note_overload(self) -> None:
        """Called when the batcher rejects or a deadline expires."""
        with self._lock:
            self._set(True)

    def _set(self, degraded: bool) -> None:
        self._calm = 0
        if degraded:
            self._since = time.monotonic()
        if degraded != self.degraded:
            self.degraded = degraded
            self.transitions.append((time.time(), degraded))

    # ------------------------------------------------------------------ score shaping
    def sheds(self, head_id: str) -> bool:
        return self.degraded and head_id in self.cfg.expensive_heads

    def shed_score(self, window: AnalysisWindow, head_id: str, model_version: str) -> HeadScore:
        self.shed_count += 1
        return HeadScore(
            session_id=window.session_id,
            window_id=window.window_id,
            head_id=HeadID(head_id),
            raw_score=0.0,
            p_spoof=None,
            abstain=True,
            abstain_reason=AbstainReason.TIMEOUT,
            confidence=0.0,
            latency_ms=0,
            model_version=model_version,
            calibration_version="n/a",
            evidence={"load_shed": True, "degraded": True},
        )

    def mark(self, score: HeadScore) -> HeadScore:
        if score.abstain:
            return score.model_copy(update={"evidence": {**score.evidence, "degraded": True}})
        return score.model_copy(
            update={
                "confidence": score.confidence * self.cfg.confidence_factor,
                "evidence": {**score.evidence, "degraded": True},
            }
        )


def load_controller_from_env() -> LoadController | None:
    """Enabled with ``VG_LOAD_SHEDDING=1`` (on by default in deploy configs, B17)."""
    if os.getenv("VG_LOAD_SHEDDING") != "1":
        return None
    return LoadController(
        DegradeConfig(
            enter_ms=float(os.getenv("VG_DEGRADE_ENTER_MS", "600")),
            exit_ms=float(os.getenv("VG_DEGRADE_EXIT_MS", "300")),
        )
    )
