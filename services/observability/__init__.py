"""Observability for the gateway pipeline (B17-T04/T06): Prometheus metrics + drift monitor."""

from __future__ import annotations

import threading
from typing import Any

from prometheus_client import generate_latest

from services.observability.drift import DriftMonitor
from services.observability.metrics import PipelineMetrics


class Observability:
    """The object ``Services.observer`` points at; every hook is cheap and never raises."""

    def __init__(
        self, metrics: PipelineMetrics | None = None, drift: DriftMonitor | None = None
    ) -> None:
        self.metrics = metrics or PipelineMetrics()
        self.drift = drift or DriftMonitor()
        self._stop = threading.Event()

    def on_session(self, delta: int) -> None:
        self.metrics.on_session(delta)

    def on_window(
        self, head_scores: list[Any], fused: Any, elapsed_s: float, degraded: bool
    ) -> None:  # noqa: ANN401
        try:
            self.metrics.on_window(head_scores, fused, elapsed_s, degraded)
        except Exception:  # noqa: BLE001, S110 - observability must never break a call (I10)
            pass

    def on_decision(self, band: str, tier: str, shadow: bool) -> None:
        self.metrics.on_decision(band, tier, shadow)

    def on_close(self, head_scores: list[Any], final_state: str) -> None:
        per_head: dict[str, list[float]] = {}
        for s in head_scores:
            if not s.abstain and s.p_spoof is not None:
                per_head.setdefault(getattr(s.head_id, "value", str(s.head_id)), []).append(
                    s.p_spoof
                )
        self.drift.observe_session(per_head, final_state)

    def refresh(self, queue_depth: int | None = None) -> None:
        """Periodic gauges: drift PSI per head, batcher queue, GPU."""
        for head in self.drift.heads():
            st = self.drift.status(head)
            if st.psi is not None:
                self.metrics.drift_psi.labels(head=head).set(st.psi)
        if queue_depth is not None:
            self.metrics.queue_depth.set(queue_depth)
        self.metrics.sample_gpu()

    def start_sampler(
        self, interval_s: float = 15.0, queue_depth: Any = None
    ) -> None:  # noqa: ANN401
        def loop() -> None:
            while not self._stop.wait(interval_s):
                self.refresh(queue_depth() if callable(queue_depth) else None)

        threading.Thread(target=loop, name="vg-observability", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def exposition(self) -> bytes:
        self.refresh()
        return generate_latest(self.metrics.registry)
