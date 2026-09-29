"""Prometheus metrics for the detection path (B17-T04).

``PipelineMetrics`` is the observer the gateway pipeline calls once per window
and once per policy decision. Everything is a label-bounded counter, gauge or
histogram — never a per-session label (cardinality) and never audio (I5).

Exposed families (prefix ``vg_``):

    vg_windows_total{state}                       windows scored, by fused state
    vg_window_latency_seconds                     full per-window path latency
    vg_head_latency_seconds{head}                 per-head scoring latency
    vg_head_abstain_total{head,reason}            abstentions (the first thing on-call checks)
    vg_head_scores_total{head}                    non-abstained head scores
    vg_head_p_spoof{head}                         score distribution (drift input)
    vg_alerts_total{band,tier,shadow}             policy decisions at ELEVATED/HIGH
    vg_degraded_windows_total                     windows scored under load shedding (B14-T06)
    vg_active_sessions                            live calls
    vg_inference_queue_depth                      batcher queue (B14)
    vg_model_info{head,version}                   1 for the model version in use
    vg_score_drift_psi{head}                      population stability index vs reference (drift.py)
    vg_gpu_utilization_ratio{gpu}                 when NVML / torch.cuda is available
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0)
P_BUCKETS = tuple(i / 20 for i in range(1, 21))


class PipelineMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.windows = Counter("vg_windows_total", "Windows scored", ["state"], registry=r)
        self.window_latency = Histogram(
            "vg_window_latency_seconds", "Per-window latency", buckets=LATENCY_BUCKETS, registry=r
        )
        self.head_latency = Histogram(
            "vg_head_latency_seconds",
            "Per-head latency",
            ["head"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.abstain = Counter(
            "vg_head_abstain_total", "Head abstentions", ["head", "reason"], registry=r
        )
        self.scored = Counter(
            "vg_head_scores_total", "Head scores (not abstained)", ["head"], registry=r
        )
        self.p_spoof = Histogram(
            "vg_head_p_spoof", "Head p_spoof distribution", ["head"], buckets=P_BUCKETS, registry=r
        )
        self.alerts = Counter(
            "vg_alerts_total",
            "Policy decisions at ELEVATED/HIGH",
            ["band", "tier", "shadow"],
            registry=r,
        )
        self.degraded = Counter(
            "vg_degraded_windows_total", "Windows under load shedding", registry=r
        )
        self.sessions = Gauge("vg_active_sessions", "Live sessions", registry=r)
        self.queue_depth = Gauge(
            "vg_inference_queue_depth", "Inference batcher queue depth", registry=r
        )
        self.model_info = Gauge(
            "vg_model_info", "Model version in use", ["head", "version"], registry=r
        )
        self.drift_psi = Gauge(
            "vg_score_drift_psi", "PSI of bona fide-proxy scores vs reference", ["head"], registry=r
        )
        self.gpu_util = Gauge(
            "vg_gpu_utilization_ratio", "GPU utilisation (0-1)", ["gpu"], registry=r
        )
        self._seen_versions: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ pipeline hooks
    def on_session(self, delta: int) -> None:
        self.sessions.inc(delta)

    def on_window(
        self,
        head_scores: Iterable[Any],
        fused: Any,
        elapsed_s: float,  # noqa: ANN401
        degraded: bool = False,
    ) -> None:
        self.windows.labels(state=getattr(fused.state, "value", str(fused.state))).inc()
        self.window_latency.observe(elapsed_s)
        if degraded:
            self.degraded.inc()
        for s in head_scores:
            head = getattr(s.head_id, "value", str(s.head_id))
            self.head_latency.labels(head=head).observe(s.latency_ms / 1000)
            key = (head, s.model_version)
            if key not in self._seen_versions:
                for h, v in [k for k in self._seen_versions if k[0] == head]:
                    self.model_info.labels(head=h, version=v).set(0)
                    self._seen_versions.discard((h, v))
                self._seen_versions.add(key)
                self.model_info.labels(head=head, version=s.model_version).set(1)
            if s.abstain:
                reason = getattr(s.abstain_reason, "value", str(s.abstain_reason))
                self.abstain.labels(head=head, reason=reason).inc()
            else:
                self.scored.labels(head=head).inc()
                self.p_spoof.labels(head=head).observe(float(s.p_spoof))

    def on_decision(self, band: str, tier: str, shadow: bool) -> None:
        if band in ("ELEVATED", "HIGH"):
            self.alerts.labels(band=band, tier=tier, shadow=str(bool(shadow)).lower()).inc()

    def sample_gpu(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    self.gpu_util.labels(gpu=str(i)).set(torch.cuda.utilization(i) / 100)
        except Exception:  # noqa: BLE001, S110 - NVML missing: metric simply absent
            pass
