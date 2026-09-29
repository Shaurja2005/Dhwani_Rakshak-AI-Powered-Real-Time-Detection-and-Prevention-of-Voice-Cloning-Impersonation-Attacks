"""Build the Grafana dashboards (B17-T05) as JSON from one place.

    python deploy/observability/grafana/build_dashboards.py

* ``voiceguard-ops.json``    — on-call: latency, abstain rate, load shedding, queue,
  sessions, GPU, model versions in use.
* ``voiceguard-fraud.json``  — fraud analytics: alert rate by band/tier, shadow vs live,
  score distributions, drift (PSI) per head.

Every panel query only uses metric names defined in
``services/observability/metrics.py`` (checked by tests/unit/test_b17_ops.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "dashboards"
DS = {"type": "prometheus", "uid": "vg-prometheus"}


def panel(
    pid: int,
    title: str,
    exprs: list[tuple[str, str]],
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    kind: str = "timeseries",
    unit: str | None = None,
    description: str = "",
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "id": pid,
        "type": kind,
        "title": title,
        "description": description,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"refId": chr(65 + i), "expr": e, "legendFormat": leg, "datasource": DS}
            for i, (e, leg) in enumerate(exprs)
        ],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {},
    }
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    return p


def dashboard(uid: str, title: str, panels: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "uid": uid,
        "title": title,
        "tags": ["voiceguard"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-3h", "to": "now"},
        "panels": panels,
        "editable": False,
    }


def ops() -> dict[str, Any]:
    q95 = "histogram_quantile(0.95, sum by (le) (rate(vg_window_latency_seconds_bucket[5m])))"
    q50 = "histogram_quantile(0.50, sum by (le) (rate(vg_window_latency_seconds_bucket[5m])))"
    return dashboard(
        "vg-ops",
        "VoiceGuard — Operations",
        [
            panel(1, "Active calls", [("sum(vg_active_sessions)", "calls")], 0, 0, 6, 4, "stat"),
            panel(
                2,
                "Windows / s",
                [("sum(rate(vg_windows_total[1m]))", "windows/s")],
                6,
                0,
                6,
                4,
                "stat",
            ),
            panel(
                3,
                "Per-window latency p95",
                [(q95, "p95")],
                12,
                0,
                6,
                4,
                "stat",
                "s",
                "Target: < 0.7 s CPU-only, < 0.3 s on T4/L4 (IMPLEMENTATION_PLAN B14).",
            ),
            panel(
                4,
                "Load shedding (windows/s)",
                [("sum(rate(vg_degraded_windows_total[5m]))", "degraded")],
                18,
                0,
                6,
                4,
                "stat",
            ),
            panel(5, "Per-window latency", [(q50, "p50"), (q95, "p95")], 0, 4, 12, 8, unit="s"),
            panel(
                6,
                "Per-head latency p95",
                [
                    (
                        "histogram_quantile(0.95, sum by (le, head) (rate(vg_head_latency_seconds_bucket[5m])))",
                        "{{head}}",
                    )
                ],
                12,
                4,
                12,
                8,
                unit="s",
            ),
            panel(
                7,
                "Abstain ratio by head",
                [
                    (
                        "sum by (head) (rate(vg_head_abstain_total[5m])) / (sum by (head) "
                        "(rate(vg_head_abstain_total[5m])) + sum by (head) (rate(vg_head_scores_total[5m])))",
                        "{{head}}",
                    )
                ],
                0,
                12,
                6,
                8,
                unit="percentunit",
                description="A spike means audio quality changed, a head broke, or the model is untrained.",
            ),
            panel(
                11,
                "Abstentions / s by reason",
                [
                    (
                        "sum by (head, reason) (rate(vg_head_abstain_total[5m]))",
                        "{{head}} {{reason}}",
                    )
                ],
                6,
                12,
                6,
                8,
            ),
            panel(
                8,
                "Inference queue depth",
                [("max(vg_inference_queue_depth)", "queue")],
                12,
                12,
                6,
                8,
            ),
            panel(
                9,
                "GPU utilisation",
                [("avg by (gpu) (vg_gpu_utilization_ratio)", "gpu {{gpu}}")],
                18,
                12,
                6,
                8,
                unit="percentunit",
            ),
            panel(
                10,
                "Model versions in use",
                [("max by (head, version) (vg_model_info) == 1", "{{head}} {{version}}")],
                0,
                20,
                24,
                6,
                "table",
            ),
        ],
    )


def fraud() -> dict[str, Any]:
    return dashboard(
        "vg-fraud",
        "VoiceGuard — Fraud analytics",
        [
            panel(
                1,
                "Alerts / hour by band",
                [("sum by (band) (increase(vg_alerts_total[1h]))", "{{band}}")],
                0,
                0,
                12,
                8,
                "barchart",
            ),
            panel(
                2,
                "Alerts by tier (shadow vs live)",
                [
                    (
                        "sum by (tier, shadow) (increase(vg_alerts_total[1h]))",
                        "{{tier}} shadow={{shadow}}",
                    )
                ],
                12,
                0,
                12,
                8,
            ),
            panel(
                3,
                "Window states",
                [("sum by (state) (rate(vg_windows_total[5m]))", "{{state}}")],
                0,
                8,
                12,
                8,
                description="ABSTAIN is not LOW: it means the system could not judge the audio.",
            ),
            panel(
                4,
                "Score drift (PSI vs reference, bona fide proxy)",
                [("max by (head) (vg_score_drift_psi)", "{{head}}")],
                12,
                8,
                12,
                8,
                description="> 0.25 sustained: a new generator family may be in the wild (runbook).",
            ),
            panel(
                5,
                "p_spoof distribution by head",
                [("sum by (le, head) (rate(vg_head_p_spoof_bucket[15m]))", "{{head}} ≤{{le}}")],
                0,
                16,
                24,
                9,
                "heatmap",
            ),
        ],
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in (("voiceguard-ops.json", ops()), ("voiceguard-fraud.json", fraud())):
        (OUT / name).write_text(
            json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
