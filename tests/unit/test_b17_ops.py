"""B17 — metrics, drift, dashboards/alerts consistency, Dockerfiles, compose, k8s, registry rollout."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


def exported_metric_names() -> set[str]:
    from services.observability.metrics import PipelineMetrics

    suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_count", "_sum"), "gauge": ("",)}
    return {
        fam.name + sfx
        for fam in PipelineMetrics().registry.collect()
        for sfx in suffixes.get(fam.type, ("",))
    }


def promql_metrics(expr: str) -> set[str]:
    return set(re.findall(r"\b(vg_[a-z0-9_]+)", expr))


# ---------------------------------------------------------------- T04 metrics through the pipeline
def test_metrics_endpoint_reflects_live_calls() -> None:
    from packages.vg_core.stub_head import StubHead
    from sdks.python.voiceguard import call_metadata
    from services.api_gateway.app import create_app
    from services.api_gateway.auth import KeyStore
    from services.api_gateway.pipeline import Services
    from services.api_gateway.webhooks import WebhookRegistry
    from services.observability import Observability

    obs = Observability()
    svc = Services(
        observer=obs, heads_factory=lambda: [StubHead(h, abstain_fraction=0.3) for h in "ABC"]
    )
    keys = KeyStore()
    raw, _ = keys.create("bank-a", {"analyze"})
    c = TestClient(create_app(svc, keys, WebhookRegistry()))
    import base64
    import io

    import soundfile as sf

    t = np.arange(6 * 16000) / 16000
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), 16000, format="WAV")
    body = {
        "call_metadata": call_metadata("bank-a"),
        "audio_base64": base64.b64encode(buf.getvalue()).decode(),
    }
    assert (
        c.post(
            "/v1/analyze/file", json=body, headers={"Authorization": f"Bearer {raw}"}
        ).status_code
        == 200
    )
    text = c.get("/metrics").text  # public: no API key needed
    assert "vg_windows_total" in text and "vg_window_latency_seconds_bucket" in text
    assert re.search(r'vg_head_latency_seconds_count\{head="A"\} [1-9]', text)
    assert 'vg_model_info{head="A",version="stub' in text
    assert "vg_active_sessions 0.0" in text  # session opened and closed
    assert "vg_head_abstain_total" in text  # stub abstains 30% of the time
    no_obs = TestClient(create_app(Services(), KeyStore(), WebhookRegistry()))
    assert no_obs.get("/metrics").status_code == 404


def test_observer_never_breaks_the_call() -> None:
    from services.observability import Observability

    obs = Observability()
    obs.on_window([object()], object(), 0.1, False)  # garbage in: swallowed (I10)


# ---------------------------------------------------------------- T06 drift
def test_drift_monitor_detects_shift_on_bona_fide_proxy() -> None:
    from services.observability.drift import DriftMonitor, psi

    rng = np.random.default_rng(0)
    assert psi(np.array([10, 10, 10]), np.array([10, 10, 10])) < 1e-6
    mon = DriftMonitor(min_samples=200)
    mon.set_reference("A", rng.beta(2, 8, 5000))  # genuine calls: low p_spoof
    assert mon.status("A").level == "insufficient_data"
    mon.observe_session({"A": list(rng.beta(2, 8, 400))}, "LOW")
    st = mon.status("A")
    assert st.level == "stable" and st.psi < 0.1 and st.ks < 0.1
    mon.observe_session(
        {"A": list(rng.beta(8, 2, 5000))}, "HIGH"
    )  # flagged calls are not the proxy
    assert mon.status("A").level == "stable"
    mon.observe_confirmed_genuine({"A": list(rng.beta(5, 5, 5000))})  # new family scored "genuine"
    shifted = mon.status("A")
    assert shifted.level == "shifted" and shifted.psi > 0.25


def test_drift_reference_roundtrip(tmp_path: Path) -> None:
    from services.observability.drift import DriftMonitor

    mon = DriftMonitor(min_samples=10)
    mon.observe_confirmed_genuine({"B": [0.1] * 50})
    mon.freeze_current_as_reference("B")
    mon.save(tmp_path / "ref.json")
    other = DriftMonitor(min_samples=10)
    other.load(tmp_path / "ref.json")
    other.observe_confirmed_genuine({"B": [0.1] * 50})
    assert other.status("B").level == "stable"
    with pytest.raises(ValueError):
        DriftMonitor(min_samples=10).freeze_current_as_reference("A")


# ---------------------------------------------------------------- T05 dashboards + alerts
def test_dashboards_and_alerts_only_use_exported_metrics() -> None:
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(ROOT / "deploy/observability/grafana/build_dashboards.py")], check=True
    )
    known = exported_metric_names()
    exprs = []
    for f in (ROOT / "deploy/observability/grafana/dashboards").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d["uid"] in ("vg-ops", "vg-fraud") and d["panels"]
        exprs += [t["expr"] for p in d["panels"] for t in p["targets"]]
    rules = yaml.safe_load((ROOT / "deploy/observability/alerts.yml").read_text(encoding="utf-8"))
    alerts = [r for g in rules["groups"] for r in g["rules"]]
    exprs += [r["expr"] for r in alerts]
    unknown = {m for e in exprs for m in promql_metrics(e)} - known
    assert not unknown, unknown
    runbook = (ROOT / "docs/runbooks/ONCALL.md").read_text(encoding="utf-8").lower()
    anchors = {
        "#" + re.sub(r"[^a-z0-9 -]", "", h).strip().replace(" ", "-")
        for h in re.findall(r"^## (.+)$", runbook, re.M)
    }
    for r in alerts:
        assert r["annotations"]["runbook"].split("ONCALL.md")[1] in anchors, r["alert"]


# ---------------------------------------------------------------- T01 Dockerfiles
def test_dockerfiles_are_generated_hardened_and_in_sync() -> None:
    import subprocess
    import sys

    assert (
        subprocess.run(
            [sys.executable, str(ROOT / "deploy/docker/render_dockerfiles.py"), "--check"]
        ).returncode
        == 0
    )
    for f in (ROOT / "services").glob("*/Dockerfile"):
        text = f.read_text(encoding="utf-8")
        assert text.count("\nFROM ") + text.startswith("FROM ") >= 2, f  # multi-stage
        assert (
            "USER 10001:10001" in text and "HEALTHCHECK" in text and "python:3.11-slim" in text
        ), f
        assert "build-essential" not in text.split("AS runtime")[1], f  # no compilers in runtime
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in ("data/", "runs/", "**/*.db", ".env", "**/*.pt"):
        assert pattern in ignore


# ---------------------------------------------------------------- T02 compose
def test_prod_compose_is_isolated_and_hardened() -> None:
    c = yaml.safe_load((ROOT / "deploy/docker-compose.prod.yml").read_text(encoding="utf-8"))
    assert c["networks"]["vg-internal"]["internal"] is True
    hard = c["x-hardening"]
    assert (
        hard["read_only"]
        and "ALL" in hard["cap_drop"]
        and "no-new-privileges:true" in hard["security_opt"]
    )
    svcs = c["services"]
    for name in ("gateway", "privacy", "privacy-retention", "prometheus", "grafana"):
        assert svcs[name]["networks"], name
    assert svcs["privacy"]["networks"] == ["vg-internal"]  # no route out
    assert svcs["gateway"]["environment"]["VG_LOAD_SHEDDING"] == "1"
    assert "127.0.0.1" in svcs["grafana"]["ports"][0]
    assert svcs["grafana"]["environment"]["GF_ANALYTICS_REPORTING_ENABLED"] == "false"
    assert "tmpfs" in hard and "/tmp" in hard["tmpfs"][0]


# ---------------------------------------------------------------- T03 k8s
def test_k8s_manifests_hardened_and_config_in_sync() -> None:
    base = ROOT / "deploy/k8s/base"
    docs = [
        d
        for f in base.glob("*.yaml")
        if f.name != "kustomization.yaml"
        for d in yaml.safe_load_all(f.read_text(encoding="utf-8"))
        if d
    ]
    kinds = {d["kind"] for d in docs}
    assert {"Deployment", "Service", "CronJob", "NetworkPolicy", "HorizontalPodAutoscaler"} <= kinds
    deny = [
        d for d in docs if d["kind"] == "NetworkPolicy" and d["metadata"]["name"] == "default-deny"
    ]
    assert deny and deny[0]["spec"]["policyTypes"] == ["Ingress", "Egress"]
    for d in docs:
        if d["kind"] in ("Deployment", "CronJob"):
            spec = (
                d["spec"]["template"]["spec"]
                if d["kind"] == "Deployment"
                else d["spec"]["jobTemplate"]["spec"]["template"]["spec"]
            )
            assert spec["securityContext"]["runAsNonRoot"] is True, d["metadata"]["name"]
            for ctr in spec["containers"]:
                sc = ctr["securityContext"]
                assert sc["readOnlyRootFilesystem"] and sc["allowPrivilegeEscalation"] is False
    for rel in ("consent_matrix.yaml", "tenants/default.yaml"):
        assert (base / "config" / rel).read_text(encoding="utf-8") == (
            ROOT / "config/privacy" / rel
        ).read_text(encoding="utf-8"), rel


# ---------------------------------------------------------------- T07 registry-driven hot swap
def test_registry_promote_and_rollback_hot_swap_served_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    import services.inference.served_head as sh
    from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel, save_checkpoint
    from packages.vg_models.registry import ModelRegistry

    for v, seed in (("A@tiny-v1", 0), ("A@tiny-v2", 1)):
        torch.manual_seed(seed)
        save_checkpoint(
            HeadAModel(HeadAConfig(frontend="tiny", frontend_layers=[3], emb_dim=8)),
            tmp_path / f"{v}.pt",
            {"lineage": "research"},
        )
    reg_path = tmp_path / "released.yaml"
    reg = ModelRegistry(reg_path)
    for v in ("A@tiny-v1", "A@tiny-v2"):
        reg.register(
            v, "head_a", tmp_path / f"{v}.pt", "research", eval_run_id="r1", fairness_gate="pass"
        )
    monkeypatch.setenv("VG_INFERENCE_BACKEND", "registry")
    monkeypatch.setenv("VG_MODEL_REGISTRY", str(reg_path))
    monkeypatch.setattr(sh, "_shared", None)
    monkeypatch.setattr(sh, "_registry_state", {"mtime": None, "version": None})
    try:
        server = sh.shared_server()
        assert not server.backend.trained  # nothing active yet: placeholder abstains "untrained"
        reg.promote("A@tiny-v1")
        _bump(reg_path)
        assert sh.shared_server().backend.model_version == "A@tiny-v1" and server.backend.trained
        reg.promote("A@tiny-v2")
        _bump(reg_path)
        assert sh.shared_server().backend.model_version == "A@tiny-v2"
        reg.rollback("head_a")
        _bump(reg_path)
        assert sh.shared_server().backend.model_version == "A@tiny-v1"  # instant rollback
        assert sh.shared_server().infer(np.zeros(16000, np.float32)) == pytest.approx(
            sh.shared_server().infer(np.zeros(16000, np.float32))
        )
    finally:
        if sh._shared is not None:
            sh._shared.close()
        monkeypatch.setattr(sh, "_shared", None)


def _bump(path: Path) -> None:
    """Make sure the mtime changes even on coarse filesystem clocks."""
    import os

    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 1 + dt.datetime.now().microsecond / 1e6))


def test_model_rollout_cli(tmp_path: Path) -> None:
    import scripts.model_rollout as mr
    from packages.vg_eval.report import RunRecord

    art = tmp_path / "m.pt"
    art.write_bytes(b"x")
    real = RunRecord("A@cli-v1", "head_a", {}, {"eval_sets": ["itw"], "synthetic": False})
    real.gate = {"status": "pass", "detail": ""}
    real.save(tmp_path / "runs")
    syn = RunRecord("A@cli-syn", "head_a", {}, {"eval_sets": ["synthetic"], "synthetic": True})
    syn.save(tmp_path / "runs")
    reg = ["--registry", str(tmp_path / "r.yaml")]
    assert (
        mr.main(
            [
                *reg,
                "register",
                "--version",
                "A@cli-v1",
                "--kind",
                "head_a",
                "--path",
                str(art),
                "--lineage",
                "research",
            ]
        )
        == 0
    )
    assert mr.main([*reg, "promote", "--version", "A@cli-v1", "--actor", "t"]) == 2  # no eval yet
    assert (
        mr.main(
            [
                *reg,
                "attach-eval",
                "--version",
                "A@cli-v1",
                "--run-id",
                syn.run_id,
                "--runs-dir",
                str(tmp_path / "runs"),
            ]
        )
        == 2
    )  # synthetic refused
    assert (
        mr.main(
            [
                *reg,
                "attach-eval",
                "--version",
                "A@cli-v1",
                "--run-id",
                real.run_id,
                "--runs-dir",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    assert mr.main([*reg, "promote", "--version", "A@cli-v1", "--actor", "t", "--commercial"]) == 2
    assert mr.main([*reg, "promote", "--version", "A@cli-v1", "--actor", "t"]) == 0
    assert mr.main([*reg, "status"]) == 0
