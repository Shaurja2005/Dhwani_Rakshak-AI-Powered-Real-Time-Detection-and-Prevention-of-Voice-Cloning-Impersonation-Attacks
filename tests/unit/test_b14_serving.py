"""B14 — distillation, INT8, ONNX export/parity, batching server, load shedding, inference service."""

from __future__ import annotations

import base64
import datetime as dt
import importlib.util
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from ml.export.distill import (
    DistillConfig,
    Distiller,
    build_student,
    param_count,
    truncate_frontend,
)
from ml.export.quantize import compare, model_size_mb, quantize_dynamic_int8, write_report
from ml.export.to_onnx import (
    INPUT_NAME,
    OUTPUT_NAMES,
    ExportWrapper,
    OnnxUnavailableError,
    export_onnx,
)
from packages.vg_core.models import (
    AbstainReason,
    AnalysisWindow,
    CallMetadata,
    HeadScore,
    SessionContext,
)
from packages.vg_core.sample_store import put_samples
from packages.vg_core.stub_head import StubHead
from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel
from services.api_gateway.pipeline import Services, SessionPipeline
from services.inference.backends import TorchBackend, xlsr_shaped_proxy
from services.inference.batching import BatchingInferenceServer, OverloadedError
from services.inference.degrade import DegradeConfig, LoadController
from services.inference.main import create_app
from services.inference.served_head import BatchedHeadA

SID = "b14-session"
HAS_ONNX = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)


def tiny(seed: int = 0) -> HeadAModel:
    torch.manual_seed(seed)
    return HeadAModel(HeadAConfig(frontend="tiny", frontend_layers=[3, 4, 5], emb_dim=32)).eval()


def wavs(n: int = 6, samples: int = 16000, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(
        0.1 * np.random.default_rng(seed).standard_normal((n, samples)).astype(np.float32)
    )


def window(wid: int = 0, voiced_ms: int = 2500) -> AnalysisWindow:
    ref = f"shm://{SID}/{wid}"
    put_samples(ref, (0.1 * np.random.default_rng(wid).standard_normal(48000)).astype(np.float32))
    return AnalysisWindow(
        session_id=SID,
        window_id=wid,
        start_ms=wid * 1000,
        end_ms=wid * 1000 + 3000,
        samples_ref=ref,
        voiced_ms=voiced_ms,
        snr_db=30.0,
        clipping_ratio=0.0,
        quality_ok=True,
        quality_flags=[],
        original_sample_rate=16000,
    )


CTX = SessionContext(session_id=SID, tenant_id="t")


class FakeBackend:
    name = "fake"

    def __init__(self, trained: bool = True, delay_s: float = 0.0, fail: bool = False) -> None:
        self.trained = trained
        self.model_version = "A@fake-v1"
        self.delay_s = delay_s
        self.fail = fail
        self.batch_sizes: list[int] = []

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(len(batch))
        time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("backend down")
        return batch.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------- T01 distillation
def test_student_is_truncated_and_smaller() -> None:
    teacher = tiny()
    student = build_student(teacher, keep_layers=4)
    assert len(student.frontend.layers) == 4 and student.frontend.num_hidden_states == 5
    assert student.cfg.frontend_layers == [3, 4, 5][:2] and student.cfg.version.endswith(
        "-distilled"
    )
    assert param_count(student) < param_count(teacher)
    assert len(teacher.frontend.layers) == 10  # teacher untouched
    with pytest.raises(TypeError):
        truncate_frontend(torch.nn.Identity(), 2)  # type: ignore[arg-type]


def test_distillation_moves_student_towards_teacher() -> None:
    teacher, x = tiny(seed=0), wavs(8)
    student = build_student(teacher, keep_layers=3)
    d = Distiller([teacher], student, DistillConfig(lr=3e-3, gamma=0.0))
    before = d.agreement(x)["mse"]
    losses = [d.step(x) for _ in range(25)]
    after = d.agreement(x)
    assert losses[-1] < losses[0]
    assert after["mse"] < before and set(after) == {"mse", "pearson"}
    assert all(not p.requires_grad for p in student.frontend.parameters())  # frozen front-end


def test_distill_cli_end_to_end_and_student_reloads(tmp_path: Path) -> None:
    import yaml

    from ml.export import distill
    from packages.vg_models.heads.head_a_ssl.model import load_checkpoint, save_checkpoint

    save_checkpoint(tiny(), tmp_path / "teacher.pt", {"lineage": "research"})
    cfg = {
        "lineage": "research",
        "allow_noncommercial": True,
        "seed": 0,
        "teachers": [str(tmp_path / "teacher.pt")],
        "keep_layers": 3,
        "emb_dim": 16,
        "out": str(tmp_path / "student.pt"),
        "distill": {"epochs": 1},
        "data": {"source": "synthetic", "n_train": 8, "n_dev": 4},
        "batch_size": 4,
        "seconds": 1.0,
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert distill.main(["--config", str(tmp_path / "cfg.yaml")]) == 0
    student, meta = load_checkpoint(tmp_path / "student.pt")
    assert meta["keep_layers"] == 3 and meta["distilled_from"] == cfg["teachers"]
    assert len(student.frontend.layers) == 3  # truncation survives a reload
    with torch.no_grad():
        student(wavs(2))


# ---------------------------------------------------------------- T02 INT8
def test_int8_quantization_report() -> None:
    fp32 = tiny()
    int8 = quantize_dynamic_int8(tiny())
    labels = np.array([0, 1] * 3)
    rep = compare(fp32, int8, wavs(6), labels)
    assert rep.n == 6 and rep.size_int8_mb < rep.size_fp32_mb
    assert rep.mean_abs_score_diff < 0.05 and 0.0 <= rep.decision_flip_rate <= 1.0
    assert rep.eer_fp32 is not None and rep.eer_int8 is not None
    md = write_report(rep, "tiny", "test-cpu", "random noise")
    assert "decision_flip_rate" in md and "convolutions remain FP32" in md
    assert model_size_mb(fp32) > 0


def test_int8_works_on_transformer_encoder_layers() -> None:
    b = TorchBackend(xlsr_shaped_proxy(num_layers=1), trained=False, int8=True)
    out = b.infer(np.zeros((2, 16000), np.float32))
    assert out.shape == (2,) and b.model_version.endswith("-int8") and not b.trained


# ---------------------------------------------------------------- T03 ONNX
def test_export_wrapper_matches_head_a_calibration() -> None:
    m = tiny()
    w = ExportWrapper(m, cal_scale=8.0, cal_bias=0.5)
    with torch.no_grad():
        logit, raw = w(wavs(3))
        _, ref = m(wavs(3))
    assert torch.allclose(raw, ref)
    p_head = 1 / (1 + torch.exp(8.0 * raw + 0.5))  # HeadA.score convention
    assert torch.allclose(torch.sigmoid(logit), p_head, atol=1e-6)
    assert INPUT_NAME == "wav" and OUTPUT_NAMES == ["spoof_logit", "raw_score"]


@pytest.mark.skipif(HAS_ONNX, reason="onnx installed; covered by the parity test")
def test_export_without_onnx_fails_with_install_hint(tmp_path: Path) -> None:
    with pytest.raises(OnnxUnavailableError, match="pip install onnx"):
        export_onnx(tiny(), tmp_path / "m.onnx")


@pytest.mark.skipif(not HAS_ONNX, reason="pip install onnx onnxruntime (docs/SETUP_PENDING.md B14)")
def test_onnx_export_and_parity(tmp_path: Path) -> None:
    from ml.export.parity_check import check_parity
    from services.inference.backends import OrtBackend

    m = tiny()
    path = export_onnx(m, tmp_path / "head_a.onnx", seconds=1.0)
    rep = check_parity(m, path)
    assert rep.passed and len(rep.lengths) > 1, rep
    ort_b = OrtBackend(path)
    x = wavs(2, 24000).numpy()
    np.testing.assert_allclose(ort_b.infer(x), TorchBackend(m, True).infer(x), atol=1e-3)


# ---------------------------------------------------------------- T04 batching + served head
def test_batcher_groups_concurrent_requests() -> None:
    be = FakeBackend(delay_s=0.02)
    srv = BatchingInferenceServer(be, max_batch=8, max_wait_ms=30)
    try:
        futs = [srv.submit(np.full(1600, i, np.float32)) for i in range(8)]
        assert [f.result(timeout=5) for f in futs] == pytest.approx(list(range(8)))
        assert max(be.batch_sizes) > 1 and srv.stats.items == 8
        # different lengths never share a tensor
        a, b = srv.submit(np.ones(1600, np.float32)), srv.submit(np.ones(3200, np.float32))
        assert a.result(timeout=5) == b.result(timeout=5) == 1.0
    finally:
        srv.close()


def test_batcher_backpressure_expiry_and_errors() -> None:
    slow = FakeBackend(delay_s=0.3)
    srv = BatchingInferenceServer(slow, max_batch=1, max_wait_ms=0, max_queue=1)
    try:
        first = srv.submit(np.ones(160, np.float32))
        time.sleep(0.05)  # worker is now busy with `first`
        stale = srv.submit(np.ones(160, np.float32), timeout_ms=10)
        with pytest.raises(OverloadedError):
            srv.submit(np.ones(160, np.float32))  # queue full: rejected, never queued
        assert first.result(timeout=5) == 1.0
        with pytest.raises(TimeoutError):
            stale.result(timeout=5)  # its deadline passed before compute: dropped
        assert srv.stats.rejected == 1 and srv.stats.expired == 1
    finally:
        srv.close()
    bad = BatchingInferenceServer(FakeBackend(fail=True))
    try:
        with pytest.raises(RuntimeError, match="backend down"):
            bad.infer(np.ones(160, np.float32))
    finally:
        bad.close()


def test_batched_head_a_semantics() -> None:
    trained = BatchingInferenceServer(FakeBackend(trained=True))
    untrained = BatchingInferenceServer(FakeBackend(trained=False))
    busy = BatchingInferenceServer(FakeBackend(delay_s=1.0), max_batch=1, max_queue=0)
    try:
        h = BatchedHeadA(trained)
        h.warmup()
        s = h.score(window(1), CTX)
        assert not s.abstain and 0 <= s.p_spoof <= 1 and s.model_version == "A@fake-v1"
        u = BatchedHeadA(untrained)
        u.warmup()
        su = u.score(window(2), CTX)
        assert (
            su.abstain
            and su.abstain_reason == AbstainReason.UNTRAINED
            and "raw_score" in su.evidence
        )
        o = BatchedHeadA(busy)
        o.warmup()
        so = o.score(window(3), CTX)
        assert (
            so.abstain
            and so.abstain_reason == AbstainReason.TIMEOUT
            and so.evidence["error"] == "OverloadedError"
        )
        assert (
            h.score(window(4, voiced_ms=200), CTX).abstain_reason
            == AbstainReason.INSUFFICIENT_SPEECH
        )
    finally:
        for srv in (trained, untrained, busy):
            srv.close()


def test_triton_config_matches_export_contract() -> None:
    cfg = Path("deploy/triton/models/head_a/config.pbtxt").read_text(encoding="utf-8")
    assert (
        'backend: "onnxruntime"' in cfg
        and "dynamic_batching" in cfg
        and "timeout_action: REJECT" in cfg
    )
    for name in [INPUT_NAME, *OUTPUT_NAMES]:
        assert f'name: "{name}"' in cfg


def test_inference_service_http() -> None:
    app = create_app(FakeBackend())
    with TestClient(app) as c:
        assert c.get("/healthz").json()["model_version"] == "A@fake-v1"
        body = {"wav_f32_base64": base64.b64encode(np.full(1600, 0.25, "<f4").tobytes()).decode()}
        r = c.post("/v1/infer/head_a", json=body).json()
        assert r["raw_score"] == pytest.approx(0.25) and r["trained"] is True
        assert c.post("/v1/infer/head_a", json={"wav_f32_base64": "%%"}).status_code == 400
        assert (
            c.post(
                "/v1/infer/head_a", json={"wav_f32_base64": base64.b64encode(b"\0" * 8).decode()}
            ).status_code
            == 400
        )
        assert c.get("/v1/stats").json()["items"] == 1
    full = create_app(FakeBackend(), max_queue=0)
    with TestClient(full) as c:
        assert c.post("/v1/infer/head_a", json=body).status_code == 503  # caller abstains (I10)


# ---------------------------------------------------------------- T06 degradation
def test_load_controller_hysteresis_and_score_shaping() -> None:
    ctl = LoadController(
        DegradeConfig(enter_ms=100, exit_ms=50, exit_after=3, ema_alpha=1.0, min_degraded_s=0)
    )
    ctl.observe(20)
    assert not ctl.degraded and not ctl.sheds("A")
    ctl.observe(150)
    assert ctl.degraded and ctl.sheds("A") and not ctl.sheds("B")
    ctl.observe(40)
    ctl.observe(40)
    assert ctl.degraded  # needs 3 calm windows
    ctl.observe(80)  # not calm: counter resets
    for _ in range(3):
        ctl.observe(10)
    assert not ctl.degraded and [d for _, d in ctl.transitions] == [True, False]
    ctl.note_overload()
    assert ctl.degraded
    held = LoadController(DegradeConfig(exit_ms=50, exit_after=1, ema_alpha=1.0, min_degraded_s=60))
    held.note_overload()
    held.observe(1)
    assert held.degraded  # calm, but not held long enough
    shed = ctl.shed_score(window(5), "A", "A@x")
    assert (
        shed.abstain and shed.abstain_reason == AbstainReason.TIMEOUT and shed.evidence["load_shed"]
    )
    s = HeadScore(
        session_id=SID,
        window_id=5,
        head_id="B",
        raw_score=0.1,
        p_spoof=0.7,
        abstain=False,
        confidence=0.5,
        latency_ms=3,
        model_version="B@x",
        calibration_version="c",
    )
    m = ctl.mark(s)
    assert m.confidence == pytest.approx(0.4) and m.evidence["degraded"] and m.p_spoof == 0.7


def test_load_controller_inflight_limit() -> None:
    ctl = LoadController(DegradeConfig(max_inflight=1, enter_ms=1e9))
    with ctl.track():
        assert not ctl.degraded
        with ctl.track():
            assert ctl.degraded
    assert ctl.inflight == 0


def _meta() -> CallMetadata:
    return CallMetadata(
        session_id=str(uuid.uuid4()),
        tenant_id="bank-a",
        direction="inbound",
        started_at=dt.datetime.now(dt.UTC),
        channel="voip",
        codec_hint="pcm",
        source_sample_rate=16000,
        consent_basis="legitimate_use",
    )


def _voiced(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )


def test_pipeline_sheds_expensive_head_when_degraded() -> None:
    ctl = LoadController(DegradeConfig(enter_ms=1e9, exit_ms=0, exit_after=10**6))
    ctl.note_overload()
    svc = Services(
        heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "ABC"], load=ctl
    )
    pipe = SessionPipeline(_meta(), svc, with_context=False)
    events = pipe.push_pcm16k(_voiced(5.0))
    ws = [e for e in events if e.type == "window_score"]
    assert ws and all(e.extra.get("degraded") for e in ws)
    a = [s for s in pipe.head_scores if s.head_id.value == "A"]
    others = [s for s in pipe.head_scores if s.head_id.value != "A"]
    assert a and all(s.abstain and s.evidence.get("load_shed") for s in a)
    assert others and all(s.evidence.get("degraded") and not s.abstain for s in others)
    assert all(e.data.p_spoof is not None for e in ws)  # still scoring on B + C
    assert ctl.shed_count == len(a)


def test_pipeline_without_controller_is_unchanged() -> None:
    svc = Services(
        heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "AB"], load=None
    )
    pipe = SessionPipeline(_meta(), svc, with_context=False)
    ws = [e for e in pipe.push_pcm16k(_voiced(4.0)) if e.type == "window_score"]
    assert ws and not any(e.extra for e in ws)
    assert not any(s.evidence.get("degraded") for s in pipe.head_scores)


def test_head_c_warmup_leaves_torch_threads_alone() -> None:
    from packages.vg_models.heads.head_c_prosody.head import HeadC

    before = torch.get_num_threads()
    HeadC().warmup()
    assert torch.get_num_threads() == before


def test_loadtest_script_smoke(tmp_path: Path) -> None:
    import scripts.loadtest as lt

    be = FakeBackend(trained=False, delay_s=0.005)
    res = lt.run_level(2, 4.0, be, ["A", "B"], shedding=True, max_batch=4)
    assert res.sessions == 2 and res.windows >= 2 and res.lat_ms
    assert len(lt.speechlike(1.0, 0)) == 32000
    assert threading.active_count() < 50


def test_quality_gate_tone_scaling_regression() -> None:
    """Found by the B14 load test: raw Goertzel power vs sum(x^2) flagged broadband audio."""
    import scripts.loadtest as lt
    from packages.vg_audio import quality as q

    t = np.arange(48000) / 16000
    speech = np.frombuffer(lt.speechlike(3.0, 0), "<i2").astype(np.float32) / 32768
    assert not q.detect_hold_music_or_tone(speech) and not q.detect_dtmf(speech)
    assert q.detect_hold_music_or_tone((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    dtmf = 0.2 * np.sin(2 * np.pi * 770 * t) + 0.2 * np.sin(2 * np.pi * 1336 * t)
    assert q.detect_dtmf(dtmf.astype(np.float32))
    tone = q._bin_fraction(
        (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
        440,
        16000,
        float(np.sum((0.3 * np.sin(2 * np.pi * 440 * t)) ** 2)),
    )
    assert tone == pytest.approx(1.0, abs=0.01)
    noise = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    assert q._bin_fraction(noise, 440, 16000, float(np.sum(noise**2))) < 0.01
