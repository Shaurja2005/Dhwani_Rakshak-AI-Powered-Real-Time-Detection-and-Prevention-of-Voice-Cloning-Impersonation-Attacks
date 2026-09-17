"""B4 — Head A (SSL anti-spoof) unit tests. CPU-only, tiny random-init front-end."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.data.license_gate import LicenseViolation
from ml.data.manifest import ManifestRow
from ml.training import train_head_a
from ml.training.augment import Augmenter
from ml.training.continual import EWC, replay_buffer
from ml.training.dataset import FamilyBalancedSampler, crop_or_pad, group_key
from ml.training.losses import AMSoftmax, OCSoftmax
from ml.training.metrics import compute_eer, fit_platt
from ml.training.robust import SAM, DomainAdversary, LoRALinear, apply_lora, mixup
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadScore, SessionContext
from packages.vg_core.sample_store import SampleStore, put_samples
from packages.vg_models.heads.head_a_ssl.backends import AASISTLite, Nes2Net
from packages.vg_models.heads.head_a_ssl.frontend import LayerWeightedSum, TinySSLFrontend
from packages.vg_models.heads.head_a_ssl.head import HeadA
from packages.vg_models.heads.head_a_ssl.model import (
    HeadAConfig,
    HeadAModel,
    load_checkpoint,
    save_checkpoint,
)

SID = "01900b1a-0000-7000-8000-00000000b4a1"
CTX = SessionContext(session_id=SID, tenant_id="t")


def _tiny_model(backend: str = "nes2net") -> HeadAModel:
    torch.manual_seed(0)
    return HeadAModel(
        HeadAConfig(frontend="tiny", frontend_layers=[3, 4, 5], backend=backend, emb_dim=32)
    )


def _window(
    wid: int = 0, voiced_ms: int = 2500, quality_ok: bool = True, pcm: bool = True
) -> AnalysisWindow:
    ref = f"shm://{SID}/{wid}"
    if pcm:
        rng = np.random.default_rng(wid)
        put_samples(ref, (0.1 * rng.standard_normal(48000)).astype(np.float32))
    return AnalysisWindow(
        session_id=SID,
        window_id=wid,
        start_ms=0,
        end_ms=3000,
        samples_ref=ref,
        voiced_ms=voiced_ms,
        snr_db=20.0,
        clipping_ratio=0.0,
        quality_ok=quality_ok,
        original_sample_rate=8000,
    )


def _row(i: int, label: str = "bona_fide", family: str | None = None) -> ManifestRow:
    return ManifestRow(
        utt_id=f"u{i}",
        path=f"p{i}",
        label=label,
        generator_family=family,  # type: ignore[arg-type]
        language="hi",
        speaker_id=f"s{i}",
        source_corpus="indicsynth",
        license="CC-BY-NC-4.0",
        commercial_use=False,
        duration_s=3.0,
        sample_rate=16000,
    )


# ---------------------------------------------------------------- sample store (cross-block helper)
def test_sample_store_ttl_eviction_and_session_drop() -> None:
    st = SampleStore(max_entries=2, ttl_s=0.05)
    st.put("shm://s1/0", np.zeros(4))
    st.put("shm://s1/1", np.zeros(4))
    st.put("shm://s2/0", np.zeros(4))
    assert st.get("shm://s1/0") is None  # evicted (bounded)
    assert st.drop_session("s1") == 1
    time.sleep(0.2)  # well beyond Windows monotonic-clock resolution (15.6 ms)
    assert st.get("shm://s2/0") is None  # expired
    arr = SampleStore().put("x", np.ones(3))
    assert arr == "x"


# ---------------------------------------------------------------- front-end + back-ends (T02/T03)
def test_layer_weighted_sum_starts_uniform_and_validates_range() -> None:
    fe = TinySSLFrontend(hidden_size=32, num_layers=4)
    states = fe(torch.randn(2, 16000))
    assert len(states) == 5 and states[0].shape[0] == 2 and states[0].shape[2] == 32
    lws = LayerWeightedSum([1, 2, 3])
    assert np.allclose(lws.weights(), 1 / 3)
    assert lws(states).shape == states[0].shape
    with pytest.raises(ValueError):
        LayerWeightedSum([9])(states)


@pytest.mark.parametrize("cls", [Nes2Net, AASISTLite])
def test_backends_map_ssl_features_to_embeddings(cls: type) -> None:
    net = cls(in_dim=64, emb_dim=24).eval()
    assert net(torch.randn(3, 50, 64)).shape == (3, 24)


def test_nes2net_has_no_projection_bottleneck() -> None:
    net = Nes2Net(in_dim=1024, emb_dim=160)
    first_conv = next(m for m in net.blocks.modules() if isinstance(m, torch.nn.Conv1d))
    assert net.trim == 1024 and first_conv.in_channels == 1024 // 16


@pytest.mark.parametrize("backend", ["nes2net", "aasist"])
def test_model_forward_and_version(backend: str) -> None:
    m = _tiny_model(backend).eval()
    emb, score = m(torch.randn(2, 16000))
    assert emb.shape == (2, 32) and torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert score.shape == (2,) and float(score.abs().max()) <= 1.0 + 1e-5
    assert m.model_version == f"A@tiny-{backend}-v0.1.0"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    m = _tiny_model().eval()
    save_checkpoint(m, tmp_path / "c.pt", {"lineage": "research"})
    m2, meta = load_checkpoint(tmp_path / "c.pt")
    x = torch.randn(1, 16000)
    assert meta["lineage"] == "research"
    assert torch.allclose(m(x)[1], m2(x)[1], atol=1e-5)


# ---------------------------------------------------------------- losses (T04)
def test_oc_softmax_rewards_correct_side_of_margins() -> None:
    loss = OCSoftmax()
    labels = torch.tensor([1, 0])
    good = loss(torch.tensor([0.95, 0.0]), labels)
    bad = loss(torch.tensor([0.0, 0.95]), labels)
    assert good < bad


def test_am_softmax_decreases_when_embeddings_match_centres() -> None:
    torch.manual_seed(0)
    loss = AMSoftmax(8)
    bona = torch.nn.functional.normalize(torch.randn(8), dim=0)
    emb_good = torch.stack([bona, torch.nn.functional.normalize(loss.spoof_center.detach(), dim=0)])
    emb_bad = emb_good.flip(0)
    y = torch.tensor([1, 0])
    assert loss(emb_good, bona, y) < loss(emb_bad, bona, y)


# ---------------------------------------------------------------- augmentation (T05)
@pytest.mark.parametrize("algo", [1, 2, 3, 4])
def test_rawboost_keeps_shape_and_range(algo: int) -> None:
    x = (0.3 * np.sin(np.arange(16000) / 10)).astype(np.float32)
    y = Augmenter(rawboost_algo=algo, p_rawboost=1.0)(x, np.random.default_rng(0), "k")
    assert y.shape == x.shape and y.dtype == np.float32 and np.abs(y).max() <= 1
    assert not np.allclose(x, y)


def test_crop_or_pad() -> None:
    rng = np.random.default_rng(0)
    assert len(crop_or_pad(np.ones(10), 25, rng, True)) == 25
    assert len(crop_or_pad(np.arange(100.0), 40, rng, False)) == 40


# ---------------------------------------------------------------- data balancing (T07)
def test_family_balanced_sampler_equalises_attack_families() -> None:
    rows = [_row(i) for i in range(100)]
    rows += [_row(1000 + i, "spoof", "big_family") for i in range(900)]
    rows += [_row(5000 + i, "spoof", "small_family") for i in range(10)]
    s = FamilyBalancedSampler(rows, num_samples=2000, seed=1)
    counts: dict[str, int] = {}
    for i in s:
        counts[group_key(rows[i])] = counts.get(group_key(rows[i]), 0) + 1
    assert counts["bona_fide"] == 1000
    assert counts["spoof:big_family"] == counts["spoof:small_family"] == 500


# ---------------------------------------------------------------- robustness (T08)
def test_sam_step_changes_params() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 1)
    opt = SAM(lin.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    before = lin.weight.detach().clone()

    def closure() -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(lin(x), y)
        loss.backward()
        return loss

    closure()
    opt.step(closure)
    assert not torch.allclose(before, lin.weight)


def test_mixup_and_gradient_reversal() -> None:
    wav, y = torch.randn(4, 100), torch.tensor([0, 1, 0, 1])
    mixed, ya, yb, lam = mixup(wav, y, 0.4)
    assert mixed.shape == wav.shape and 0 <= lam <= 1 and ya.shape == yb.shape
    emb = torch.randn(4, 8, requires_grad=True)
    adv = DomainAdversary(8, 2, lam=1.0)
    adv(emb, torch.tensor([0, 1, 0, 1])).backward()
    emb2 = emb.detach().clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(adv.clf(emb2), torch.tensor([0, 1, 0, 1])).backward()
    assert emb.grad is not None and emb2.grad is not None
    assert torch.allclose(emb.grad, -emb2.grad, atol=1e-6)  # gradient reversed


def test_lora_starts_as_identity_and_only_adapters_train() -> None:
    parent = torch.nn.Module()
    parent.q_proj = torch.nn.Linear(6, 6)  # type: ignore[assignment]
    x = torch.randn(2, 6)
    ref = parent.q_proj(x)
    assert apply_lora(parent) == 1 and isinstance(parent.q_proj, LoRALinear)
    assert torch.allclose(parent.q_proj(x), ref)
    trainable = {n for n, p in parent.named_parameters() if p.requires_grad}
    assert trainable == {"q_proj.a", "q_proj.b"}


# ---------------------------------------------------------------- continual learning (T10)
def test_ewc_penalty_zero_at_anchor_and_replay_is_stratified() -> None:
    m = _tiny_model()
    batches = [(torch.randn(2, 16000), torch.tensor([1, 0]))]
    ewc = EWC(m, batches, OCSoftmax())
    assert float(ewc.penalty(m)) == 0.0
    with torch.no_grad():
        m.center.add_(1.0)
    assert float(ewc.penalty(m)) > 0
    rows = [_row(i) for i in range(50)] + [_row(100 + i, "spoof", f"f{i % 2}") for i in range(50)]
    buf = replay_buffer(rows, per_group=5)
    assert len(buf) == 15


# ---------------------------------------------------------------- metrics
def test_eer_and_platt() -> None:
    assert compute_eer(np.array([0.9, 0.8, 0.7]), np.array([0.1, 0.2, 0.3]))[0] == 0.0
    rng = np.random.default_rng(0)
    bona, spoof = rng.normal(1, 1, 5000), rng.normal(-1, 1, 5000)
    eer, _ = compute_eer(bona, spoof)
    assert 0.13 < eer < 0.19  # theoretical Φ(-1) ≈ 0.159
    s = np.concatenate([bona, spoof])
    a, b = fit_platt(s, np.concatenate([np.zeros(5000), np.ones(5000)]))
    assert a > 0  # higher score -> lower p_spoof


# ---------------------------------------------------------------- training loop (T06)
def test_smoke_training_writes_loadable_checkpoint(tmp_path: Path) -> None:
    import yaml

    cfg = yaml.safe_load(Path("ml/training/configs/head_a_smoke.yaml").read_text())
    cfg["out_dir"] = str(tmp_path)
    cfg["train"]["epochs"] = 1
    cfg["data"]["n_train"], cfg["data"]["n_dev"] = 16, 8
    result = train_head_a.train(cfg)
    _, meta = load_checkpoint(result["checkpoint"])
    assert meta["lineage"] == "research" and "calibration" in meta
    assert (tmp_path / "head_a_smoke" / "metrics.jsonl").exists()


def test_training_refuses_noncommercial_data_in_commercial_lineage(tmp_path: Path) -> None:
    import yaml

    cfg = yaml.safe_load(Path("ml/training/configs/head_a_smoke.yaml").read_text())
    cfg.update(out_dir=str(tmp_path), lineage="commercial", allow_noncommercial=False)
    with pytest.raises(LicenseViolation):
        train_head_a.train(cfg)


def test_stage_configs_parse_and_declare_lineage() -> None:
    import yaml

    for stage in (1, 2, 3):
        cfg = yaml.safe_load(Path(f"ml/training/configs/head_a_stage{stage}.yaml").read_text())
        assert cfg["lineage"] == "research" and cfg["allow_noncommercial"] is True
        assert cfg["frontend_layers"] == [5, 6, 7, 8, 9]
        HeadAConfig.from_dict(cfg)


# ---------------------------------------------------------------- inference head (T09)
@pytest.fixture(scope="module")
def head() -> HeadA:
    h = HeadA(device="cpu", budget_ms=5000)
    h.warmup()
    return h


def test_untrained_head_abstains_but_records_raw_score(head: HeadA) -> None:
    s = head.score(_window(0), CTX)
    assert isinstance(s, HeadScore) and s.abstain and s.p_spoof is None
    assert s.abstain_reason == AbstainReason.UNTRAINED
    assert s.evidence["untrained"] is True and "raw_score" in s.evidence
    assert s.model_version.startswith("A@tiny-")


def test_head_abstains_on_quality_speech_and_missing_samples(head: HeadA) -> None:
    assert (
        head.score(_window(1, quality_ok=False), CTX).abstain_reason == AbstainReason.QUALITY_GATE
    )
    assert (
        head.score(_window(2, voiced_ms=500), CTX).abstain_reason
        == AbstainReason.INSUFFICIENT_SPEECH
    )
    s = head.score(_window(3, pcm=False), CTX)
    assert s.abstain_reason == AbstainReason.INSUFFICIENT_SPEECH


def test_head_times_out_to_abstain() -> None:
    h = HeadA(device="cpu", budget_ms=1, emit_untrained_scores=True)
    h.warmup()
    h._infer = lambda pcm: (time.sleep(0.2), 0.0)[1]  # type: ignore[method-assign]
    s = h.score(_window(4), CTX)
    assert s.abstain and s.abstain_reason == AbstainReason.TIMEOUT


def test_head_never_raises() -> None:
    h = HeadA(device="cpu", budget_ms=5000, emit_untrained_scores=True)
    h.warmup()

    def boom(pcm: np.ndarray) -> float:
        raise RuntimeError("cuda exploded")

    h._infer = boom  # type: ignore[method-assign]
    assert h.score(_window(5), CTX).abstain


def test_trained_checkpoint_emits_calibrated_scores(tmp_path: Path) -> None:
    m = _tiny_model().eval()
    save_checkpoint(
        m,
        tmp_path / "c.pt",
        {"calibration": {"scale": 5.0, "bias": 0.0, "version": "cal-2026-09-17"}},
    )
    h = HeadA(checkpoint=str(tmp_path / "c.pt"), device="cpu", budget_ms=5000)
    h.warmup()
    s = h.score(_window(6), CTX)
    assert not s.abstain and 0 <= (s.p_spoof or 0) <= 1
    assert s.calibration_version == "cal-2026-09-17" and s.evidence["untrained"] is False
