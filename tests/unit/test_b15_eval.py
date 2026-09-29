"""B15 — metrics, protocols, fairness gate, report ids, attacks, harness CLI, model registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ml.eval import adversarial as adv
from packages.vg_eval import fairness as F
from packages.vg_eval import metrics as M
from packages.vg_eval import protocols as P
from packages.vg_eval.report import RunRecord, plot_before_after, rebuild, render, row_id

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- T09 metrics
def test_eer_extremes_and_known_value() -> None:
    assert M.eer(np.array([2.0, 3.0]), np.array([0.0, 1.0]))[0] == 0.0
    b, s = RNG.normal(0, 1, 4000), RNG.normal(0, 1, 4000)
    assert abs(M.eer(b, s)[0] - 0.5) < 0.03
    # two unit Gaussians 2 sigma apart: EER = Phi(-1) = 0.1587
    b, s = RNG.normal(1, 1, 20000), RNG.normal(-1, 1, 20000)
    assert M.eer(b, s)[0] == pytest.approx(0.1587, abs=0.01)
    with pytest.raises(ValueError):
        M.eer(np.array([]), np.array([1.0]))
    assert M.eer(np.array([1.0, np.nan, 2.0]), np.array([0.0]))[0] == 0.0  # NaN = abstain


def test_dcf_family() -> None:
    good_b, good_s = RNG.normal(3, 1, 2000), RNG.normal(-3, 1, 2000)
    rand_b, rand_s = RNG.normal(0, 1, 2000), RNG.normal(0, 1, 2000)
    assert M.min_dcf(good_b, good_s) < 0.1
    assert 0.9 < M.min_dcf(rand_b, rand_s) <= 1.0 + 1e-9
    t = M.min_tdcf(good_b, good_s, asv_p_miss=0.02, asv_p_fa=0.02, asv_p_miss_spoof=0.3)
    assert 0 <= t < 0.2
    with pytest.raises(ValueError):
        M.min_tdcf(good_b, good_s, asv_p_miss=1.0, asv_p_fa=1.0, asv_p_miss_spoof=0.0)
    tar, non, spf = RNG.normal(4, 1, 500), RNG.normal(0, 1, 500), RNG.normal(-4, 1, 500)
    assert M.min_adcf(tar, non, spf) < 0.1
    assert M.min_adcf(RNG.normal(0, 1, 500), non, RNG.normal(0, 1, 500)) > 0.8


def test_auc_pauc_ece_bootstrap() -> None:
    assert M.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert M.auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert M.pauc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    chance = M.pauc(RNG.normal(0, 1, 5000), RNG.normal(0, 1, 5000), 0.05)
    assert chance == pytest.approx(0.5, abs=0.05)
    p = RNG.uniform(0, 1, 20000)
    assert M.ece(p, RNG.uniform(0, 1, 20000) < p) < 0.02  # calibrated
    assert M.ece(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)
    bb, ss = RNG.normal(1, 1, 300), RNG.normal(-1, 1, 300)
    lo, hi = M.bootstrap_ci(lambda b, s: M.eer(b, s)[0], bb, ss, n=100)
    assert lo <= M.eer(bb, ss)[0] <= hi and hi - lo < 0.1
    thr = M.threshold_at_fpr(np.arange(100.0), 0.05)
    assert M.rates_at(np.arange(100.0), np.array([-1.0]), thr)[0] <= 0.05


# ---------------------------------------------------------------- protocols
def table(n: int = 400, shift_unseen: float = 1.5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        spoof = i % 2 == 1
        fam = ("f_seen" if i % 4 == 1 else "f_unseen") if spoof else None
        ds = "in" if i % 3 else "cross"
        if spoof:
            score = RNG.normal(-1.5 + (shift_unseen if fam == "f_unseen" else 0), 1)
        else:
            score = RNG.normal(1.5, 1)
        if ds == "cross":
            score += RNG.normal(0, 1.5)
        p = 1 / (1 + np.exp(score))
        rows.append(
            {
                "utt_id": f"u{i}",
                "label": "spoof" if spoof else "bona_fide",
                "score": score,
                "p_spoof": p,
                "attack_family": fam,
                "dataset": ds,
                "language": "ta" if i % 8 == 0 else "hi",
                "gender": ["female", "male"][(i // 2) % 2],
                "codec": "none",
                "snr_db": float(i % 30),
                "window_scores": [p] * 5,
            }
        )
    return pd.DataFrame(rows)


def test_logo_marks_unseen_and_pools() -> None:
    rows = {r.slice: r for r in P.logo(table(), ["f_seen"], bootstrap=0)}
    assert rows["f_seen"].extra["seen_in_training"]
    assert not rows["f_unseen"].extra["seen_in_training"]
    assert rows["ALL_UNSEEN"].eer > rows["ALL_SEEN"].eer  # unseen family is harder
    assert rows["f_seen"].n_bona == 200  # every family vs the full bona fide pool
    assert [f["held_out"] for f in P.logo_folds(["b", "a", "b"])] == ["a", "b"]


def test_cross_dataset_degradation_and_abstain_coverage() -> None:
    df = table()
    df.loc[df.index[:40], "score"] = np.nan  # abstentions
    rows = {r.slice: r for r in P.cross_dataset(df, ["in"], bootstrap=0)}
    assert rows["in"].extra["in_domain"] and rows["cross"].extra["eer_minus_in_domain"] > 0
    overall = P.summarize(df, "overall", "all", bootstrap=0)
    assert overall.coverage == pytest.approx(0.9) and overall.n_bona + overall.n_spoof == 360
    assert P.summarize(df[df.label == "spoof"], "x", "y", 0).eer is None  # one class only


def test_conditions_snr_bins_and_operational() -> None:
    df = table()
    cond = pd.concat(
        [
            df.assign(condition="clean"),
            df.assign(condition="noise", score=df["score"] * 0.3 + RNG.normal(0, 1, len(df))),
        ]
    )
    rows = {r.slice: r for r in P.per_condition(cond, "snr_sweep", bootstrap=0)}
    assert rows["noise"].extra["eer_minus_clean"] > 0
    assert "eer_minus_clean" not in rows["clean"].extra
    assert P.snr_bin(7.0) == "[5,10) dB" and P.snr_bin(None) == "clean/unknown"
    assert P.time_to_first_alert([0.1, 0.2, 0.9], 0.5) == 5.0
    assert P.time_to_first_alert([0.1, float("nan")], 0.5) is None
    assert P.score_stability([0.1, 0.9, 0.1, 0.9], 0.5)["flips_per_min"] == pytest.approx(45.0)
    ops = {r.slice: r for r in P.operational(df, 0.5)}
    assert ops["spoof"].extra["alerted_fraction"] > ops["bona_fide"].extra["alerted_fraction"]


# ---------------------------------------------------------------- T05 fairness
def test_fairness_gap_and_gate() -> None:
    df = table()
    thr = F.operating_threshold(df)
    res = F.fpr_by(df, "gender", thr)
    assert {g.group for g in res.groups} == {"female", "male"} and res.gap >= 0
    biased = df.copy()
    biased.loc[(biased.label == "bona_fide") & (biased.gender == "female"), "score"] -= 2.0
    rb = F.fpr_by(biased, "gender", thr)
    assert rb.gap > 0.2
    assert F.fairness_gate([rb], None).status == "not_configured"  # never invent the threshold
    assert F.fairness_gate([rb], 0.05).status == "fail"
    assert F.fairness_gate([res], 1.0, min_group_n=10_000).status == "insufficient_data"
    assert F.fairness_gate([res], 1.0).status == "pass"
    with pytest.raises(ValueError):
        F.operating_threshold(df, "fpr")


# ---------------------------------------------------------------- T10 report
def test_report_ids_are_stable_and_synthetic_is_refused(tmp_path: Path) -> None:
    rec = RunRecord("A@x-v1", "head_a", {"bootstrap": 0}, {"eval_sets": ["s"], "synthetic": False})
    rec.add_rows(P.logo(table(), ["f_seen"], bootstrap=0))
    df = table()
    rec.add_fairness(
        [F.fpr_by(df, "gender", F.operating_threshold(df))], F.GateDecision("pass", "ok")
    )
    same = RunRecord("A@x-v1", "head_a", {"bootstrap": 0}, {"eval_sets": ["s"], "synthetic": False})
    assert rec.run_id == same.run_id and rec.run_id.startswith("r")
    md = render([rec])
    rid = row_id(rec.run_id, "logo", "ALL_UNSEEN")
    assert f'<a id="{rid}">' in md and "Release gate: **pass**" in md
    rec.save(tmp_path)
    syn = RunRecord("A@syn", "x", {}, {"eval_sets": ["synthetic"], "synthetic": True})
    syn.save(tmp_path)
    out = rebuild(tmp_path, tmp_path / "REPORT.md", loadtest_dir=None).read_text(encoding="utf-8")
    assert "A@x-v1" in out and "A@syn" not in out
    assert "SYNTHETIC" in render([syn], allow_synthetic=True)
    assert "no detection evaluation on real data" in render([syn])
    loaded = RunRecord.load(tmp_path / "A_x-v1.json")
    assert loaded.run_id == rec.run_id and len(loaded.rows) == len(rec.rows)


def test_before_after_chart(tmp_path: Path) -> None:
    a = RunRecord("A@before", "x", {}, {"eval_sets": ["s"]})
    b = RunRecord("A@after", "x", {}, {"eval_sets": ["s"]})
    a.add_rows(P.per_language(table(), bootstrap=0))
    b.add_rows(P.per_language(table(shift_unseen=0.0), bootstrap=0))
    png = plot_before_after(a, b, tmp_path / "chart.png")
    assert png.exists() and png.stat().st_size > 5000


# ---------------------------------------------------------------- T08 attacks
def test_laundering_transforms() -> None:
    x = (0.2 * np.sin(2 * np.pi * 200 * np.arange(32000) / 16000)).astype(np.float32)
    rng = np.random.default_rng(0)
    assert len(adv.pitch_shift(2)(x, rng)) == len(x)
    assert len(adv.speed(1.1)(x, rng)) < len(x) < len(adv.speed(0.9)(x, rng))
    y = adv.codec_roundtrip("g711u")(x, rng)
    assert len(y) == len(x) and np.corrcoef(x, y)[0, 1] > 0.9
    assert adv.perturbation_snr(x, adv.add_noise(10)(x, rng)) == pytest.approx(10, abs=0.5)
    suite = adv.laundering_suite(["g711u"])
    assert "reencode_g711u" in suite and "speed_0.9" in suite


def test_white_box_attacks_raise_bona_fide_score() -> None:
    w = torch.linspace(-1, 1, 1600)

    def score_fn(x: torch.Tensor) -> torch.Tensor:  # differentiable toy detector
        return torch.tanh((x * w).sum(-1))

    x = 0.1 * torch.randn(4, 1600, generator=torch.Generator().manual_seed(0))
    x_adv = adv.pgd(score_fn, x, eps=0.01, steps=5)
    assert (score_fn(x_adv) > score_fn(x)).all()
    assert (x_adv - x).abs().max() <= 0.01 + 1e-6
    h = adv.learn_universal_filter(score_fn, x, taps=16, steps=30, lr=0.05)
    assert score_fn(adv.apply_filter(x, h)).mean() > score_fn(x).mean()


# ---------------------------------------------------------------- T01 harness end to end
def test_harness_cli_synthetic(tmp_path: Path) -> None:
    from ml.eval import run_eval
    from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel, save_checkpoint

    torch.manual_seed(0)
    ckpt = tmp_path / "a.pt"
    model = HeadAModel(HeadAConfig(frontend="tiny", frontend_layers=[3, 4], emb_dim=16))
    save_checkpoint(model, ckpt, {"lineage": "research", "train_families": ["family_0"]})
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(
        "synthetic_n: 48\nfairness: {attributes: [gender], max_gap: 0.5, min_group_n: 5}\n"
        "sweeps: {max_utts: 12, codecs: [g711u], snr_db: [5]}\n"
        "adversarial: {max_utts: 12, codecs: [g711u], pgd_steps: 2, filter_steps: 2}\n"
        "bootstrap: 5\n",
        encoding="utf-8",
    )
    out = tmp_path / "REPORT.md"
    args = ["--synthetic", "--config", str(cfg), "--out", str(out)]
    assert (
        run_eval.main(["--model", f"head_a:{ckpt}", *args, "--runs-dir", str(tmp_path / "r")]) == 0
    )
    md = out.read_text(encoding="utf-8")
    for section in (
        "Leave-one-generator-out",
        "Cross-dataset",
        "Per language",
        "Codec sweep",
        "SNR sweep",
        "Adversarial",
        "Operational",
        "Fairness",
    ):
        assert section in md, section
    assert "pgd_linf" in md and "universal_filter" in md and "clean_3s_crop" in md
    rec = RunRecord.load(next((tmp_path / "r").glob("*.json")))
    assert rec.data["seen_families"] == ["family_0"]  # read from the checkpoint meta
    refused = ["--model", "baseline:flatness", "--synthetic", "--config", str(cfg)]
    assert run_eval.main([*refused, "--out", str(run_eval.REPORT)]) == 2


# ---------------------------------------------------------------- B17-T07 registry (used by the harness)
def test_model_registry_promote_rollback(tmp_path: Path) -> None:
    from packages.vg_models.registry import ModelRegistry, RegistryError

    reg = ModelRegistry(tmp_path / "released.yaml")
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"model-a")
    b.write_bytes(b"model-b")
    reg.register("A@v1", "head_a", a, "research")
    reg.register("A@v2", "head_a", b, "research")
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("A@v1", "head_a", a, "research")
    with pytest.raises(RegistryError, match="no evaluation"):
        reg.promote("A@v1")
    reg.attach_eval("A@v1", "r1", "pass")
    reg.attach_eval("A@v2", "r2", "fail")
    with pytest.raises(RegistryError, match="commercially"):
        reg.promote("A@v1", commercial_deployment=True)
    reg.promote("A@v1")
    with pytest.raises(RegistryError, match="fairness"):
        reg.promote("A@v2")
    reg.attach_eval("A@v2", "r3", "pass")
    reg.promote("A@v2")
    assert ModelRegistry(tmp_path / "released.yaml").resolve("head_a") == str(b)  # persisted
    reg.rollback("head_a")
    assert reg.active("head_a").version == "A@v1"
    a.write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="checksum"):
        reg.resolve("head_a")
    assert [h["action"] for h in reg.history][-2:] == ["promote", "rollback"]


def test_serving_rows_from_load_tests() -> None:
    from packages.vg_eval.report import render_serving

    md = render_serving("docs/benchmarks/loadtest")
    assert "lt-cpu_i7-13620H_proxyL6_int8.8" in md  # the B14 headline row is citable
    assert render_serving("does/not/exist") == ""
