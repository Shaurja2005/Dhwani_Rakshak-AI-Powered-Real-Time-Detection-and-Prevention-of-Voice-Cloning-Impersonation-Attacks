"""B9 — calibration, fusion, temporal risk engine, operating points, persistence, ablation."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.vg_core.models import HeadID, HeadScore, RiskState
from packages.vg_models.calibration import (
    CalibrationStore,
    Calibrator,
    expected_calibration_error,
    logit,
    sigmoid,
)
from services.fusion.ablation import ablation_table
from services.fusion.engine import RiskEngine
from services.fusion.fuser import FusionModel, fit_fusion, fuse, head_matrix
from services.fusion.operating_point import (
    OperatingProfile,
    StateMachine,
    cost_optimal_threshold,
    profile_from_scores,
    threshold_at_fpr,
)
from services.fusion.persistence import SQLiteTimelineStore, create_router
from services.fusion.temporal import TemporalState, state_changes_per_minute

SID = "01900b1a-0000-7000-8000-00000000f9f1"


def hs(
    head: str,
    wid: int,
    p: float | None,
    raw: float = 0.0,
    version: str | None = None,
    evidence: dict[str, object] | None = None,
) -> HeadScore:
    return HeadScore(
        session_id=SID,
        window_id=wid,
        head_id=HeadID(head),
        raw_score=raw,
        p_spoof=p,
        abstain=p is None,
        confidence=0.5,
        latency_ms=1,
        model_version=version or f"{head}@x-y-v0.1.0",
        calibration_version="cal-2026-09-17",
        evidence=evidence or {},
    )


class CallSimulator:
    """Per-window calibrated head probabilities with head-specific separation and abstain rates."""

    SEP = {"A": 3.0, "B": 1.0, "C": 0.6}
    ABSTAIN = {"A": 0.05, "B": 0.2, "C": 0.3}

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def window(self, wid: int, synthetic: bool) -> list[HeadScore]:
        out = []
        for h, sep in self.SEP.items():
            if self.rng.random() < self.ABSTAIN[h]:
                out.append(hs(h, wid, None))
                continue
            z = (sep if synthetic else -sep) + self.rng.normal(0, 1.5)
            out.append(hs(h, wid, float(sigmoid(z))))
        return out

    def run(
        self, pattern: list[bool], engine: RiskEngine | None = None
    ) -> tuple[RiskEngine, list[str], list[str]]:
        e = engine or RiskEngine(SID)
        states, naive = [], []
        for w, syn in enumerate(pattern):
            fused, risk = e.update(w, self.window(w, syn))
            states.append(risk.state.value)
            naive.append(fused.state.value)
        return e, states, naive


# ---------------------------------------------------------------- calibration (T01)
@pytest.mark.parametrize("method", ["platt", "temperature"])
def test_calibration_fit_reaches_low_ece(method: str) -> None:
    rng = np.random.default_rng(0)
    # Temperature scaling has no bias term, so it can only fix scale, not a class-prior shift.
    y = rng.random(6000) < (0.3 if method == "platt" else 0.5)
    true_logit = np.where(y, 1.5, -1.5) + rng.normal(0, 1.2, len(y))
    raw = 4.0 * true_logit  # overconfident raw scale
    split = 4000
    cal = Calibrator.fit("A", "A@x-y-v0.1.0", raw[:split], y[:split], method=method)  # type: ignore[arg-type]
    before = expected_calibration_error(sigmoid(raw[split:]), y[split:])
    after = expected_calibration_error(cal.transform(raw[split:]), y[split:])
    assert after < 0.05 < before  # B9 DoD target


def test_calibration_store_is_keyed_by_model_version(tmp_path: object) -> None:
    store = CalibrationStore(f"{tmp_path}/cal.json")
    store.put(Calibrator("A", "A@x-y-v0.1.0", "platt", a=2.0, b=-1.0, version="cal-1"))
    reloaded = CalibrationStore(f"{tmp_path}/cal.json")
    p, v = reloaded.calibrated_p(hs("A", 0, 0.9, raw=1.0))
    assert v == "cal-1" and p == pytest.approx(float(sigmoid(1.0)))
    p2, v2 = reloaded.calibrated_p(hs("A", 0, 0.9, raw=1.0, version="A@x-y-v0.2.0"))
    assert (p2, v2) == (0.9, "cal-2026-09-17")  # new model version: no stale calibration
    assert reloaded.calibrated_p(hs("A", 0, None)) == (None, "cal-2026-09-17")


# ---------------------------------------------------------------- fusion (T02, T07)
def test_abstained_head_is_dropped_not_zero_filled() -> None:
    model = FusionModel(weights={h: 1.0 for h in "ABCDEF"}, missing={h: 0.0 for h in "ABCDEF"})
    with_b_abstain = fuse([hs("A", 0, 0.8), hs("B", 0, None)], model)
    a_only = fuse([hs("A", 0, 0.8)], model)
    assert with_b_abstain.p_spoof == pytest.approx(a_only.p_spoof)  # abstain != p=0 or p=0.5
    assert with_b_abstain.abstained == ["B"] and with_b_abstain.present == ["A"]
    assert fuse([hs("A", 0, None)], model).p_spoof is None


def test_missing_indicator_weight_is_learned_and_applied() -> None:
    model = FusionModel(
        weights={h: 1.0 for h in "ABCDEF"}, missing={**{h: 0.0 for h in "ABCDEF"}, "D": -1.0}
    )
    no_d = fuse([hs("A", 0, 0.5)], model)
    assert no_d.logit == pytest.approx(-1.0)


def test_contributions_sum_to_one_and_track_evidence() -> None:
    wf = fuse([hs("A", 0, 0.95), hs("B", 0, 0.55), hs("C", 0, 0.2)], FusionModel.prior())
    assert sum(wf.contributions.values()) == pytest.approx(1.0)
    assert max(wf.contributions, key=wf.contributions.get) == "A"  # type: ignore[arg-type]
    assert wf.signed["A"] > 0 > wf.signed["C"]


def test_fitted_fusion_weights_informative_head_more() -> None:
    sim = CallSimulator(1)
    rows, ys = [], []
    for w in range(3000):
        syn = bool(w % 2)
        logits, *_ = head_matrix(sim.window(w, syn))
        rows.append(logits)
        ys.append(float(syn))
    m = fit_fusion(rows, np.array(ys))
    assert m.weights["A"] > m.weights["B"] > 0 and m.weights["A"] > m.weights["C"]


def test_absent_watermark_changes_fused_score_by_exactly_zero() -> None:
    base = [hs("A", 0, 0.7), hs("B", 0, 0.4)]
    absent = hs("F", 0, None, evidence={"watermark": "absent", "neutral": True})
    present = hs("F", 0, 0.99, evidence={"watermark": "present"})
    rogue = hs(
        "F", 0, 0.01, evidence={"watermark": "absent"}
    )  # a misbehaving F trying to exonerate
    model = FusionModel.prior()
    assert fuse(base + [absent], model).p_spoof == fuse(base, model).p_spoof
    assert fuse(base + [rogue], model).p_spoof == fuse(base, model).p_spoof
    assert fuse(base + [present], model).p_spoof > fuse(base, model).p_spoof  # type: ignore[operator]


# ---------------------------------------------------------------- temporal + any-segment (T03, T04)
def test_hmm_firms_up_and_ignores_abstained_windows() -> None:
    t = TemporalState()
    posts = []
    for w in range(8):
        t.update(w, 0.8)
        posts.append(t.p_hmm)
    assert all(b > a for a, b in zip(posts, posts[1:], strict=False)) and posts[-1] > 0.9
    before = t.log_odds
    t.update(99, None)
    assert t.n_abstained == 1 and abs(t.log_odds - before) < 0.5  # transition only, no evidence


def test_engine_rates_over_simulated_calls() -> None:
    genuine_false_high = synthetic_high = splice_flagged = 0
    changes_engine, changes_naive = [], []
    n = 12
    for seed in range(n):
        _, st, naive = CallSimulator(100 + seed).run([False] * 60)
        genuine_false_high += st[-1] in ("HIGH", "ELEVATED")
        changes_engine.append(state_changes_per_minute(st))
        changes_naive.append(state_changes_per_minute(naive))
        _, st, _ = CallSimulator(200 + seed).run([True] * 60)
        synthetic_high += "HIGH" in st[:6]
        e, st, _ = CallSimulator(300 + seed).run([False] * 30 + [True] * 4 + [False] * 26)
        splice_flagged += st[-1] in ("HIGH", "ELEVATED")
    assert genuine_false_high <= 1
    assert synthetic_high == n  # HIGH within the first 6 windows of every synthetic call
    assert splice_flagged >= n - 2  # a 4-window splice inside a genuine call is still caught
    assert np.mean(changes_engine) * 4 < np.mean(
        changes_naive
    )  # no flicker vs per-window thresholds
    assert max(changes_engine) <= 4.0


def test_session_risk_exposes_max_and_mean_and_drivers() -> None:
    e, _, _ = CallSimulator(7).run([False] * 30 + [True] * 4 + [False] * 26)
    r = e.session_risk()
    assert r.p_spoof_session_max > 0.5 > r.p_spoof_session_mean
    assert any(d.factor == "any_segment_trigger" for d in r.drivers)
    assert r.model_versions["fusion"] == "fuse-lr-prior-v0.1" and "A" in r.model_versions
    assert 0 <= r.abstain_ratio <= 1 and len(r.timeline) == 60


# ------------------------------------------------ states + operating points (T05, T06)
def test_abstain_state_rules() -> None:
    e = RiskEngine(SID)
    _, risk = e.update(0, [hs("A", 0, 0.99)])
    assert risk.state == RiskState.ABSTAIN  # not enough scored evidence yet
    e2 = RiskEngine(SID)
    for w in range(10):
        _, risk = e2.update(w, [hs("A", w, None), hs("B", w, None)])
    assert risk.state == RiskState.ABSTAIN and risk.abstain_ratio == 1.0


def test_hysteresis_escalates_fast_and_deescalates_slowly() -> None:
    sm = StateMachine(
        OperatingProfile(elevated=0.4, high=0.7, exit_margin=0.05, min_dwell_windows=3)
    )
    assert sm.update(0.75, 5, 0.0) == RiskState.HIGH
    assert sm.update(0.68, 5, 0.0) == RiskState.HIGH  # inside exit margin
    assert [sm.update(0.5, 5, 0.0) for _ in range(3)] == [
        RiskState.HIGH,
        RiskState.HIGH,
        RiskState.ELEVATED,
    ]


def test_thresholds_from_fixed_fpr_and_cost_model() -> None:
    rng = np.random.default_rng(3)
    bona = sigmoid(rng.normal(-2.5, 1.0, 20000))
    spoof = sigmoid(rng.normal(2.0, 1.2, 2000))
    t1 = threshold_at_fpr(bona, 0.01)
    assert (bona >= t1).mean() <= 0.01 and (bona >= t1).mean() > 0.008
    prof, report = profile_from_scores("tenant-x", bona, spoof)
    assert prof.high > prof.elevated and report["fpr_high"] <= 0.001
    assert report["wrongly_flagged_per_day_elevated"] <= 500
    cheap_fp = cost_optimal_threshold(bona, spoof, c_fp=1, c_fn=100, prior_spoof=0.01)
    costly_fp = cost_optimal_threshold(bona, spoof, c_fp=100, c_fn=1, prior_spoof=0.01)
    assert cheap_fp < costly_fp


def test_profile_roundtrip(tmp_path: object) -> None:
    p = OperatingProfile(name="wealth", elevated=0.3, high=0.6)
    p.save(f"{tmp_path}/p.json")
    assert OperatingProfile.load(f"{tmp_path}/p.json") == p


# ---------------------------------------------------------------- persistence + replay (T08)
def test_timeline_persistence_and_replay_api() -> None:
    store = SQLiteTimelineStore()
    e, states, _ = CallSimulator(9).run([True] * 10, RiskEngine(SID, store=store))
    rows = store.session(SID)
    assert [r.window_id for r in rows] == list(range(10)) and rows[-1].state == states[-1]
    assert (
        isinstance(rows[0].contributions, dict) and rows[-1].fusion_version == "fuse-lr-prior-v0.1"
    )
    app = FastAPI()
    app.include_router(create_router(store))
    c = TestClient(app)
    body = c.get(f"/v1/sessions/{SID}/timeline").json()
    assert len(body["windows"]) == 10 and body["windows"][-1]["risk_score"] == rows[-1].risk_score
    assert c.get("/v1/sessions/unknown/timeline").status_code == 404


# ---------------------------------------------------------------- ablation (DoD)
def test_ablation_table_generated_and_ranks_strongest_head() -> None:
    sim = CallSimulator(11)

    def data(n: int, offset: int) -> tuple[list[dict[str, float]], np.ndarray]:
        rows, ys = [], []
        for w in range(n):
            syn = bool(w % 2)
            logits, *_ = head_matrix(sim.window(offset + w, syn))
            rows.append(logits)
            ys.append(float(syn))
        return rows, np.array(ys)

    fr, fy = data(3000, 0)
    tr, ty = data(1500, 10_000)
    results, table = ablation_table(fr, fy, tr, ty)
    by = {r["heads"]: r for r in results}
    assert by["all"]["ece"] < 0.05  # type: ignore[operator]
    assert by["without A"]["delta_eer_vs_all"] > by["without B"]["delta_eer_vs_all"] > -0.01  # type: ignore[operator]
    assert table.startswith("| Heads | EER | ECE |") and "without C" in table
    assert logit(0.5) == pytest.approx(0.0)


def test_session_statistics_are_consistent_on_genuine_calls() -> None:
    for seed in range(5):
        e, _, _ = CallSimulator(500 + seed).run([False] * 40)
        r = e.session_risk()
        assert r.p_spoof_session_max >= r.p_spoof_session_mean
        assert not any(d.factor == "any_segment_trigger" for d in r.drivers)
