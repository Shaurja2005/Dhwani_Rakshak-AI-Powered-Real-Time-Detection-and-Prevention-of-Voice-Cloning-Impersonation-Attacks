"""States (B9-T05) and cost-model operating points (B9-T06).

Thresholds are per-tenant data, never constants in code. A profile is chosen
from score distributions on a deployment-like set at **fixed false-positive
rates** (default ELEVATED at 1 % FPR, HIGH at 0.1 % FPR), or by minimising an
explicit expected cost — never at EER.

Explicit ABSTAIN rules: not enough scored evidence yet (fewer than
``min_scored_windows``) or too much of the call was unscoreable
(``abstain_ratio > max_abstain_ratio``). ABSTAIN is never shown as safe.

Hysteresis prevents state flicker: a state is entered at its threshold and left
only after the score falls ``exit_margin`` below it for ``min_dwell_windows``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from packages.vg_core.models import RiskState

_ORDER = {RiskState.LOW: 0, RiskState.ELEVATED: 1, RiskState.HIGH: 2}


@dataclass
class OperatingProfile:
    name: str = "bfsi-default-v1"
    elevated: float = 0.40  # session probability thresholds (risk_score = round(100·p))
    high: float = 0.70
    exit_margin: float = 0.05
    min_dwell_windows: int = 3
    min_scored_windows: int = 2
    max_abstain_ratio: float = 0.7
    derived_from: str = "SOURCE_OF_TRUTH §9 defaults"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> OperatingProfile:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def threshold_at_fpr(bona_fide_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on bona fide scores is <= target."""
    s = np.sort(np.asarray(bona_fide_scores, dtype=float))
    if len(s) == 0:
        raise ValueError("need bona fide scores")
    k = int(np.ceil((1 - target_fpr) * len(s)))
    return float(s[min(k, len(s) - 1)] + 1e-9) if k < len(s) else float(s[-1] + 1e-9)


def expected_cost(
    threshold: float,
    bona: np.ndarray,
    spoof: np.ndarray,
    c_fp: float,
    c_fn: float,
    prior_spoof: float,
) -> float:
    fpr = float((np.asarray(bona) >= threshold).mean())
    fnr = float((np.asarray(spoof) < threshold).mean())
    return c_fp * fpr * (1 - prior_spoof) + c_fn * fnr * prior_spoof


def cost_optimal_threshold(
    bona: np.ndarray, spoof: np.ndarray, c_fp: float, c_fn: float, prior_spoof: float
) -> float:
    grid = np.unique(np.concatenate([bona, spoof, [0.0, 1.0]]))
    costs = [expected_cost(t, bona, spoof, c_fp, c_fn, prior_spoof) for t in grid]
    return float(grid[int(np.argmin(costs))])


def profile_from_scores(
    name: str,
    bona: np.ndarray,
    spoof: np.ndarray,
    fpr_elevated: float = 0.01,
    fpr_high: float = 0.001,
    calls_per_day: int = 50_000,
) -> tuple[OperatingProfile, dict[str, float]]:
    elevated = threshold_at_fpr(bona, fpr_elevated)
    high = max(threshold_at_fpr(bona, fpr_high), elevated + 1e-6)
    report = {
        "elevated_threshold": elevated,
        "high_threshold": high,
        "fpr_elevated": float((bona >= elevated).mean()),
        "fpr_high": float((bona >= high).mean()),
        "tpr_elevated": float((spoof >= elevated).mean()) if len(spoof) else float("nan"),
        "tpr_high": float((spoof >= high).mean()) if len(spoof) else float("nan"),
        "wrongly_flagged_per_day_elevated": float((bona >= elevated).mean()) * calls_per_day,
    }
    prof = OperatingProfile(
        name=name,
        elevated=min(elevated, 0.999),
        high=min(high, 0.9999),
        derived_from=f"fixed FPR {fpr_elevated:.3%} / {fpr_high:.3%} on {len(bona)} bona fide",
    )
    return prof, report


class StateMachine:
    def __init__(self, profile: OperatingProfile) -> None:
        self.p = profile
        self.state: RiskState = RiskState.ABSTAIN
        self._below = 0

    def _raw(self, p: float) -> RiskState:
        return (
            RiskState.HIGH
            if p >= self.p.high
            else RiskState.ELEVATED if p >= self.p.elevated else RiskState.LOW
        )

    def update(self, p_session: float, n_scored: int, abstain_ratio: float) -> RiskState:
        if n_scored < self.p.min_scored_windows or abstain_ratio > self.p.max_abstain_ratio:
            self.state, self._below = RiskState.ABSTAIN, 0
            return self.state
        target = self._raw(p_session)
        if self.state == RiskState.ABSTAIN or _ORDER[target] > _ORDER[self.state]:
            self.state, self._below = target, 0  # escalate immediately
            return self.state
        if _ORDER[target] == _ORDER[self.state]:
            self._below = 0
            return self.state
        # De-escalation: require the score to clear the exit margin for min_dwell windows.
        current_threshold = self.p.high if self.state == RiskState.HIGH else self.p.elevated
        if p_session < current_threshold - self.p.exit_margin:
            self._below += 1
            if self._below >= self.p.min_dwell_windows:
                self.state, self._below = self._raw(p_session + self.p.exit_margin), 0
        else:
            self._below = 0
        return self.state
