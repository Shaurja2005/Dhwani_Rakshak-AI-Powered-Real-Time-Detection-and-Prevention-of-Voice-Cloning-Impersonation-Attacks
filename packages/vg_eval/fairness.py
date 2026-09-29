"""vg_eval.fairness — per-group false-positive gaps and the release gate (B15-T05).

A false positive here is a **real customer flagged as synthetic**. The gap is
measured at one global operating threshold (the one the system would ship
with), never per-group thresholds, because that is what customers experience.

The release gate needs a maximum allowed gap. That number is a policy decision
(PROJECT_STATUS Q9), not something the harness invents: until it is configured
the gate reports ``not_configured`` rather than pass/fail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from packages.vg_eval import metrics as M


@dataclass
class GroupFPR:
    attribute: str
    group: str
    n_bona: int
    fpr: float
    ci: tuple[float, float]


@dataclass
class FairnessResult:
    attribute: str
    threshold: float
    groups: list[GroupFPR] = field(default_factory=list)

    @property
    def gap(self) -> float:
        vals = [g.fpr for g in self.groups if g.n_bona > 0]
        return float(max(vals) - min(vals)) if len(vals) > 1 else 0.0

    @property
    def ratio(self) -> float:
        vals = [g.fpr for g in self.groups if g.n_bona > 0]
        if len(vals) < 2 or min(vals) == 0:
            return float("inf") if vals and max(vals) > 0 else 1.0
        return float(max(vals) / min(vals))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def operating_threshold(
    df: pd.DataFrame, mode: str = "eer", target_fpr: float | None = None
) -> float:
    """Global threshold: at the pooled EER, or at a configured bona fide FPR."""
    s = df["score"].to_numpy(dtype=float)
    lab = df["label"].to_numpy()
    bona, spoof = s[lab == "bona_fide"], s[lab == "spoof"]
    if mode == "fpr":
        if target_fpr is None:
            raise ValueError("mode='fpr' needs target_fpr from the eval config")
        return M.threshold_at_fpr(bona, target_fpr)
    return M.eer(bona, spoof)[1]


def fpr_by(df: pd.DataFrame, attribute: str, threshold: float) -> FairnessResult:
    res = FairnessResult(attribute=attribute, threshold=threshold)
    if attribute not in df:
        return res
    bona = df[(df["label"] == "bona_fide") & np.isfinite(df["score"].astype(float))]
    for group, g in bona.groupby(bona[attribute].fillna("unknown"), sort=True):
        flagged = int((g["score"].astype(float) <= threshold).sum())
        n = len(g)
        res.groups.append(
            GroupFPR(
                attribute, str(group), n, flagged / n if n else float("nan"), _wilson(flagged, n)
            )
        )
    return res


@dataclass
class GateDecision:
    status: str  # pass | fail | not_configured | insufficient_data
    detail: str


def fairness_gate(
    results: Sequence[FairnessResult], max_gap: float | None, min_group_n: int = 30
) -> GateDecision:
    if max_gap is None:
        return GateDecision("not_configured", "max FPR gap not set (PROJECT_STATUS Q9)")
    small = [
        f"{r.attribute}={g.group}" for r in results for g in r.groups if g.n_bona < min_group_n
    ]
    failing = [
        f"{r.attribute}: gap {r.gap:.3f} > {max_gap:.3f}" for r in results if r.gap > max_gap
    ]
    if failing:
        return GateDecision("fail", "; ".join(failing))
    if small:
        return GateDecision("insufficient_data", "groups under min n: " + ", ".join(small[:10]))
    return GateDecision("pass", f"all FPR gaps <= {max_gap:.3f}")
