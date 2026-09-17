"""Cross-head fusion per window (B9-T02) with contribution attribution (B9-T07).

Logistic regression over calibrated per-head log-odds:

    z = bias + Σ_h present_h · w_h · logit(p_h) + Σ_h (1 − present_h) · m_h

* ``present_h`` is 0 when head h abstained — its logit term is *dropped*, never
  replaced by logit(0) or logit(0.5). The learned missing-indicator weight
  ``m_h`` lets the model account for what an absent head means (e.g. Head D
  absent because the caller is not enrolled). Default m_h = 0.
* Head F passes through the watermark asymmetry filter first (I4, ADR 0007),
  so an absent watermark cannot contribute anything, including via m_F
  (m_F is pinned to 0).
* Defaults before fitting (``FusionModel.prior()``): w_h = 0.5 for every head,
  a conservative partial-independence log-odds sum. Fit with ``fit_fusion`` on
  held-out per-window head scores.

Contributions: |w_h · logit(p_h)| normalised to sum to 1 over present heads
(FusedWindowScore contract); signed values are also returned for drivers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from packages.vg_core.models import HeadID, HeadScore
from packages.vg_models.calibration import CalibrationStore, logit, sigmoid
from packages.vg_models.heads.head_f_watermark.asymmetry import fusion_inputs

HEADS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class FusionModel:
    weights: dict[str, float] = field(default_factory=lambda: {h: 0.5 for h in HEADS})
    missing: dict[str, float] = field(default_factory=lambda: {h: 0.0 for h in HEADS})
    bias: float = 0.0
    version: str = "fuse-lr-prior-v0.1"

    @classmethod
    def prior(cls) -> FusionModel:
        return cls()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FusionModel:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class WindowFusion:
    p_spoof: float | None  # None when every head abstained
    logit: float
    contributions: dict[str, float]  # normalised |share|, sums to 1 over present heads
    signed: dict[str, float]  # signed log-odds contribution per present head
    present: list[str]
    abstained: list[str]
    calibration_versions: dict[str, str]
    model_versions: dict[str, str]


def head_matrix(
    scores: list[HeadScore], calibration: CalibrationStore | None = None
) -> tuple[dict[str, float], dict[str, str], dict[str, str], list[str]]:
    """Calibrated logits of usable heads, calibration + model versions, abstained head ids."""
    cal = calibration or CalibrationStore()
    usable = {s.head_id.value for s in fusion_inputs(scores)}
    logits: dict[str, float] = {}
    cal_versions: dict[str, str] = {}
    model_versions: dict[str, str] = {}
    abstained: list[str] = []
    for s in scores:
        h = s.head_id.value
        model_versions[h] = s.model_version
        p, cv = cal.calibrated_p(s)
        if h not in usable or p is None:
            abstained.append(h)
            continue
        logits[h] = float(logit(p))
        cal_versions[h] = cv
    return logits, cal_versions, model_versions, sorted(set(abstained))


def fuse(
    scores: list[HeadScore], model: FusionModel, calibration: CalibrationStore | None = None
) -> WindowFusion:
    logits, cal_v, model_v, abstained = head_matrix(scores, calibration)
    if not logits:
        return WindowFusion(None, 0.0, {}, {}, [], abstained, cal_v, model_v)
    signed = {h: model.weights.get(h, 0.0) * lg for h, lg in logits.items()}
    z = model.bias + sum(signed.values())
    for h in HEADS:
        if h not in logits and h != HeadID.F.value:
            z += model.missing.get(h, 0.0)
    total = sum(abs(v) for v in signed.values())
    contrib = {h: (abs(v) / total if total > 0 else 1.0 / len(signed)) for h, v in signed.items()}
    return WindowFusion(
        float(sigmoid(z)), float(z), contrib, signed, sorted(logits), abstained, cal_v, model_v
    )


def design(rows: list[dict[str, float]]) -> np.ndarray:
    """Rows of {head: calibrated logit} (absent key = abstained) -> [N, 2·H] features."""
    x = np.zeros((len(rows), 2 * len(HEADS)))
    for i, r in enumerate(rows):
        for j, h in enumerate(HEADS):
            if h in r:
                x[i, j] = r[h]
            elif h != "F":
                x[i, len(HEADS) + j] = 1.0
    return x


def fit_fusion(
    rows: list[dict[str, float]],
    is_spoof: np.ndarray,
    l2: float = 1e-3,
    steps: int = 4000,
    version: str = "fuse-lr-v0.1",
) -> FusionModel:
    x = design(rows)
    y = np.asarray(is_spoof, dtype=float)
    w = np.zeros(x.shape[1])
    b = float(logit(np.clip(y.mean(), 0.01, 0.99)))
    lr = 0.1
    for _ in range(steps):
        p = sigmoid(x @ w + b)
        g = p - y
        w -= lr * (x.T @ g / len(y) + l2 * w)
        b -= lr * float(g.mean())
    n = len(HEADS)
    return FusionModel(
        weights={h: float(w[j]) for j, h in enumerate(HEADS)},
        missing={h: (0.0 if h == "F" else float(w[n + j])) for j, h in enumerate(HEADS)},
        bias=b,
        version=version,
    )
