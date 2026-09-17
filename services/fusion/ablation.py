"""Head ablation table (B9 Definition of Done), generated automatically.

Input: per-window calibrated head logits (dicts, absent key = abstained) with
labels, split into fit / held-out. For the full head set and for each
leave-one-head-out set, fit the fusion model on the fit split and report on the
held-out split: EER, ECE, and the EER change versus the full model — each
head's marginal contribution.

Numbers from synthetic inputs demonstrate the machinery only; reportable
numbers come from B15 on real held-out data.
"""

from __future__ import annotations

import numpy as np

from packages.vg_models.calibration import expected_calibration_error, sigmoid
from services.fusion.fuser import HEADS, design, fit_fusion


def eer(p: np.ndarray, y: np.ndarray) -> float:
    thr = np.unique(p)
    bona, spoof = p[y == 0], p[y == 1]
    best = 1.0
    for t in thr:
        fpr = (bona >= t).mean()
        fnr = (spoof < t).mean()
        best = min(best, max(fpr, fnr))
    return float(best)


def _drop(rows: list[dict[str, float]], head: str | None) -> list[dict[str, float]]:
    return [{h: v for h, v in r.items() if h != head} for r in rows]


def ablation_table(
    fit_rows: list[dict[str, float]],
    fit_y: np.ndarray,
    test_rows: list[dict[str, float]],
    test_y: np.ndarray,
) -> tuple[list[dict[str, object]], str]:
    results: list[dict[str, object]] = []
    present = [h for h in HEADS if any(h in r for r in fit_rows)]
    base_eer = None
    for removed in [None, *present]:
        model = fit_fusion(_drop(fit_rows, removed), fit_y)
        x = design(_drop(test_rows, removed))
        w = np.array([model.weights[h] for h in HEADS] + [model.missing[h] for h in HEADS])
        p = sigmoid(x @ w + model.bias)
        e, c = eer(p, test_y), expected_calibration_error(p, test_y)
        if removed is None:
            base_eer = e
        results.append(
            {
                "heads": "all" if removed is None else f"without {removed}",
                "eer": e,
                "ece": c,
                "delta_eer_vs_all": 0.0 if removed is None else e - (base_eer or 0.0),
            }
        )
    lines = ["| Heads | EER | ECE | ΔEER vs all |", "|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['heads']} | {r['eer']:.3f} | {r['ece']:.3f} | {r['delta_eer_vs_all']:+.3f} |"
        )
    return results, "\n".join(lines)
