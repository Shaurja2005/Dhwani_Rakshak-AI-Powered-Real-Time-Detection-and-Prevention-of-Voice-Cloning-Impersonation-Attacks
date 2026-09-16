"""Cosine scoring with adaptive symmetric normalisation (AS-norm) (B7-T04).

Raw cosine similarity drifts with channel, language and speaker; AS-norm
normalises each trial by the statistics of the top-K most similar *cohort*
(impostor) scores of both sides:

    s_norm = ½ [ (s − μ_e) / σ_e + (s − μ_t) / σ_t ]

where μ_e, σ_e come from the enrolment embedding vs cohort and μ_t, σ_t from
the test embedding vs cohort. Thresholds on s_norm are far more stable across
channels than thresholds on raw cosine.

The mismatch probability is a logistic calibration of −s_norm, fitted on
target / non-target trials by ``fit_calibration``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def _top_k_stats(emb: np.ndarray, cohort: np.ndarray, k: int) -> tuple[float, float]:
    scores = cohort @ (emb / (np.linalg.norm(emb) + 1e-9))
    top = np.sort(scores)[-min(k, len(scores)) :]
    return float(top.mean()), float(max(top.std(), 1e-3))


def as_norm(
    enroll: np.ndarray,
    test: np.ndarray,
    cohort: np.ndarray | None,
    k: int = 200,
    center: bool = True,
) -> tuple[float, float]:
    """Return (cosine, AS-norm score).

    With a cohort, both embeddings (and the cohort) are first centred on the
    cohort mean — removing what all voices share — before cosine and AS-norm.
    Without a cohort (< 10 embeddings) the normalised score is the raw cosine.
    """
    if cohort is None or len(cohort) < 10:
        raw = cosine(enroll, test)
        return raw, raw
    ce = cohort / (np.linalg.norm(cohort, axis=1, keepdims=True) + 1e-9)
    if center:
        mu = ce.mean(axis=0)
        enroll, test = enroll - mu, test - mu
        ce = ce - mu
        ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    raw = cosine(enroll, test)
    mu_e, sd_e = _top_k_stats(enroll, ce, k)
    mu_t, sd_t = _top_k_stats(test, ce, k)
    return raw, 0.5 * ((raw - mu_e) / sd_e + (raw - mu_t) / sd_t)


@dataclass
class Calibration:
    """p_mismatch = sigmoid(-(scale * score + bias)). Defaults are placeholders until fitted."""

    scale: float = 1.0
    bias: float = -2.0
    version: str = "uncalibrated"
    uses_asnorm: bool = True

    def p_mismatch(self, score: float) -> float:
        z = max(min(self.scale * score + self.bias, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(z))


def fit_calibration(
    target: np.ndarray, nontarget: np.ndarray, steps: int = 2000, version: str = "cal-fit"
) -> Calibration:
    s = np.concatenate([target, nontarget])
    mismatch = np.concatenate([np.zeros(len(target)), np.ones(len(nontarget))])
    mu, sd = s.mean(), s.std() + 1e-9
    z = (s - mu) / sd
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(np.clip(a * z + b, -30, 30)))  # p_mismatch
        g = mismatch - p  # dNLL/d(a*z+b) for p = sigmoid(-(a*z+b))
        a -= 0.5 * float(np.mean(g * z))
        b -= 0.5 * float(np.mean(g))
    return Calibration(scale=a / sd, bias=b - a * mu / sd, version=version)


def eer_with_ci(
    target: np.ndarray, nontarget: np.ndarray, n_boot: int = 500, seed: int = 0
) -> dict[str, float]:
    """EER plus a 95% bootstrap CI — used for the B7 Definition of Done report."""

    def eer(t: np.ndarray, n: np.ndarray) -> float:
        thr = np.unique(np.concatenate([t, n]))
        frr = np.searchsorted(np.sort(t), thr, side="left") / len(t)
        far = 1 - np.searchsorted(np.sort(n), thr, side="left") / len(n)
        i = int(np.argmin(np.abs(frr - far)))
        return float((frr[i] + far[i]) / 2)

    rng = np.random.default_rng(seed)
    base = eer(target, nontarget)
    boots = [
        eer(rng.choice(target, len(target)), rng.choice(nontarget, len(nontarget)))
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"eer": base, "ci95_low": float(lo), "ci95_high": float(hi)}


def gap_with_ci(
    genuine: np.ndarray, clone: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Mean score gap (genuine − clone) with a 95% bootstrap CI (B7 DoD)."""
    rng = np.random.default_rng(seed)
    boots = [
        rng.choice(genuine, len(genuine)).mean() - rng.choice(clone, len(clone)).mean()
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "gap": float(genuine.mean() - clone.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }
