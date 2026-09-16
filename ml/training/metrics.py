"""Training-time dev metrics. Official reported numbers come from B15 (ml/eval), never from here."""

from __future__ import annotations

import numpy as np


def compute_eer(bona_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """EER and its threshold; higher score = more bona fide."""
    scores = np.concatenate([bona_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(bona_scores)), np.zeros(len(spoof_scores))])
    order = np.argsort(scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    n_bona, n_spoof = len(bona_scores), len(spoof_scores)
    frr = np.concatenate([[0.0], np.cumsum(labels) / n_bona])  # bona rejected at threshold
    far = np.concatenate([[1.0], 1 - np.cumsum(1 - labels) / n_spoof])  # spoof accepted
    thresholds = np.concatenate([[scores[0] - 1e-6], scores])
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(thresholds[i])


def fit_platt(scores: np.ndarray, is_spoof: np.ndarray, steps: int = 500) -> tuple[float, float]:
    """Fit p_spoof = sigmoid(-(scale*score + bias)) by logistic regression."""
    a, b = 1.0, 0.0
    for _ in range(steps):
        z = np.clip(a * scores + b, -30, 30)
        p = 1 / (1 + np.exp(z))  # p_spoof
        g = p - is_spoof  # -dNLL/dz; ascend it
        a += 0.1 * float(np.mean(g * scores))
        b += 0.1 * float(np.mean(g))
    return a, b
