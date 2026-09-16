"""Symmetry check for invariant I3 (B3-T08).

Train a small classifier on *channel features only* (codec chain, SNR, RIR,
sample rate) and require that it cannot predict bona fide vs spoof better than
the majority-class baseline plus a sampling margin. If it can, augmentation is
asymmetric and the detector would learn the channel instead of the spoof.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ml.data.manifest import ManifestRow


@dataclass
class SymmetryResult:
    accuracy: float
    baseline: float
    margin: float
    n: int

    @property
    def passed(self) -> bool:
        return self.accuracy <= self.baseline + self.margin

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} acc={self.accuracy:.3f} baseline={self.baseline:.3f} "
            f"margin={self.margin:.3f} n={self.n}"
        )


def channel_features(rows: Sequence[ManifestRow]) -> np.ndarray:
    vocab = sorted(
        {tok for r in rows for tok in r.codec_chain} | {f"sr:{r.sample_rate}" for r in rows}
    )
    index = {t: i for i, t in enumerate(vocab)}
    x = np.zeros((len(rows), len(vocab) + 3), dtype=np.float64)
    for i, r in enumerate(rows):
        for tok in [*r.codec_chain, f"sr:{r.sample_rate}"]:
            x[i, index[tok]] = 1.0
        x[i, -3] = (r.snr_db if r.snr_db is not None else 40.0) / 40.0
        x[i, -2] = 0.0 if r.snr_db is None else 1.0
        x[i, -1] = 0.0 if r.rir_id is None else 1.0
    return x


def _fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, steps: int = 300) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    lr = 0.5
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(xb @ w)))
        w -= lr * (xb.T @ (p - y) / len(y) + l2 * w)
    return w


def _predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])
    return (xb @ w > 0).astype(np.float64)


def check_symmetry(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> SymmetryResult:
    n = len(y)
    if n < folds * 2:
        raise ValueError("not enough rows for a symmetry check")
    order = np.random.default_rng(seed).permutation(n)
    correct = 0
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        w = _fit_logreg(x[train], y[train])
        correct += int(np.sum(_predict(w, x[test]) == y[test]))
    acc = correct / n
    p = float(np.mean(y))
    baseline = max(p, 1 - p)
    # ~3 standard errors of a binomial proportion, floor 2 points.
    margin = max(0.02, 3 * float(np.sqrt(baseline * (1 - baseline) / n)))
    return SymmetryResult(accuracy=acc, baseline=baseline, margin=margin, n=n)


def check_manifest_symmetry(rows: Sequence[ManifestRow], seed: int = 0) -> SymmetryResult:
    y = np.array([1.0 if r.label == "spoof" else 0.0 for r in rows])
    return check_symmetry(channel_features(rows), y, seed=seed)
