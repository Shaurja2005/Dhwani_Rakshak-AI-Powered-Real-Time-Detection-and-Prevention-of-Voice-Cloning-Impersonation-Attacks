"""Lightweight gradient-boosted decision trees (B5-T03).

Pure numpy, no lightgbm/sklearn dependency: histogram splits on quantile bins,
logistic loss, depth-limited trees. Prediction is fully vectorised and takes
well under a millisecond for one window with ~100 trees of depth 3.

Serialises to JSON so the artifact is inspectable and diff-able.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Tree:
    # Complete binary tree arrays of size 2**depth - 1 (internal) and 2**depth (leaves).
    feature: np.ndarray
    threshold: np.ndarray
    leaf: np.ndarray
    depth: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        node = np.zeros(len(x), dtype=np.int64)
        for _ in range(self.depth):
            f = self.feature[node]
            go_right = x[np.arange(len(x)), f] >= self.threshold[node]
            node = 2 * node + 1 + go_right
        return self.leaf[node - (2**self.depth - 1)]


@dataclass
class GBDT:
    feature_names: list[str]
    base: float = 0.0
    lr: float = 0.1
    trees: list[Tree] = field(default_factory=list)
    medians: np.ndarray | None = None  # NaN imputation values

    # ------------------------------------------------------------------ inference
    def _impute(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.medians is not None:
            x = np.where(np.isnan(x), self.medians, x)
        return x

    def _stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        key = len(self.trees)
        cache = getattr(self, "_stack_cache", None)
        if cache is None or cache[0] != key:
            depth = max(t.depth for t in self.trees)
            if any(t.depth != depth for t in self.trees):
                raise ValueError("vectorised prediction needs equal-depth trees")
            cache = (
                key,
                np.stack([t.feature for t in self.trees]),
                np.stack([t.threshold for t in self.trees]),
                np.stack([t.leaf for t in self.trees]),
                depth,
            )
            object.__setattr__(self, "_stack_cache", cache)
        return cache[1], cache[2], cache[3], cache[4]

    def decision(self, x: np.ndarray) -> np.ndarray:
        """Raw log-odds that the input is spoof. All trees traversed in one vectorised pass."""
        x = self._impute(np.atleast_2d(x))
        if not self.trees:
            return np.full(len(x), self.base)
        feat, thr, leaf, depth = self._stacked()  # [K, nodes], [K, nodes], [K, leaves]
        k = np.arange(len(self.trees))[:, None]
        rows = np.arange(len(x))[None, :]
        node = np.zeros((len(self.trees), len(x)), dtype=np.int64)
        for _ in range(depth):
            f = feat[k, node]
            node = 2 * node + 1 + (x[rows, f] >= thr[k, node])
        return self.base + self.lr * leaf[k, node - (2**depth - 1)].sum(axis=0)

    def feature_importance(self) -> dict[str, int]:
        counts = np.zeros(len(self.feature_names), dtype=int)
        for t in self.trees:
            np.add.at(counts, t.feature, 1)
        return {
            n: int(c)
            for n, c in sorted(zip(self.feature_names, counts, strict=True), key=lambda p: -p[1])
            if c
        }

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        n_trees: int = 100,
        depth: int = 3,
        lr: float = 0.1,
        n_bins: int = 32,
        min_leaf: int = 5,
        l2: float = 1.0,
        seed: int = 0,
        subsample: float = 0.8,
    ) -> GBDT:
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> imputed as 0
            medians = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        x = np.where(np.isfinite(x), x, medians)
        p0 = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        model = cls(
            feature_names=list(feature_names),
            base=float(np.log(p0 / (1 - p0))),
            lr=lr,
            medians=medians,
        )
        edges = [
            np.unique(np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
            for j in range(x.shape[1])
        ]
        binned = np.stack(
            [np.searchsorted(e, x[:, j], side="right") for j, e in enumerate(edges)], axis=1
        )
        f = np.full(len(x), model.base)
        for _ in range(n_trees):
            p = 1 / (1 + np.exp(-f))
            g, h = p - y, np.maximum(p * (1 - p), 1e-6)
            rows = np.where(rng.random(len(x)) < subsample)[0]
            tree = _grow(binned, edges, g, h, rows, depth, n_bins, min_leaf, l2)
            model.trees.append(tree)
            f += lr * tree.predict(x)
        return model

    # ------------------------------------------------------------------ IO
    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "base": self.base,
            "lr": self.lr,
            "medians": None if self.medians is None else self.medians.tolist(),
            "trees": [
                {
                    "feature": t.feature.tolist(),
                    "threshold": t.threshold.tolist(),
                    "leaf": t.leaf.tolist(),
                    "depth": t.depth,
                }
                for t in self.trees
            ],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> GBDT:
        return cls(
            feature_names=d["feature_names"],
            base=d["base"],
            lr=d["lr"],
            medians=None if d.get("medians") is None else np.array(d["medians"]),
            trees=[
                Tree(
                    np.array(t["feature"]),
                    np.array(t["threshold"]),
                    np.array(t["leaf"]),
                    int(t["depth"]),
                )
                for t in d["trees"]
            ],
        )

    def save(self, path: Path | str, meta: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"model": self.to_json(), "meta": meta or {}}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> tuple[GBDT, dict[str, Any]]:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(blob["model"]), blob.get("meta", {})


def _grow(
    binned: np.ndarray,
    edges: list[np.ndarray],
    g: np.ndarray,
    h: np.ndarray,
    rows: np.ndarray,
    depth: int,
    n_bins: int,
    min_leaf: int,
    l2: float,
) -> Tree:
    n_internal = 2**depth - 1
    feature = np.zeros(n_internal, dtype=np.int64)
    threshold = np.full(n_internal, np.inf)
    leaf = np.zeros(2**depth)
    assign = {0: rows}
    for node in range(n_internal):
        idx = assign.pop(node, np.array([], dtype=np.int64))
        best = (0.0, 0, n_bins)  # gain, feature, bin
        if len(idx) >= 2 * min_leaf:
            gs, hs = g[idx].sum(), h[idx].sum()
            parent = gs * gs / (hs + l2)
            for j in range(binned.shape[1]):
                gh = np.bincount(binned[idx, j], weights=g[idx], minlength=n_bins + 1)
                hh = np.bincount(binned[idx, j], weights=h[idx], minlength=n_bins + 1)
                cnt = np.bincount(binned[idx, j], minlength=n_bins + 1)
                gl, hl, cl = np.cumsum(gh)[:-1], np.cumsum(hh)[:-1], np.cumsum(cnt)[:-1]
                ok = (cl >= min_leaf) & (len(idx) - cl >= min_leaf)
                if not ok.any():
                    continue
                gain = gl**2 / (hl + l2) + (gs - gl) ** 2 / (hs - hl + l2) - parent
                gain = np.where(ok, gain, -np.inf)
                b = int(np.argmax(gain))
                if gain[b] > best[0] and b < len(edges[j]):
                    best = (float(gain[b]), j, b)
        _, j, b = best
        left, right = 2 * node + 1, 2 * node + 2
        if b < n_bins and best[0] > 0:
            feature[node], threshold[node] = j, edges[j][b]
            mask = binned[idx, j] <= b
            assign[left], assign[right] = idx[mask], idx[~mask]
        else:
            # No useful split: everything goes left (threshold = +inf) and stays together.
            assign[left], assign[right] = idx, np.array([], dtype=np.int64)
    for k in range(2**depth):
        idx = assign.get(n_internal + k, np.array([], dtype=np.int64))
        leaf[k] = -g[idx].sum() / (h[idx].sum() + l2) if len(idx) else 0.0
    return Tree(feature, threshold, leaf, depth)
