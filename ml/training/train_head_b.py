"""ml.training.train_head_b — train the Head B GBDT (B5-T03).

Usage:
    python -m ml.training.train_head_b --config ml/training/configs/head_b.yaml
    python -m ml.training.train_head_b --config ml/training/configs/head_b.yaml \
        --override data.source=synthetic out_dir=runs

Extracts Head B features for 3 s windows cropped from each utterance, fits the
GBDT, Platt-calibrates on dev, and stores bona fide reference statistics used
for explanations. Output: ``<out_dir>/<run_name>/head_b.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.data.license_gate import DataPolicy, enforce
from ml.data.manifest import ManifestRow
from ml.training.dataset import crop_or_pad
from ml.training.metrics import compute_eer
from ml.training.train_head_a import load_split_rows
from packages.vg_models.heads.head_b_dsp.gbdt import GBDT
from packages.vg_models.heads.head_b_dsp.pipeline import analyse

WINDOW = 48000


def featurise(
    rows: list[ManifestRow], load: Callable[[ManifestRow], np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    feats, labels, names = [], [], None
    for r in rows:
        x = crop_or_pad(load(r), WINDOW, rng, train=True)
        a = analyse(x.astype(np.float32), r.sample_rate)
        names = names or list(a.features)
        feats.append([a.features[n] for n in names])
        labels.append(1.0 if r.label == "spoof" else 0.0)
    return np.array(feats, dtype=np.float64), np.array(labels), names or []


def fit_platt_logit(z: np.ndarray, y: np.ndarray, steps: int = 500) -> tuple[float, float]:
    """p_spoof = sigmoid(a*z + b), z = GBDT log-odds."""
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(a * z + b, -30, 30)))
        a -= 0.1 * float(np.mean((p - y) * z))
        b -= 0.1 * float(np.mean(p - y))
    return a, b


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    policy = DataPolicy.from_config(cfg)
    train_rows, dev_rows, load = load_split_rows(cfg)
    train_rows = enforce(train_rows, policy, purpose="train")
    dev_rows = enforce(dev_rows, policy, purpose="eval")
    xtr, ytr, names = featurise(train_rows, load, cfg["seed"])
    xdv, ydv, _ = featurise(dev_rows, load, cfg["seed"] + 1)

    g = cfg.get("gbdt", {})
    model = GBDT.fit(
        xtr,
        ytr,
        names,
        n_trees=g.get("n_trees", 100),
        depth=g.get("depth", 3),
        lr=g.get("lr", 0.1),
        min_leaf=g.get("min_leaf", 5),
        seed=cfg["seed"],
    )
    z = model.decision(xdv)
    eer, _ = compute_eer(-z[ydv == 0], -z[ydv == 1])  # higher -z = more bona fide
    a, b = fit_platt_logit(z, ydv)
    bona = xtr[ytr == 0]
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # features that are NaN for every bona fide row
        reference = {
            n: (float(np.nanmean(bona[:, i])), float(np.nanstd(bona[:, i])))
            for i, n in enumerate(names)
            if np.isfinite(bona[:, i]).any()
        }

    out = Path(cfg.get("out_dir", "runs")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_version": f"B@dsp-gbdt-v{cfg.get('version', '0.1.0')}",
        "lineage": policy.lineage,
        "allow_noncommercial": policy.allow_noncommercial,
        "dev_eer_training_only": eer,  # not reportable: B15 `make eval` is the source of truth
        "calibration": {
            "scale": a,
            "bias": b,
            "version": cfg.get("calibration_version", "cal-train"),
        },
        "bona_fide_reference": reference,
        "train_corpora": sorted({r.source_corpus for r in train_rows}),
        "languages": sorted({r.language for r in train_rows}),
        "n_train": len(train_rows),
        "top_features": dict(list(model.feature_importance().items())[:15]),
    }
    model.save(out / "head_b.json", meta)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {
        "dev_eer": eer,
        "model": str(out / "head_b.json"),
        "model_version": meta["model_version"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for kv in args.override:
        k, v = kv.split("=", 1)
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(v)
    print(json.dumps(train(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
