"""ml.training.train_head_c — train the Head C prosody model (B6-T05, B6-T06).

Usage:
    python -m ml.training.train_head_c --config ml/training/configs/head_c.yaml

Steps: extract prosody for 3 s windows -> fit per-language bona fide
normalisation -> train ProsodyNet (BCE) -> Platt-calibrate on dev -> write a
per-language FPR / miss-rate report into the checkpoint metadata. The run
FAILS (non-zero exit) if the per-language FPR gap exceeds ``fairness.max_fpr_gap``
and ``fairness.enforce`` is true.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ml.data.license_gate import DataPolicy, enforce
from ml.data.manifest import ManifestRow
from ml.training.dataset import crop_or_pad
from ml.training.metrics import compute_eer
from ml.training.train_head_a import load_split_rows, seed_everything
from ml.training.train_head_b import fit_platt_logit
from packages.vg_models.heads.head_c_prosody.model import ProsodyNet, analyse, global_vector, save
from packages.vg_models.heads.head_c_prosody.normalize import LanguageNorm, per_language_report

WINDOW = 48000


class FairnessGateFailed(RuntimeError):  # noqa: N818
    pass


def featurise(
    rows: list[ManifestRow], load: Callable[[ManifestRow], np.ndarray], seed: int
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feats, frames, labels = [], [], []
    for r in rows:
        a = analyse(crop_or_pad(load(r), WINDOW, rng, train=True), r.language)
        feats.append(a.features)
        frames.append(a.frames)
        labels.append(1.0 if r.label == "spoof" else 0.0)
    return feats, np.stack(frames), np.array(labels, dtype=np.float32)


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(cfg["seed"])
    policy = DataPolicy.from_config(cfg)
    tr_rows, dv_rows, load = load_split_rows(cfg)
    tr_rows = enforce(tr_rows, policy, purpose="train")
    dv_rows = enforce(dv_rows, policy, purpose="eval")
    ftr, xtr, ytr = featurise(tr_rows, load, cfg["seed"])
    fdv, xdv, ydv = featurise(dv_rows, load, cfg["seed"] + 1)

    names = list(ftr[0])
    bona_idx = [i for i, y in enumerate(ytr) if y == 0]
    norm = LanguageNorm.fit([ftr[i] for i in bona_idx], [tr_rows[i].language for i in bona_idx])
    gtr = np.stack(
        [global_vector(f, names, norm, r.language) for f, r in zip(ftr, tr_rows, strict=True)]
    )
    gdv = np.stack(
        [global_vector(f, names, norm, r.language) for f, r in zip(fdv, dv_rows, strict=True)]
    )

    t = cfg.get("train", {})
    model = ProsodyNet(len(names), hidden=t.get("hidden", 32))
    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 1e-3)), weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xtr_t, gtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(gtr), torch.from_numpy(ytr)
    bs = int(t.get("batch_size", 32))
    for _ in range(int(t.get("epochs", 10))):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(model(xtr_t[b], gtr_t[b]), ytr_t[b]).backward()
            opt.step()

    model.eval()
    with torch.inference_mode():
        z = model(torch.from_numpy(xdv), torch.from_numpy(gdv)).numpy().astype(np.float64)
    eer, _ = compute_eer(-z[ydv == 0], -z[ydv == 1])
    a, b = fit_platt_logit(z, ydv.astype(np.float64))
    p_dev = 1 / (1 + np.exp(-np.clip(a * z + b, -30, 30)))
    fair_cfg = cfg.get("fairness", {})
    report = per_language_report(
        p_dev,
        ydv,
        [r.language for r in dv_rows],
        threshold=fair_cfg.get("threshold", 0.5),
        max_fpr_gap=fair_cfg.get("max_fpr_gap", 0.05),
    )

    out = Path(cfg.get("out_dir", "runs")) / cfg["run_name"]
    meta = {
        "model_version": f"C@prosody-bigru-v{cfg.get('version', '0.1.0')}",
        "lineage": policy.lineage,
        "allow_noncommercial": policy.allow_noncommercial,
        "dev_eer_training_only": eer,
        "calibration": {
            "scale": a,
            "bias": b,
            "version": cfg.get("calibration_version", "cal-train"),
        },
        "per_language_dev": report,
        "normalised_languages": sorted(norm.per_language),
        "train_corpora": sorted({r.source_corpus for r in tr_rows}),
        "n_train": len(tr_rows),
    }
    save(model, names, norm, out / "head_c.pt", meta)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if fair_cfg.get("enforce", False) and not report["fairness_ok"]:
        raise FairnessGateFailed(f"per-language FPR gap {report['max_fpr_gap']:.3f} exceeds limit")
    return {"dev_eer": eer, "model": str(out / "head_c.pt"), "fairness_ok": report["fairness_ok"]}


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
