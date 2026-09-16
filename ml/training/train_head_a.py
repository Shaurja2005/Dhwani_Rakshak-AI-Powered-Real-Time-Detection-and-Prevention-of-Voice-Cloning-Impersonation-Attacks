"""ml.training.train_head_a — config-driven Head A training (B4-T06/T07/T08).

Usage:
    python -m ml.training.train_head_a --config ml/training/configs/head_a_stage1.yaml
    python -m ml.training.train_head_a --config ml/training/configs/head_a_smoke.yaml

A config with ``data.source: synthetic`` trains on generated toy audio: this
only proves the loop runs end to end. It is not a detector.

Every run writes ``<out_dir>/<run_name>/``: best.pt (with lineage, calibration
and data provenance in its metadata), config.yaml and metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ml.data.license_gate import DataPolicy
from ml.data.manifest import ManifestRow, read_manifests
from ml.training.augment import Augmenter
from ml.training.dataset import FamilyBalancedSampler, ManifestAudioDataset, soundfile_loader
from ml.training.losses import AMSoftmax, build_loss
from ml.training.metrics import compute_eer, fit_platt
from ml.training.robust import SAM, DomainAdversary, consistency_loss, mixup
from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data
def synthetic_rows(
    n: int, seed: int, prefix: str
) -> tuple[list[ManifestRow], dict[str, np.ndarray]]:
    """Toy corpus: 'spoof' = clean harmonic stack, 'bona fide' = jittery harmonic + breath noise."""
    rng = np.random.default_rng(seed)
    rows, audio = [], {}
    for i in range(n):
        label = "spoof" if i % 2 else "bona_fide"
        t = np.arange(32000) / 16000
        f0 = rng.uniform(100, 250) * (
            1 + (0.02 * np.sin(2 * np.pi * 5 * t) if label == "bona_fide" else 0)
        )
        x = sum(np.sin(2 * np.pi * np.cumsum(f0 * k) / 16000) / k for k in range(1, 6))
        if label == "bona_fide":
            x = x + 0.3 * rng.standard_normal(len(t))
        utt = f"{prefix}{i:05d}"
        audio[utt] = (0.3 * x / np.max(np.abs(x))).astype(np.float32)
        rows.append(
            ManifestRow(
                utt_id=utt,
                path=f"mem://{utt}",
                label=label,  # type: ignore[arg-type]
                generator_family=f"synthetic_{i % 3}" if label == "spoof" else None,
                language="hi",
                speaker_id=f"{prefix}spk{i % 20}",
                source_corpus="vg_indic_telephony",
                license="internal-synthetic",
                commercial_use=False,
                duration_s=2.0,
                sample_rate=16000,
            )
        )
    return rows, audio


def load_split_rows(cfg: dict[str, Any]) -> tuple[list[ManifestRow], list[ManifestRow], Any]:
    d = cfg["data"]
    if d.get("source") == "synthetic":
        tr, a1 = synthetic_rows(d.get("n_train", 64), cfg["seed"], "tr")
        dv, a2 = synthetic_rows(d.get("n_dev", 32), cfg["seed"] + 1, "dv")
        audio = {**a1, **a2}
        return tr, dv, lambda r: audio[r.utt_id]
    rows = read_manifests(d["manifests"])
    by_id = {r.utt_id: r for r in rows}
    splits = json.loads(Path(d["splits"]).read_text(encoding="utf-8"))["splits"]
    langs = set(d.get("languages") or [])
    keep = (lambda r: r.language in langs) if langs else (lambda r: True)
    train = [by_id[u] for u in splits["train"] if keep(by_id[u])]
    dev = [by_id[u] for u in splits["dev"] if keep(by_id[u])]
    return train, dev, soundfile_loader(d.get("data_root", "data"))


# ---------------------------------------------------------------- loop
def evaluate(
    model: HeadAModel, loader: DataLoader, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for wav, y, _ in loader:
            _, s = model(wav.to(device))
            scores.append(s.cpu().numpy())
            labels.append(y.numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    eer, _ = compute_eer(s[y == 1], s[y == 0])
    return eer, s, y


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(cfg["seed"])
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = DataPolicy.from_config(cfg)  # hard-fails on missing lineage / license flags (I7)
    t = cfg.get("train", {})
    r = cfg.get("robust", {})

    train_rows, dev_rows, load = load_split_rows(cfg)
    aug = Augmenter(rawboost_algo=int(cfg.get("augment", {}).get("rawboost_algo", 0)))
    ds_tr = ManifestAudioDataset(
        train_rows, policy, load, t.get("seconds", 4.0), True, aug, cfg["seed"]
    )
    ds_dv = ManifestAudioDataset(
        dev_rows, policy, load, t.get("seconds", 4.0), False, None, cfg["seed"], "eval"
    )
    sampler = (
        FamilyBalancedSampler(ds_tr.rows, seed=cfg["seed"])
        if t.get("balance_families", True)
        else None
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=t.get("batch_size", 16),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=t.get("num_workers", 0),
        drop_last=True,
    )
    dl_dv = DataLoader(
        ds_dv, batch_size=t.get("batch_size", 16), num_workers=t.get("num_workers", 0)
    )

    mcfg = HeadAConfig.from_dict(cfg)
    model = HeadAModel(mcfg).to(device)
    if cfg.get("init_checkpoint"):  # e.g. Stage 2 starts from Stage 1
        blob = torch.load(
            cfg["init_checkpoint"], map_location=device, weights_only=False
        )  # noqa: S614
        model.load_state_dict(blob["state"], strict=False)
    loss_fn = build_loss(cfg.get("loss", "oc_softmax"), mcfg.emb_dim).to(device)
    domains = sorted({c for row in ds_tr.rows for c in [_domain(row)]})
    adversary = (
        DomainAdversary(mcfg.emb_dim, len(domains), r.get("domain_adv_lambda", 0.1)).to(device)
        if r.get("domain_adversarial") and len(domains) > 1
        else None
    )
    params = [p for p in model.parameters() if p.requires_grad] + list(loss_fn.parameters())
    if adversary is not None:
        params += list(adversary.parameters())
    opt_kwargs = dict(lr=float(t.get("lr", 1e-4)), weight_decay=float(t.get("weight_decay", 1e-4)))
    opt: torch.optim.Optimizer = (
        SAM(params, torch.optim.AdamW, rho=r.get("sam_rho", 0.05), **opt_kwargs)
        if r.get("sam")
        else torch.optim.AdamW(params, **opt_kwargs)
    )

    out = Path(cfg.get("out_dir", "runs")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    best = {"eer": 1.0, "epoch": -1}
    metrics_f = (out / "metrics.jsonl").open("w", encoding="utf-8")

    def compute_loss(wav: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        lam, y_b = 1.0, y
        if r.get("mixup_alpha", 0) > 0:
            wav, y, y_b, lam = mixup(wav, y, float(r["mixup_alpha"]))
        emb, score = model(wav)

        def base(labels: torch.Tensor) -> torch.Tensor:
            if isinstance(loss_fn, AMSoftmax):
                return loss_fn(emb, model.center, labels)
            return loss_fn(score, labels)

        loss = lam * base(y) + (1 - lam) * base(y_b) if lam < 1 else base(y)
        if r.get("consistency_weight", 0) > 0:
            noisy = wav + 0.01 * torch.randn_like(wav)
            loss = loss + float(r["consistency_weight"]) * consistency_loss(emb, model.embed(noisy))
        if adversary is not None:
            dom = torch.tensor(
                [domains.index(_domain(ds_tr.rows[int(i)])) for i in idx], device=device
            )
            loss = loss + adversary(emb, dom)
        return loss

    for epoch in range(int(t.get("epochs", 10))):
        model.train()
        ds_tr.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0, losses = time.time(), []
        for wav, y, idx in dl_tr:
            wav, y = wav.to(device), y.to(device)
            opt.zero_grad()
            loss = compute_loss(wav, y, idx)
            loss.backward()
            if isinstance(opt, SAM):

                def closure(
                    w: torch.Tensor = wav, yy: torch.Tensor = y, ii: torch.Tensor = idx
                ) -> torch.Tensor:
                    second = compute_loss(w, yy, ii)
                    second.backward()
                    return second

                opt.step(closure)
            else:
                opt.step()
            losses.append(float(loss.item()))
        eer, scores, labels = evaluate(model, dl_dv, device)
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "dev_eer": eer,
            "seconds": round(time.time() - t0, 1),
            "layer_weights": model.layer_sum.weights(),
        }
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        print(json.dumps(rec))
        if eer <= best["eer"]:
            scale, bias = fit_platt(scores, (labels == 0).astype(float))
            best = {"eer": eer, "epoch": epoch}
            save_checkpoint(
                model,
                out / "best.pt",
                {
                    "lineage": policy.lineage,
                    "allow_noncommercial": policy.allow_noncommercial,
                    "run_name": cfg["run_name"],
                    "dev_eer_training_only": eer,  # not a reportable number: use B15 `make eval`
                    "calibration": {
                        "scale": scale,
                        "bias": bias,
                        "version": f"cal-{time.strftime('%Y-%m-%d')}",
                    },
                    "train_corpora": sorted({row.source_corpus for row in ds_tr.rows}),
                    "languages": sorted({row.language for row in ds_tr.rows}),
                    "n_train": len(ds_tr.rows),
                },
            )
    metrics_f.close()
    return {**best, "checkpoint": str(out / "best.pt"), "model_version": model.model_version}


def _domain(row: ManifestRow) -> str:
    codecs = [c for c in row.codec_chain if c.startswith("codec:")]
    return codecs[-1] if codecs else "clean"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value (top-level or a.b)")
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
