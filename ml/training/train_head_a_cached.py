"""Train the Head A back-end on cached frozen-SSL features (base model, step 5).

    python -m ml.training.train_head_a_cached --config ml/training/configs/head_a_base.yaml

Trains the layer weights + Nes2Net back-end + bona fide centre on
``ml/training/feature_cache.py`` output. Every epoch: mixed-precision training
(fp16 autocast + GradScaler) with family-balanced sampling and feature-space masking,
then dev EER. Writes ``runs/<run_name>/``:

    best.pt        standard Head A checkpoint (loads with ``load_checkpoint`` / HeadA / make eval),
                   with Platt calibration fitted on dev and full provenance in its metadata
    last.pt        optimizer + scaler + epoch, so ``--resume`` (default) continues after a stop
    metrics.jsonl  one line per epoch

``train.train_sets`` / ``train.dev_sets`` list the cache splits to use (default
``[train]`` / ``[dev]``); augmented views are just more splits. With several dev sets the
pooled dev EER drives early stopping and each set's EER is logged.

Dev EER here is a training signal only. Reportable numbers come from ``make eval`` (B15).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from ml.training.feature_cache import resolve
from ml.training.losses import AMSoftmax, build_loss
from ml.training.metrics import compute_eer, fit_platt
from ml.training.runtime import gpu_report, prepare_process, setup_cuda
from packages.vg_models.heads.head_a_ssl.frontend import SSLFrontend
from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel, save_checkpoint

SHORT_NAMES = {
    "facebook/wav2vec2-xls-r-300m": "xlsr300m",
    "facebook/wav2vec2-xls-r-1b": "xlsr1b",
    "microsoft/wavlm-large": "wavlmlarge",
}


class CachedFeatures(Dataset[tuple[torch.Tensor, int, int]]):
    """(feats [L,T,D] fp16, label 1 = bona fide, index). Opens the memmap lazily per worker."""

    def __init__(self, cache_dir: Path, train: bool, mask_prob: float = 0.0, seed: int = 0) -> None:
        self.dir = cache_dir
        self.meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        done = np.load(cache_dir / "done.npy")
        if not done.all():
            raise SystemExit(
                f"{cache_dir}: feature cache incomplete ({done.sum()}/{len(done)}); "
                "re-run feature_cache to finish it"
            )
        self.labels = np.asarray(self.meta["labels"], dtype=np.int64)
        self.families = list(self.meta["families"])
        self.train, self.mask_prob, self.seed, self.epoch = train, mask_prob, seed, 0
        self._feats: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        if self._feats is None:
            self._feats = np.load(self.dir / "feats.npy", mmap_mode="r")
        x = torch.from_numpy(np.array(self._feats[i]))  # copy out of the memmap
        # torch's RNG: DataLoader reseeds it per worker and per epoch (persistent workers too)
        if self.train and self.mask_prob > 0 and float(torch.rand(())) < self.mask_prob:
            t, d = x.shape[1], x.shape[2]  # SpecAugment-style: one time span + one channel band
            w = int(torch.randint(1, max(2, t // 8), ()))
            s = int(torch.randint(0, t - w, ()))
            x[:, s : s + w, :] = 0
            cw = int(torch.randint(1, max(2, d // 16), ()))
            cs = int(torch.randint(0, d - cw, ()))
            x[:, :, cs : cs + cw] = 0
        return x, int(self.labels[i]), i


class MultiCached(Dataset[tuple[torch.Tensor, int, int]]):
    """Several cache splits read as one (e.g. clean ``train`` + augmented ``train_aug1``)."""

    def __init__(self, parts: list[CachedFeatures]) -> None:
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])
        self.labels = np.concatenate([p.labels for p in parts])
        self.families = [f for p in parts for f in p.families]
        self.part_of = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])
        first = parts[0].meta
        keys = ("layers", "frames", "dim", "frontend", "frontend_revision")
        for p in parts[1:]:
            if [p.meta.get(k) for k in keys] != [first.get(k) for k in keys]:
                raise SystemExit(
                    f"{p.dir}: cache was built with a different front-end/geometry than "
                    f"{parts[0].dir} ({keys}); rebuild it"
                )
        self.meta = {
            **first,
            **{
                k: sorted({x for p in parts for x in p.meta[k]})
                for k in ("corpora", "languages", "families")
            },
        }
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        x, y, _ = self.parts[k][i - int(self.offsets[k])]
        return x, y, i


class BalancedSampler(Sampler[int]):
    """Half bona fide, half spoof split equally across attack families (seeded per epoch)."""

    def __init__(self, labels: np.ndarray, families: list[str], seed: int) -> None:
        self.bona = np.flatnonzero(labels == 1)
        fams: dict[str, list[int]] = {}
        for i, f in enumerate(families):
            if labels[i] == 0:
                fams.setdefault(f, []).append(i)
        self.fams = [np.asarray(v) for _, v in sorted(fams.items())]
        self.n = len(labels)
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self):  # type: ignore[no-untyped-def]  # noqa: ANN204
        rng = np.random.default_rng((self.seed, self.epoch))
        n_b = self.n // 2
        idx = [rng.choice(self.bona, n_b, replace=len(self.bona) < n_b)]
        per = (self.n - n_b) // max(1, len(self.fams))
        idx += [rng.choice(f, per, replace=len(f) < per) for f in self.fams]
        out = np.concatenate(idx)
        rng.shuffle(out)
        return iter(out.tolist())


class StubFrontend(SSLFrontend):
    """Stands in for the frozen SSL model: the cache already holds its hidden states."""

    def __init__(self, hidden: int, num_hidden_states: int, name: str) -> None:
        super().__init__()
        self.hidden_size, self.num_hidden_states, self.name = hidden, num_hidden_states, name

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise RuntimeError("cached-feature model: feed features, not audio")


def forward_cached(model: HeadAModel, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    layers = model.cfg.frontend_layers
    hs: list[torch.Tensor | None] = [None] * (max(layers) + 1)
    for j, layer in enumerate(layers):
        hs[layer] = feats[:, j]
    emb = F.normalize(model.backend(model.layer_sum(tuple(hs))), dim=-1)  # type: ignore[arg-type]
    return emb, model.score(emb)


@torch.no_grad()
def evaluate(
    model: HeadAModel, loader: DataLoader, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            _, s = forward_cached(model, x.to(device, non_blocking=True))
        scores.append(s.float().cpu().numpy())
        labels.append(y.numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    return compute_eer(s[y == 1], s[y == 0])[0], s, y


def train(cfg: dict[str, Any], resume: bool = True) -> dict[str, Any]:
    torch.manual_seed(int(cfg.get("seed", 0)))
    device = setup_cuda(cfg.get("device"))
    t = cfg.get("train", {})
    cache = resolve(cfg["cache"]["dir"])
    train_sets = list(t.get("train_sets", ["train"]))
    dev_sets = list(t.get("dev_sets", ["dev"]))
    mask = float(t.get("feature_mask_prob", 0.5))
    ds_tr = MultiCached(
        [CachedFeatures(cache / s, True, mask, int(cfg.get("seed", 0))) for s in train_sets]
    )
    ds_dv = MultiCached([CachedFeatures(cache / s, False) for s in dev_sets])
    meta = ds_tr.meta
    augmentation = [
        {"split": s, **p.meta["augmentation"]}
        for s, p in zip(train_sets, ds_tr.parts, strict=True)
        if "augmentation" in p.meta
    ]
    if meta["layers"] != list(cfg["frontend_layers"]):
        raise SystemExit("cache layers differ from config frontend_layers; rebuild the cache")
    workers = int(t.get("num_workers", 4))
    sampler = BalancedSampler(ds_tr.labels, ds_tr.families, int(cfg.get("seed", 0)))
    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(t.get("batch_size", 64)),
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
        persistent_workers=workers > 0,
    )
    dl_dv = DataLoader(
        ds_dv,
        batch_size=int(t.get("batch_size", 64)) * 2,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
    )

    cfg = {
        **cfg,
        "frontend_revision": cfg.get("frontend_revision") or meta.get("frontend_revision"),
    }
    mcfg = HeadAConfig.from_dict(cfg)
    stub = StubFrontend(
        meta["dim"],
        max(mcfg.frontend_layers) + 2,
        SHORT_NAMES.get(mcfg.frontend, mcfg.frontend.split("/")[-1]),
    )
    model = HeadAModel(mcfg, frontend=stub).to(device)
    loss_fn = build_loss(cfg.get("loss", "oc_softmax"), mcfg.emb_dim).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(loss_fn.parameters())
    opt = torch.optim.AdamW(
        params, lr=float(t.get("lr", 3e-4)), weight_decay=float(t.get("weight_decay", 1e-4))
    )
    epochs = int(t.get("epochs", 30))
    steps = epochs * len(dl_tr)
    warm = int(0.03 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / max(1, warm)
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    out = resolve(cfg.get("out_dir", "runs")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    best, start, bad = {"eer": 1.0, "epoch": -1}, 0, 0
    if resume and (out / "last.pt").exists():
        st = torch.load(
            out / "last.pt", map_location=device, weights_only=False
        )  # noqa: S614 - own file
        model.load_state_dict(st["model"])
        loss_fn.load_state_dict(st["loss"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        best, start, bad = st["best"], st["epoch"] + 1, st.get("bad", 0)
        print(
            f"resumed from epoch {st['epoch']} (best dev EER {best['eer']:.4f} @ {best['epoch']})"
        )
    elif t.get("init_from"):
        # fine-tune: start from an earlier model (e.g. the previous language's best.pt)
        init = resolve(t["init_from"])
        src = torch.load(init, map_location=device, weights_only=False)  # noqa: S614 - own file
        state = {k: v for k, v in src["state"].items() if not k.startswith("frontend.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("frontend.") for k in missing):
            raise SystemExit(f"{init} does not match this model ({missing}, {unexpected})")
        print(f"fine-tuning from {init} ({src['meta'].get('run_name')})")
    patience = int(t.get("patience", 6))
    stop_on = str(t.get("early_stop", "pooled"))  # pooled | mean_by_set
    if bad >= patience:  # the resumed run had already early-stopped: nothing left to train
        print(f"already finished (early stop at epoch {start - 1})")
        start = epochs

    with (out / "metrics.jsonl").open("a", encoding="utf-8") as mf:
        for epoch in range(start, epochs):
            model.train()
            ds_tr.epoch = sampler.epoch = epoch
            t0, losses = time.time(), []
            for x, y, _ in dl_tr:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    emb, score = forward_cached(model, x)
                    loss = (
                        loss_fn(emb, model.center, y)
                        if isinstance(loss_fn, AMSoftmax)
                        else loss_fn(score.float(), y)
                    )
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(float(loss.item()))
            eer, scores, labels = evaluate(model, dl_dv, device)
            by_set = {}
            if len(dev_sets) > 1:  # pooled EER drives early stopping; per-set EERs are for reading
                for k, name in enumerate(dev_sets):
                    m = ds_dv.part_of == k
                    s_k, y_k = scores[m], labels[m]
                    by_set[name] = round(compute_eer(s_k[y_k == 1], s_k[y_k == 0])[0], 5)
            if stop_on == "mean_by_set" and by_set:
                # every dev set (each language, clean and augmented) counts equally, so
                # forgetting an earlier language is penalised even if it is small
                eer = float(np.mean(list(by_set.values())))
            rec = {
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 5),
                "dev_eer": round(eer, 5),
                **({"dev_eer_by_set": by_set, "early_stop": stop_on} if by_set else {}),
                "lr": sched.get_last_lr()[0],
                "seconds": round(time.time() - t0, 1),
                "layer_weights": [round(w, 3) for w in model.layer_sum.weights()],
                "gpu": gpu_report(),
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            print(json.dumps(rec), flush=True)
            if eer < best["eer"]:
                bad = 0
                scale, bias = fit_platt(scores, (labels == 0).astype(float))
                best = {"eer": eer, "epoch": epoch}
                save_checkpoint(
                    model,
                    out / "best.pt",
                    {
                        "lineage": cfg["lineage"],
                        "allow_noncommercial": cfg["allow_noncommercial"],
                        "run_name": cfg["run_name"],
                        "trainer": "train_head_a_cached",
                        "init_from": str(t["init_from"]) if t.get("init_from") else None,
                        "dev_eer_training_only": eer,  # not reportable: use make eval (B15)
                        "calibration": {
                            "scale": scale,
                            "bias": bias,
                            "version": f"cal-{time.strftime('%Y-%m-%d')}",
                        },
                        "train_datasets": sorted(set(meta["corpora"])),
                        "train_families": sorted({f for f in meta["families"] if f != "bona_fide"}),
                        "languages": sorted(set(meta["languages"])),
                        "n_train": len(ds_tr),
                        "train_sets": train_sets,
                        "dev_sets": dev_sets,
                        "augmentation": augmentation,
                        "feature_cache": {
                            "dir": str(cache),
                            "frames": meta["frames"],
                            "seconds": meta["seconds"],
                        },
                        "frontend_revision": meta.get("frontend_revision"),
                    },
                )
            else:
                bad += 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "loss": loss_fn.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "bad": bad,
                },
                out / "last.pt",
            )
            if bad >= patience:
                print(f"early stop: no dev improvement for {patience} epochs")
                break
    return {**best, "checkpoint": str(out / "best.pt"), "model_version": model.model_version}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(prepare_process()))
    print(json.dumps(train(cfg, resume=not args.fresh)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
