"""Frozen-SSL feature cache for fast back-end training (base model, step 4).

    python -m ml.training.feature_cache --config ml/training/configs/head_a_base.yaml

With a frozen front-end, XLS-R gives the same hidden states every epoch, so computing
them once and training the back-end on the cache is ~20-50x cheaper than the
on-the-fly trainer. Per split the cache is

    <cache_dir>/<split>/feats.npy     float16 memmap [N, L, T, D]   (L = frontend_layers)
    <cache_dir>/<split>/meta.json     utt ids, labels, attack families, speakers, spec
    <cache_dir>/<split>/done.npy      bool[N] progress -> a stopped run resumes where it was

GPU use: fp16 autocast, only the layers the head reads are computed
(``truncate_to_used_layers``), pinned-memory loader with worker processes decoding FLAC
in parallel, cuDNN autotune, and the batch size halves automatically on out-of-memory.
Trade-off (documented, not hidden): one fixed 4 s crop per utterance, and waveform
augmentation only as a fixed number of pre-computed *augmented views*: a cache split
with ``source: <split>`` and ``view: k`` stores clips that went through a random call
channel before XLS-R (ml/training/channel_aug.py, label-blind, I3), plus
``chains.json`` recording what each clip got. The trainer adds feature-space masking;
the on-the-fly trainer (train_head_a.py) remains the path for fresh augmentation every
epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ml.data.license_gate import DataPolicy, enforce
from ml.data.manifest import ManifestRow, read_manifest
from ml.training.runtime import gpu_report, prepare_process, setup_cuda

ROOT = Path(__file__).resolve().parents[2]
SR = 16000


def resolve(p: str | Path) -> Path:
    """Relative paths in configs are relative to the project root, not the shell's cwd."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- selection
def select_rows(
    rows: list[ManifestRow], max_bona: int | None, spoof_per_family: int | None, seed: int
) -> list[ManifestRow]:
    """All/at most ``max_bona`` bona fide + up to ``spoof_per_family`` per attack family."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        groups["bona_fide" if r.label == "bona_fide" else f"spoof:{r.generator_family}"].append(r)
    out: list[ManifestRow] = []
    for key, members in sorted(groups.items()):
        cap = max_bona if key == "bona_fide" else spoof_per_family
        if cap is not None and len(members) > cap:
            idx = rng.choice(len(members), size=cap, replace=False)
            members = [members[i] for i in sorted(idx)]
        out += members
    return sorted(out, key=lambda r: r.utt_id)


# ---------------------------------------------------------------- audio
class CropDataset(Dataset[tuple[torch.Tensor, int]]):
    """Picklable (Windows spawn workers): decodes FLAC and returns one fixed 4 s crop."""

    def __init__(self, paths: list[str], seconds: float, train: bool, seed: int) -> None:
        self.paths, self.n, self.train, self.seed = paths, int(seconds * SR), train, seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        import soundfile as sf

        from ml.training.dataset import crop_or_pad

        x, sr = sf.read(self.paths[i], dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly

            x = resample_poly(x, SR, sr).astype(np.float32)
        x = crop_or_pad(x, self.n, np.random.default_rng((self.seed, i)), self.train)
        return torch.from_numpy(np.ascontiguousarray(x)), i


def build_frontend_for_cache(cfg: dict[str, Any], device: str) -> tuple[torch.nn.Module, str]:
    from packages.vg_models.heads.head_a_ssl.model import (
        HeadAConfig,
        HeadAModel,
        truncate_to_used_layers,
    )

    mcfg = HeadAConfig.from_dict(cfg)
    model = truncate_to_used_layers(HeadAModel(mcfg))
    fe = model.frontend.eval().to(device)
    fe.requires_grad_(False)
    rev = getattr(getattr(getattr(fe, "model", None), "config", None), "_commit_hash", None)
    return fe, str(rev or cfg.get("frontend_revision") or "unknown")


# ---------------------------------------------------------------- extraction
def extract_split(
    cfg: dict[str, Any],
    split: str,
    rows: list[ManifestRow],
    out: Path,
    device: str,
    batch: int,
    workers: int,
    augment: dict[str, Any] | None = None,
) -> Path:
    """``augment`` = {"view": k, **cfg["augment"]} builds an augmented view (channel_aug)."""
    layers = list(cfg["frontend_layers"])
    seconds = float(cfg.get("train", {}).get("seconds", 4.0))
    fe, revision = build_frontend_for_cache(cfg, device)
    with torch.inference_mode():  # probe output geometry once
        hs = fe(torch.zeros(1, int(seconds * SR), device=device))
    t_frames, dim = int(hs[0].shape[1]), int(hs[0].shape[2])
    out.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    meta = {
        "split": split,
        "n": n,
        "layers": layers,
        "frames": t_frames,
        "dim": dim,
        "seconds": seconds,
        "frontend": cfg["frontend"],
        "frontend_revision": revision,
        "utt_ids": [r.utt_id for r in rows],
        "labels": [int(r.label == "bona_fide") for r in rows],
        "families": [r.generator_family or "bona_fide" for r in rows],
        "speakers": [r.speaker_id for r in rows],
        "corpora": [r.source_corpus for r in rows],
        "languages": [r.language for r in rows],
    }
    if augment is not None:
        meta["augmentation"] = augment
    feats_path, done_path, meta_path = out / "feats.npy", out / "done.npy", out / "meta.json"
    chains_path = out / "chains.json"
    chains: list[str | None] = (
        json.loads(chains_path.read_text(encoding="utf-8"))
        if augment is not None and chains_path.exists()
        else [None] * n
    )
    shape = (n, len(layers), t_frames, dim)
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            old["utt_ids"] != meta["utt_ids"]
            or old["layers"] != layers
            or old.get("augmentation") != meta.get("augmentation")
        ):
            raise SystemExit(f"{out} holds a different selection; delete it to rebuild")
        feats = np.load(feats_path, mmap_mode="r+")
        done = np.load(done_path)
    else:
        need = np.prod(shape) * 2
        import shutil

        free = shutil.disk_usage(out).free
        print(f"{split}: cache needs {need / 1e9:.1f} GB, free {free / 1e9:.0f} GB")
        if need > free * 0.95:
            raise SystemExit(
                "not enough disk space for the feature cache: lower the selection sizes"
            )
        feats = np.lib.format.open_memmap(feats_path, mode="w+", dtype=np.float16, shape=shape)
        done = np.zeros(n, dtype=bool)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    todo = np.flatnonzero(~done)
    print(
        f"{split}: {n} utterances, {len(todo)} to extract -> {feats_path} "
        f"[{len(layers)}x{t_frames}x{dim} fp16 each]",
        flush=True,
    )
    if len(todo) == 0:
        return out
    data_root = resolve(cfg.get("data", {}).get("data_root", "data"))
    paths = [str(data_root / rows[i].path) for i in todo]
    ds: Dataset[Any]
    if augment is not None:
        from ml.training.channel_aug import AugCropDataset

        ds = AugCropDataset(
            paths,
            [rows[i].utt_id for i in todo],
            seconds,
            int(cfg.get("seed", 0)),
            int(augment["view"]),
            augment,
        )
    else:
        # random crops for training splits ("train", "indic_bn/train", ...), centre crops otherwise
        is_train = Path(split).name.startswith("train")
        ds = CropDataset(paths, seconds, train=is_train, seed=int(cfg.get("seed", 0)))

    def save_progress() -> None:
        feats.flush()
        np.save(done_path, done)
        if augment is not None:
            chains_path.write_text(json.dumps(chains), encoding="utf-8")

    t0, done_since, last = time.time(), 0, time.time()
    pos = 0
    while pos < len(todo):
        loader = DataLoader(
            torch.utils.data.Subset(ds, range(pos, len(todo))),
            batch_size=batch,
            num_workers=workers,
            pin_memory=device.startswith("cuda"),
            persistent_workers=False,
            prefetch_factor=4 if workers else None,
        )
        try:
            for item in loader:
                wav, local = item[0].to(device, non_blocking=True), item[1]
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.float16,
                        enabled=device.startswith("cuda"),
                    ),
                ):
                    hs = fe(wav)
                    x = torch.stack([hs[j] for j in layers], dim=1).to(torch.float16)  # B,L,T,D
                idx = todo[local.numpy()]
                feats[idx] = x.cpu().numpy()
                done[idx] = True
                if len(item) > 2:  # augmented view: remember which channel each clip got
                    for j, chain in zip(idx.tolist(), item[2], strict=True):
                        chains[j] = chain
                pos += len(idx)
                done_since += len(idx)
                if time.time() - last > 30:
                    rate = done_since / (time.time() - t0)
                    eta = (len(todo) - pos) / max(rate, 1e-6) / 60
                    print(
                        f"  {split}: {pos}/{len(todo)}  {rate:.0f} utt/s  ETA {eta:.0f} min  "
                        f"batch {batch}  {gpu_report()}",
                        flush=True,
                    )
                    save_progress()
                    last = time.time()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"  out of GPU memory -> batch size {batch}", flush=True)
            continue
    save_progress()
    print(f"{split}: done in {(time.time() - t0) / 60:.1f} min ({gpu_report()})", flush=True)
    return out


def load_rows(cfg: dict[str, Any], split: str) -> list[ManifestRow]:
    d = cfg["data"]
    files = sorted(resolve(Path(d["manifests"]).parent).glob(Path(d["manifests"]).name))
    ids = set(json.loads(resolve(d["splits"]).read_text(encoding="utf-8"))["splits"][split])
    rows = [r for f in files for r in read_manifest(f) if r.utt_id in ids]
    policy = DataPolicy.from_config(cfg)
    return enforce(rows, policy, purpose="train" if split == "train" else "eval")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(prepare_process()))
    device = setup_cuda(cfg.get("device"))
    if device == "cpu":
        print(
            "WARNING: no CUDA GPU visible to PyTorch; extraction will be very slow. "
            "Run scripts/setup/check_gpu.py."
        )
    c = cfg["cache"]
    for split in args.splits:
        sel = c.get(split, {})
        # an augmented split names its source split and view: {source: train, view: 1, ...}
        source, view = sel.get("source", split), sel.get("view")
        augment = None if view is None else {"view": int(view), **cfg["augment"]}
        rows = select_rows(
            load_rows(cfg, source),
            sel.get("max_bona"),
            sel.get("spoof_per_family"),
            int(cfg.get("seed", 0)) + 1000 * int(view or 0),
        )
        extract_split(
            cfg,
            split,
            rows,
            resolve(c["dir"]) / split,
            device,
            args.batch or int(c.get("batch", 24)),
            args.workers if args.workers is not None else int(c.get("workers", 6)),
            augment,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
