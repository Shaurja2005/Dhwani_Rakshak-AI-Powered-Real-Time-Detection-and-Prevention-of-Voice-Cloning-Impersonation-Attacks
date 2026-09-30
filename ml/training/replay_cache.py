"""Compact replay caches for continual per-language training (B4-T13).

    python -m ml.training.replay_cache --out data/features/xlsr300m_L5-9/replay/asv5_train \
        --from data/features/xlsr300m_L5-9/train data/features/xlsr300m_L5-9/train_aug1 \
        --bona 2000 --spoof 4000

A replay cache is a small, balanced copy of already-computed features (same layout as a
feature_cache split, so the trainer reads it like any other split). Each finished
language keeps one, so later languages can be trained with a reminder of every earlier
one, and the language's big cache can then be deleted by hand. Copying is CPU/disk only.

Selection: ``--bona`` bona fide clips at random, ``--spoof`` spoof clips split equally
across attack families (families with fewer clips give what they have; the rest is
filled at random). Seeded, so rebuilding gives the same replay.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRY = ("layers", "frames", "dim", "seconds", "frontend", "frontend_revision")
PER_ROW = ("utt_ids", "labels", "families", "speakers", "corpora", "languages")


def _complete(d: Path) -> bool:
    return (d / "done.npy").exists() and bool(np.load(d / "done.npy").all())


def select(
    labels: np.ndarray, families: list[str], n_bona: int, n_spoof: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bona = np.flatnonzero(labels == 1)
    pick = [rng.choice(bona, size=min(n_bona, len(bona)), replace=False)]
    fams: dict[str, list[int]] = {}
    for i in np.flatnonzero(labels == 0):
        fams.setdefault(families[i], []).append(int(i))
    left, taken = n_spoof, []
    for k, (_, idx) in enumerate(sorted(fams.items())):
        share = left // (len(fams) - k)
        got = rng.choice(idx, size=min(share, len(idx)), replace=False)
        taken.append(got)
        left -= len(got)
    rest = np.setdiff1d(np.flatnonzero(labels == 0), np.concatenate(taken) if taken else [])
    if left > 0 and len(rest):
        taken.append(rng.choice(rest, size=min(left, len(rest)), replace=False))
    return np.sort(np.concatenate(pick + taken).astype(np.int64))


def build_replay(sources: list[Path], out: Path, n_bona: int, n_spoof: int, seed: int = 0) -> Path:
    if _complete(out):
        print(f"{out}: replay cache already built")
        return out
    metas = []
    for s in sources:
        if not _complete(s):
            raise SystemExit(f"{s}: source cache missing or incomplete")
        metas.append(json.loads((s / "meta.json").read_text(encoding="utf-8")))
    first = metas[0]
    for s, m in zip(sources[1:], metas[1:], strict=True):
        if [m.get(k) for k in GEOMETRY] != [first.get(k) for k in GEOMETRY]:
            raise SystemExit(f"{s}: different front-end/geometry than {sources[0]}")
    src_of = np.concatenate([np.full(m["n"], k) for k, m in enumerate(metas)])
    local = np.concatenate([np.arange(m["n"]) for m in metas])
    labels = np.concatenate([np.asarray(m["labels"]) for m in metas])
    families = [f for m in metas for f in m["families"]]
    idx = select(labels, families, n_bona, n_spoof, seed)

    shape = (len(idx), len(first["layers"]), first["frames"], first["dim"])
    need = float(np.prod(shape)) * 2
    out.parent.mkdir(parents=True, exist_ok=True)
    if need > shutil.disk_usage(out.parent).free * 0.95:
        raise SystemExit(f"not enough disk space for the replay cache ({need / 1e9:.1f} GB)")
    if out.exists():
        shutil.rmtree(out)  # an interrupted copy: rebuild (it only takes minutes)
    out.mkdir(parents=True)
    feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16, shape=shape)
    chains: list[str | None] = []
    src_chains = [
        (
            json.loads((s / "chains.json").read_text(encoding="utf-8"))
            if (s / "chains.json").exists()
            else None
        )
        for s in sources
    ]
    for k, s in enumerate(sources):  # sequential per source: fast memmap reads
        mine = np.flatnonzero(src_of[idx] == k)
        if not len(mine):
            continue
        src = np.load(s / "feats.npy", mmap_mode="r")
        for a in range(0, len(mine), 256):
            rows = mine[a : a + 256]
            feats[rows] = src[local[idx[rows]]]
    for j in idx:
        c = src_chains[src_of[j]]
        chains.append(c[local[j]] if c is not None else None)
    feats.flush()
    meta: dict[str, Any] = {k: first.get(k) for k in GEOMETRY}
    meta |= {"split": out.name, "n": len(idx), "replay_of": [str(s) for s in sources]}
    for key in PER_ROW:
        meta[key] = [metas[src_of[j]][key][local[j]] for j in idx]
    augs = [m["augmentation"] for m in metas if "augmentation" in m]
    if augs:
        meta["augmentation_sources"] = augs
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "chains.json").write_text(json.dumps(chains), encoding="utf-8")
    np.save(out / "done.npy", np.ones(len(idx), dtype=bool))
    n_b = int(np.sum(np.asarray(meta["labels"]) == 1))
    print(f"{out}: {len(idx)} clips ({n_b} bona fide, {len(idx) - n_b} spoof), {need / 1e9:.1f} GB")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="sources", type=Path, nargs="+", required=True)
    ap.add_argument("--bona", type=int, required=True)
    ap.add_argument("--spoof", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    build_replay(args.sources, args.out, args.bona, args.spoof, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
