"""ml.data.splits — speaker-disjoint, generator-disjoint splits (B3-T10).

Rules:
* eval-only corpora always land in ``eval``.
* held-out generator families appear only in ``eval`` (leave-one-generator-out).
* speakers never cross splits; speaker assignment is a stable hash of (seed, speaker).

Usage:
    python -m ml.data.splits --holdout-families xtts,rvc --out splits/v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from ml.data.manifest import MANIFEST_GLOB, ManifestRow, read_manifests
from ml.data.registry import Registry, load_registry

SPLITS = ("train", "dev", "eval")


def _unit(seed: int, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def make_splits(
    rows: Iterable[ManifestRow],
    seed: int = 1337,
    dev_frac: float = 0.1,
    eval_frac: float = 0.1,
    holdout_families: Iterable[str] = (),
    registry: Registry | None = None,
) -> dict[str, list[str]]:
    reg = registry or load_registry()
    eval_only = {d.name for d in reg.datasets if d.eval_only}
    holdout = set(holdout_families)
    rows = list(rows)

    # Speakers pinned to eval by eval-only corpora or held-out generators.
    forced_eval = {
        r.speaker_id
        for r in rows
        if r.source_corpus in eval_only or (r.generator_family in holdout)
    }

    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for r in sorted(rows, key=lambda r: r.utt_id):
        if r.source_corpus in eval_only or r.generator_family in holdout:
            split = "eval"
        elif r.speaker_id in forced_eval:
            # Speaker already appears in eval: keep them there (disjointness).
            split = "eval"
        else:
            u = _unit(seed, r.speaker_id)
            split = "eval" if u < eval_frac else "dev" if u < eval_frac + dev_frac else "train"
        out[split].append(r.utt_id)
    return out


def check_disjoint(rows: Iterable[ManifestRow], splits: dict[str, list[str]]) -> None:
    by_id = {r.utt_id: r for r in rows}
    speakers = {s: {by_id[u].speaker_id for u in ids} for s, ids in splits.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and speakers[a] & speakers[b]:
                raise AssertionError(f"speakers leak between {a} and {b}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=MANIFEST_GLOB)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--holdout-families", default="")
    ap.add_argument("--out", type=Path, default=Path("splits/default.json"))
    args = ap.parse_args(argv)
    rows = read_manifests(args.manifests)
    holdout = [f for f in args.holdout_families.split(",") if f]
    splits = make_splits(rows, seed=args.seed, holdout_families=holdout)
    check_disjoint(rows, splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "holdout_families": holdout, "splits": splits}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({k: len(v) for k, v in splits.items()}, "->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
