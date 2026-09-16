"""Speaker-embedder benchmark (B7-T01).

Compares embedders on a trial list, per language / accent, so the production
model is chosen on Indian-accented audio rather than VoxCeleb numbers.

Usage:
    python -m packages.vg_models.heads.head_d_speaker.benchmark \
        --manifest data/manifests/speaker_bench.jsonl --data-root data \
        --embedders ecapa,wespeaker,titanet,mfcc_stats --out docs/benchmarks/speaker_embedders.json

The manifest uses the B3 schema; only bona fide rows are used. Trials: every
utterance vs the centroid of the *other* utterances of the same speaker
(target) and vs other speakers' centroids of the same language (non-target).
Reports EER with a bootstrap 95% CI and embedding time per second of audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from packages.vg_models.heads.head_d_speaker.embedders import Embedder, build_embedder
from packages.vg_models.heads.head_d_speaker.scoring import eer_with_ci


def trials(utts: Sequence[tuple[str, str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """utts: (speaker, language, embedding). Returns (target scores, non-target scores)."""
    by_spk: dict[str, list[np.ndarray]] = defaultdict(list)
    lang_of: dict[str, str] = {}
    for spk, lang, e in utts:
        by_spk[spk].append(e)
        lang_of[spk] = lang
    tgt, non = [], []
    for spk, lang, e in utts:
        others = [x for x in by_spk[spk] if x is not e]
        if others:
            c = np.mean(others, axis=0)
            tgt.append(float(e @ c / (np.linalg.norm(c) + 1e-9)))
        for other, embs in by_spk.items():
            if other != spk and lang_of[other] == lang:
                c = np.mean(embs, axis=0)
                non.append(float(e @ c / (np.linalg.norm(c) + 1e-9)))
    return np.array(tgt), np.array(non)


def benchmark(
    rows: Sequence[Any], load: Callable[[Any], np.ndarray], embedder: Embedder
) -> dict[str, Any]:
    embs: list[tuple[str, str, np.ndarray]] = []
    audio_s, t_total = 0.0, 0.0
    for r in rows:
        x = load(r)
        t0 = time.perf_counter()
        embs.append((r.speaker_id, r.language, embedder.embed(x)))
        t_total += time.perf_counter() - t0
        audio_s += len(x) / 16000
    result: dict[str, Any] = {
        "embedder": embedder.name,
        "ms_per_audio_second": 1000 * t_total / max(audio_s, 1e-9),
    }
    langs = sorted({lang for _, lang, _ in embs})
    for lang in ["all", *langs]:
        subset = embs if lang == "all" else [u for u in embs if u[1] == lang]
        t, n = trials(subset)
        if len(t) >= 5 and len(n) >= 5:
            result[lang] = {**eer_with_ci(t, n), "n_target": len(t), "n_nontarget": len(n)}
    return result


def main(argv: list[str] | None = None) -> int:
    from ml.data.manifest import read_manifest
    from ml.training.dataset import soundfile_loader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--embedders", default="ecapa,wespeaker,titanet,mfcc_stats")
    ap.add_argument("--out", type=Path, default=Path("docs/benchmarks/speaker_embedders.json"))
    args = ap.parse_args(argv)
    rows = [r for r in read_manifest(args.manifest) if r.label == "bona_fide"]
    load = soundfile_loader(args.data_root)
    results = []
    for name in args.embedders.split(","):
        try:
            results.append(benchmark(rows, load, build_embedder(name)))
        except Exception as exc:  # noqa: BLE001 - report unavailable embedders instead of crashing
            results.append({"embedder": name, "error": f"{type(exc).__name__}: {exc}"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
