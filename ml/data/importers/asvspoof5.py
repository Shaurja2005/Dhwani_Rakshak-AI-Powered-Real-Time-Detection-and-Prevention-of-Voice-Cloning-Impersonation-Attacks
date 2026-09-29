"""ASVspoof 5 protocol files -> VoiceGuard manifests + official splits (base-model setup, step 3).

    python -m ml.data.importers.asvspoof5                         # reads data/asvspoof5/extracted

Writes
    data/manifests/asvspoof5_train.jsonl, _dev.jsonl, _eval.jsonl   (one ManifestRow per utterance)
    splits/asvspoof5.json                                            (official train / dev / eval)

Protocol columns (README.txt §3): SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q
CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP. The attack label (A01..A32) is the generator
family, so leave-one-generator-out works per attack; codec conditions (C01..C11, dev/eval)
go into ``codec_chain`` for the per-codec breakdown.

Licence as shipped in the dataset's LICENSE.txt: database ODC-By 1.0, bona fide speech
CC BY 4.0. ``commercial_use`` stays False until a human confirms it in ml/data/registry.yaml
(AGENTS §8) — the research lineage used for the base model is unaffected either way.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import soundfile as sf

from ml.data.manifest import ManifestRow, write_manifest

ROOT = Path(__file__).resolve().parents[3]
SPLITS = {
    "train": ("ASVspoof5.train.tsv", "flac_T"),
    "dev": ("ASVspoof5.dev.track_1.tsv", "flac_D"),
    "eval": ("ASVspoof5.eval.track_1.tsv", "flac_E_eval"),
}
LICENSE = "ODC-By-1.0 (bona fide: CC-BY-4.0)"


def parse_line(line: str) -> dict[str, str] | None:
    p = line.split()
    if len(p) < 9:
        return None
    return {
        "speaker": p[0],
        "utt": p[1],
        "gender": p[2],
        "codec": p[3],
        "codec_q": p[4],
        "attack_tag": p[6],
        "attack": p[7],
        "key": p[8],
    }


def to_row(r: dict[str, str], rel_audio: str, duration: float) -> ManifestRow:
    spoof = r["key"] == "spoof"
    codec = (
        []
        if r["codec"] == "-"
        else [f"codec:{r['codec']}" + (f"q{r['codec_q']}" if r["codec_q"] != "-" else "")]
    )
    return ManifestRow(
        utt_id=f"asv5_{r['utt']}",
        path=rel_audio,
        label="spoof" if spoof else "bona_fide",
        generator_family=r["attack"] if spoof else None,
        generator_name=(
            f"{r['attack']}/{r['attack_tag']}"
            if spoof and r["attack_tag"] != "-"
            else (r["attack"] if spoof else None)
        ),
        language="en",
        gender={"F": "female", "M": "male"}.get(r["gender"], "unknown"),
        speaker_id=f"asv5_{r['speaker']}",
        source_corpus="asvspoof5",
        license=LICENSE,
        commercial_use=False,
        codec_chain=codec,
        duration_s=max(duration, 1e-3),
        sample_rate=16000,
    )


def probe(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 - missing/corrupt file: reported, not fatal
        return -1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--extracted", type=Path, default=ROOT / "data/asvspoof5/extracted")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    ap.add_argument("--splits", type=Path, default=ROOT / "splits/asvspoof5.json")
    ap.add_argument("--only", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)

    proto_dir = args.extracted / "protocols"
    split_ids: dict[str, list[str]] = {}
    if args.splits.exists():
        split_ids = json.loads(args.splits.read_text(encoding="utf-8")).get("splits", {})
    for split in args.only:
        tsv, folder = SPLITS[split]
        tsv_path = next(proto_dir.rglob(tsv), None)
        if tsv_path is None:
            print(
                f"{split}: protocol {tsv} not found under {proto_dir} (run extract_asvspoof5.py first)"
            )
            return 2
        recs = [r for r in map(parse_line, tsv_path.read_text(encoding="utf-8").splitlines()) if r]
        audio = [args.extracted / folder / f"{r['utt']}.flac" for r in recs]
        print(f"{split}: {len(recs)} protocol rows; probing audio durations...", flush=True)
        with cf.ThreadPoolExecutor(args.threads) as pool:
            durations = list(pool.map(probe, audio, chunksize=256))
        rows, missing = [], 0
        for r, a, d in zip(recs, audio, durations, strict=True):
            if d < 0:
                missing += 1
                continue
            rows.append(to_row(r, a.relative_to(args.data_root).as_posix(), d))
        n = write_manifest(rows, args.manifests / f"asvspoof5_{split}.jsonl")
        split_ids[split] = [r.utt_id for r in rows]
        bona = sum(r.label == "bona_fide" for r in rows)
        fams = sorted({r.generator_family for r in rows if r.generator_family})
        print(
            f"{split}: wrote {n} rows ({bona} bona fide, {n - bona} spoof, attacks {fams[0]}..{fams[-1]})"
            + (f"; {missing} audio files MISSING (not extracted?)" if missing else "")
        )
    args.splits.parent.mkdir(parents=True, exist_ok=True)
    args.splits.write_text(
        json.dumps({"source": "ASVspoof 5 official protocols", "splits": split_ids}),
        encoding="utf-8",
    )
    print(f"splits -> {args.splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
