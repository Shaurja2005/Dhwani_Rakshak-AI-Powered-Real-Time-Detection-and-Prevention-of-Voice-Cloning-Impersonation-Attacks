"""ASVspoof 2019 LA protocol files -> manifests + splits (cross-dataset evaluation set).

    python -m ml.data.importers.asvspoof2019 --root "E:/datasets/asvspoof2019/LA"

``--root`` is the LA folder of the Kaggle / Edinburgh DataShare copy (the one containing
ASVspoof2019_LA_cm_protocols, ASVspoof2019_LA_train, _dev, _eval). Audio is used in place,
nothing is copied. Protocol lines: ``SPEAKER_ID AUDIO_FILE - SYSTEM_ID KEY``.
Licence: ODC-By 1.0 (README of the release).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf

from ml.data.manifest import ManifestRow, write_manifest

ROOT = Path(__file__).resolve().parents[3]
SPLITS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the LA folder")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    ap.add_argument("--splits", type=Path, default=ROOT / "splits/asvspoof2019_la.json")
    args = ap.parse_args(argv)
    split_ids = {}
    for split, (proto, folder) in SPLITS.items():
        p = next(args.root.rglob(proto), None)
        if p is None:
            print(f"{split}: {proto} not found under {args.root}")
            continue
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            spk, utt, _, system, key = f[:5]
            audio = args.root / folder / "flac" / f"{utt}.flac"
            if not audio.exists():
                continue
            spoof = key == "spoof"
            try:
                rel = Path(os.path.relpath(audio, args.data_root)).as_posix()
            except ValueError:  # another drive on Windows: keep the absolute path
                rel = audio.resolve().as_posix()
            rows.append(
                ManifestRow(
                    utt_id=f"asv19la_{utt}",
                    path=rel,
                    label="spoof" if spoof else "bona_fide",
                    generator_family=system if spoof else None,
                    generator_name=system if spoof else None,
                    language="en",
                    speaker_id=f"asv19la_{spk}",
                    source_corpus="asvspoof2019_la",
                    license="ODC-By-1.0",
                    commercial_use=False,
                    duration_s=max(float(sf.info(str(audio)).duration), 1e-3),
                    sample_rate=16000,
                )
            )
        n = write_manifest(rows, args.manifests / f"asvspoof2019_la_{split}.jsonl")
        split_ids[split] = [r.utt_id for r in rows]
        print(f"{split}: {n} rows")
    args.splits.parent.mkdir(parents=True, exist_ok=True)
    args.splits.write_text(
        json.dumps({"source": "ASVspoof 2019 LA official protocols", "splits": split_ids}),
        encoding="utf-8",
    )
    return 0 if split_ids else 2


if __name__ == "__main__":
    sys.exit(main())
