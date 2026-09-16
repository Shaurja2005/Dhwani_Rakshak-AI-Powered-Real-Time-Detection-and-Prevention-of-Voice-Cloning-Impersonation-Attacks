"""ml.data.manifest — unified utterance manifest schema, IO and corpus report (B3-T04).

One JSONL row per utterance under ``data/manifests/*.jsonl``. Every split,
report and fairness breakdown reads from this.

Usage:
    python ml/data/manifest.py --report [--manifests "data/manifests/*.jsonl"]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

MANIFEST_GLOB = "data/manifests/*.jsonl"

Label = Literal["bona_fide", "spoof"]


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utt_id: str
    path: str
    label: Label
    generator_family: str | None = None  # None for bona fide
    generator_name: str | None = None
    language: str
    accent: str | None = None
    gender: Literal["female", "male", "other", "unknown"] = "unknown"
    speaker_id: str
    source_corpus: str
    license: str
    commercial_use: StrictBool
    codec_chain: list[str] = Field(default_factory=list)
    snr_db: float | None = None
    rir_id: str | None = None
    duration_s: float = Field(gt=0)
    sample_rate: int = Field(gt=0)

    def model_post_init(self, __context: object) -> None:
        if self.label == "spoof" and not self.generator_family:
            raise ValueError(f"{self.utt_id}: spoof rows must record generator_family")
        if self.label == "bona_fide" and self.generator_family:
            raise ValueError(f"{self.utt_id}: bona fide rows must not have generator_family")


def write_manifest(rows: Iterable[ManifestRow], path: Path | str) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            n += 1
    return n


def read_manifest(path: Path | str) -> Iterator[ManifestRow]:
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield ManifestRow.model_validate(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: invalid manifest row: {e}") from e


def read_manifests(pattern: str = MANIFEST_GLOB) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for p in sorted(glob.glob(pattern)):
        rows.extend(read_manifest(p))
    ids = [r.utt_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate utt_id across manifests")
    return rows


def codec_key(row: ManifestRow) -> str:
    codecs = [c for c in row.codec_chain if c.startswith("codec:")]
    return "+".join(c.removeprefix("codec:") for c in codecs) or "clean"


def corpus_report(rows: list[ManifestRow]) -> str:
    """Hours by language × generator family × codec."""
    hours: dict[tuple[str, str, str], float] = defaultdict(float)
    for r in rows:
        family = r.generator_family if r.label == "spoof" else "bona_fide"
        hours[(r.language, family or "?", codec_key(r))] += r.duration_s / 3600
    if not hours:
        return "No manifest rows found."
    lines = [f"{'language':10} {'family':22} {'codec':22} {'hours':>9}", "-" * 66]
    for (lang, fam, codec), h in sorted(hours.items()):
        lines.append(f"{lang:10} {fam:22} {codec:22} {h:9.3f}")
    total = sum(hours.values())
    spoof = sum(r.duration_s for r in rows if r.label == "spoof") / 3600
    lines.append("-" * 66)
    lines.append(
        f"total {total:.3f} h | spoof {spoof:.3f} h | bona fide {total - spoof:.3f} h "
        f"| {len(rows)} utterances"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Manifest tools")
    ap.add_argument("--report", action="store_true", help="print corpus report")
    ap.add_argument("--manifests", default=MANIFEST_GLOB)
    args = ap.parse_args(argv)
    if args.report:
        rows = read_manifests(args.manifests)
        print(corpus_report(rows))
        labels = {r.label for r in rows}
        if len(rows) < 50 or len(labels) < 2:
            print("symmetry check: SKIPPED (needs >= 50 rows with both labels)")
        else:
            from ml.data.channel.symmetry import check_manifest_symmetry

            res = check_manifest_symmetry(rows)
            print(f"symmetry check: {res}")
            return 0 if res.passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
