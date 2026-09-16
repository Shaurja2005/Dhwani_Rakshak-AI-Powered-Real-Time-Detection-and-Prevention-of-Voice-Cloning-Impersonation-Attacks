"""ml.data.degrade — apply the channel simulator to a whole manifest (B3-T07).

Bona fide and spoof rows go through the exact same code path; the chain is
seeded from utt_id only (invariant I3).

Usage:
    python -m ml.data.degrade --in data/manifests/pool.jsonl \
        --out data/manifests/pool_tel.jsonl --copies 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from ml.data.channel.webrtc_chain import ChannelSimulator
from ml.data.manifest import ManifestRow, read_manifest, write_manifest


def degrade_row(
    row: ManifestRow, sim: ChannelSimulator, data_root: Path, out_dir: str, copy: int
) -> ManifestRow:
    x, sr = sf.read(str(data_root / row.path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    utt_id = f"{row.utt_id}__ch{copy}"
    y, out_sr, rec = sim.simulate(x, sr, utt_id)
    rel = f"{out_dir}/{utt_id}.flac"
    dest = data_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y, out_sr)
    return row.model_copy(
        update={
            "utt_id": utt_id,
            "path": rel,
            "codec_chain": [*row.codec_chain, *rec.codec_chain],
            "snr_db": rec.snr_db,
            "rir_id": rec.rir_id,
            "sample_rate": out_sr,
            "duration_s": len(y) / out_sr,
        }
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--audio-dir", default="degraded")
    args = ap.parse_args(argv)
    sim = ChannelSimulator()
    rows = [
        degrade_row(r, sim, args.data_root, args.audio_dir, c)
        for r in read_manifest(args.inp)
        for c in range(args.copies)
    ]
    print(f"wrote {write_manifest(rows, args.out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
