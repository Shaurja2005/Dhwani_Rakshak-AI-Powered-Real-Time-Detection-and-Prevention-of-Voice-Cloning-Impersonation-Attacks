"""Verify and extract the ASVspoof 5 tar parts (base-model setup, step 2).

    python scripts/setup/extract_asvspoof5.py                       # train + dev + eval
    python scripts/setup/extract_asvspoof5.py --parts T D           # train + dev only
    python scripts/setup/extract_asvspoof5.py --no-md5              # skip checksum (faster)

* Checks free disk space first (the extracted audio needs about as much as the tars).
* Verifies each tar against the md5 listed in the dataset's own README.txt.
* Extracts into ``data/asvspoof5/extracted/`` (flac_T, flac_D, flac_E_eval) and writes a
  ``<tar>.done`` marker, so re-running skips finished parts and resumes after a stop.
* The original tars are never modified or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def md5_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the publisher's md5 list
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def published_md5(readme: Path) -> dict[str, str]:
    out = {}
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([0-9a-f]{32})\s+\S*?([\w.]+\.tar)\s", line + " ")
        if m:
            out[m.group(2)] = m.group(1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "data/asvspoof5")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--parts", nargs="+", default=["T", "D", "E"], choices=["T", "D", "E"])
    ap.add_argument("--no-md5", action="store_true")
    args = ap.parse_args(argv)
    dest = args.dest or args.src / "extracted"
    dest.mkdir(parents=True, exist_ok=True)

    tars = sorted(p for p in args.src.glob("flac_*.tar") if p.name.split("_")[1] in args.parts)
    proto = args.src / "ASVspoof5_protocols.tar"
    todo = [t for t in tars if not (dest / f"{t.name}.done").exists()]
    need = sum(t.stat().st_size for t in todo) * 1.02
    free = shutil.disk_usage(dest).free
    print(
        f"{len(tars)} tar parts selected, {len(todo)} still to extract; "
        f"needs ~{need / 1e9:.0f} GB, free {free / 1e9:.0f} GB on {dest.anchor}"
    )
    if need > free:
        print("Not enough free space. Free some space or extract fewer parts (--parts T D).")
        return 2
    md5s = {} if args.no_md5 else published_md5(args.src / "README.txt")

    if proto.exists() and not (dest / "protocols.done").exists():
        with tarfile.open(proto) as tf:
            tf.extractall(dest / "protocols", filter="data")
        (dest / "protocols.done").write_text("ok", encoding="utf-8")
        print("protocols extracted")

    for i, t in enumerate(todo, 1):
        t0 = time.time()
        if md5s:
            expected = md5s.get(t.name)
            got = md5_of(t)
            if expected and got != expected:
                print(
                    f"[{i}/{len(todo)}] {t.name}: md5 MISMATCH ({got} != {expected}); "
                    "the file is corrupt or incomplete: re-download it. Skipping."
                )
                continue
            print(f"[{i}/{len(todo)}] {t.name}: md5 ok", flush=True)
        n = 0
        with tarfile.open(t) as tf:
            for member in tf:
                target = dest / member.name
                if member.isfile() and target.exists() and target.stat().st_size == member.size:
                    continue  # resumed run: already there
                tf.extract(member, dest, filter="data")
                n += 1
                if n % 5000 == 0:
                    print(f"    {n} files...", flush=True)
        (dest / f"{t.name}.done").write_text(f"{n} files", encoding="utf-8")
        print(f"[{i}/{len(todo)}] {t.name}: {n} files in {time.time() - t0:.0f} s", flush=True)
    print(f"done -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
