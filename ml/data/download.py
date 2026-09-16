"""ml.data.download — dataset downloaders + checksum verification (B3-T02, B3-T03).

Public files are streamed to ``data/raw/<dataset>/`` and verified by sha256.
EULA / gated / request-form corpora cannot be fetched automatically; for those
we print the access steps and verify files a human has placed on disk.

Usage:
    python -m ml.data.download --list
    python -m ml.data.download --dataset rirs_noises
    python -m ml.data.download --verify rirs_noises
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ml.data.registry import DatasetEntry, Registry, load_registry

DATA_ROOT = Path("data/raw")
_CHUNK = 1 << 20

ACCESS_HELP = {
    "eula": "Register and accept the EULA at {url}, then place files in {dest}.",
    "gated_hf": (
        "Run `huggingface-cli login`, accept the terms at {url}, then download into {dest}."
    ),
    "request_form": (
        "Submit the access request at {url} (approval can take days), "
        "then place files in {dest}."
    ),
}


class ChecksumMismatch(RuntimeError):  # noqa: N818
    pass


@dataclass
class VerifyResult:
    file: str
    status: str  # ok | missing | mismatch | unpinned
    sha256: str | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(entry: DatasetEntry, root: Path = DATA_ROOT) -> list[VerifyResult]:
    dest = root / entry.name
    results: list[VerifyResult] = []
    for f in entry.files:
        p = dest / f.name
        if not p.exists():
            results.append(VerifyResult(f.name, "missing"))
            continue
        digest = sha256_file(p)
        if f.sha256 is None:
            results.append(VerifyResult(f.name, "unpinned", digest))
        elif digest != f.sha256:
            results.append(VerifyResult(f.name, "mismatch", digest))
        else:
            results.append(VerifyResult(f.name, "ok", digest))
    return results


def _fetch(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https download URL: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(  # noqa: S310 - https enforced above
        url, headers={"User-Agent": "voiceguard-downloader"}
    )
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:  # noqa: S310
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
    tmp.replace(dest)


def download_dataset(entry: DatasetEntry, root: Path = DATA_ROOT) -> list[VerifyResult]:
    dest = root / entry.name
    if entry.access != "public":
        print(ACCESS_HELP[entry.access].format(url=entry.url, dest=dest))
        return verify_dataset(entry, root)
    if not entry.files:
        print(
            f"{entry.name}: no direct file list pinned yet. Download from {entry.url} into {dest}."
        )
        return []
    dest.mkdir(parents=True, exist_ok=True)
    for f in entry.files:
        p = dest / f.name
        if p.exists():
            continue
        print(f"fetching {f.url} -> {p}")
        _fetch(f.url, p)
    results = verify_dataset(entry, root)
    bad = [r for r in results if r.status == "mismatch"]
    if bad:
        raise ChecksumMismatch(f"{entry.name}: sha256 mismatch for {[r.file for r in bad]}")
    return results


def _print_list(reg: Registry) -> None:
    print(f"{'name':24} {'role':16} {'access':13} {'comm.':6} {'GB':>7}  license")
    for d in reg.datasets:
        print(
            f"{d.name:24} {d.role:16} {d.access:13} {str(d.commercial_use):6} "
            f"{d.size_gb:7.1f}  {d.license}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--dataset")
    g.add_argument("--verify")
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    args = ap.parse_args(argv)

    reg = load_registry()
    if args.list:
        _print_list(reg)
        return 0
    entry = reg.get(args.dataset or args.verify)
    results = (
        download_dataset(entry, args.root) if args.dataset else verify_dataset(entry, args.root)
    )
    for r in results:
        print(f"  {r.status:9} {r.file} {r.sha256 or ''}")
    if any(r.status == "unpinned" for r in results):
        print("Pin the sha256 values above in ml/data/registry.yaml after confirming the source.")
    return 1 if any(r.status in ("mismatch", "missing") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
