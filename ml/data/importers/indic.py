"""IndicSynth (spoof) + Kathbath (bona fide) -> one language's manifest + splits (B3-T13).

    python -m ml.data.importers.indic --language bengali                 # 12k + 12k
    python -m ml.data.importers.indic --language odia --spoof 6000 --bona 6000

Writes, for language ``<iso>``:

    data/indic/<iso>/{indicsynth,kathbath}/*.flac     16 kHz mono FLAC (only what is sampled)
    data/manifests/indic_<iso>.jsonl                   manifest rows (schema: ml/data/manifest.py)
    splits/indic_<iso>.json                            speaker-disjoint train / dev / eval
    data/indic/<iso>/import_report.json                counts, speakers, original sample rates

**Sampling without downloading 845 GB.** Both corpora are parquet on the Hugging Face
Hub. The importer reads each shard's footer (a few KB), lists its row groups, shuffles
all row groups of the language with a fixed seed and downloads only the chosen ones
(HTTP range reads) until the target count is reached. IndicSynth stores rows sorted
by generator and gender, so taking "the first N rows" would give one generator only;
random row groups spread the sample over every generator, speaker and gender.

**Speakers.** IndicSynth's ``Target Speaker ID`` is a Kathbath speaker, so a person's
cloned and genuine voices share one id (``kb_<iso>_<id>``) and always land in the same
split. Voice-conversion rows are split by their *target* speaker; the VC *source*
speaker can sit in another split (reported in import_report.json, not hidden).

**Resumable.** The plan (chosen row groups, pinned to the dataset revisions) is saved
first; every finished row group writes a part file. Re-running skips finished parts.

Kathbath is gated on the Hub: accept its terms at huggingface.co/datasets/ai4bharat/Kathbath
and log in once (``huggingface-cli login``) before importing. Licences: IndicSynth
CC-BY-NC-4.0 (research lineage), Kathbath CC-BY-4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess  # noqa: S404 - fixed argv (ffmpeg), no shell
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.data.manifest import ManifestRow, write_manifest

ROOT = Path(__file__).resolve().parents[3]
SR = 16000
INDICSYNTH = "vdivyasharma/IndicSynth"
KATHBATH = "ai4bharat/Kathbath"

# language -> (IndicSynth config, Kathbath config, ISO 639 code used in manifests)
LANGS: dict[str, tuple[str, str, str]] = {
    "bengali": ("Bengali", "bengali", "bn"),
    "gujarati": ("Gujarati", "gujarati", "gu"),
    "hindi": ("Hindi", "hindi", "hi"),
    "kannada": ("Kannada", "kannada", "kn"),
    "malayalam": ("Malayalam", "malayalam", "ml"),
    "marathi": ("Marathi", "marathi", "mr"),
    "odia": ("Odia", "odia", "or"),
    "punjabi": ("Punjabi", "punjabi", "pa"),
    "sanskrit": ("Sanskrit", "sanskrit", "sa"),
    "tamil": ("Tamil", "tamil", "ta"),
    "telugu": ("Telugu", "telugu", "te"),
    "urdu": ("Urdu", "urdu", "ur"),
}
SOURCES = {
    "indicsynth": {
        "repo": INDICSYNTH,
        "license": "CC-BY-NC-4.0",
        "columns": [
            "audio",
            "Generative Model",
            "Target Speaker ID",
            "Source Speaker_ID",
            "Gender",
        ],
    },
    "kathbath": {
        "repo": KATHBATH,
        "license": "CC-BY-4.0",
        "columns": ["audio_filepath", "fname", "speaker_id", "gender"],
    },
}
MIN_SECONDS = 1.0


def language(name: str) -> tuple[str, str, str]:
    key = name.strip().lower()
    for k, v in LANGS.items():
        if key in (k, v[2]):
            return v
    raise SystemExit(f"unknown language {name!r}; choose one of {', '.join(LANGS)}")


# ---------------------------------------------------------------- planning
@dataclass(frozen=True)
class Unit:
    """One parquet row group: the unit of download, progress and resume."""

    source: str
    path: str
    rg: int
    rows: int
    nbytes: int


def plan_units(units: list[Unit], target: int, seed: int) -> list[Unit]:
    """Random row groups (fixed seed) until ``target`` rows: spreads over the whole corpus."""
    order = np.random.default_rng(seed).permutation(len(units))
    out, n = [], 0
    for i in order:
        if n >= target:
            break
        out.append(units[int(i)])
        n += units[int(i)].rows
    return out


def rows_to_keep(unit: Unit, cap: int, seed: int) -> list[int]:
    """All rows of a normal row group; a random ``cap`` subset of an unusually large one."""
    if unit.rows <= cap:
        return list(range(unit.rows))
    rng = np.random.default_rng(
        (seed, unit.rg, int(hashlib.sha256(unit.path.encode()).hexdigest()[:8], 16))
    )
    return sorted(rng.choice(unit.rows, size=cap, replace=False).tolist())


def _hub() -> tuple[Any, Any, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - environment
        raise SystemExit(
            "pyarrow is missing: run  pip install -r requirements/train-gpu.txt"
        ) from e
    from huggingface_hub import HfApi, HfFileSystem

    return pq, HfApi(), HfFileSystem()


def _retry(fn: Any, what: str, attempts: int = 4) -> Any:  # noqa: ANN401
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network: retry, then explain
            if _is_gated(e):
                raise
            if k == attempts - 1:
                raise RuntimeError(f"{what}: {e}") from e
            time.sleep(2 * 2**k)
    return None


def _is_gated(e: BaseException) -> bool:
    text = f"{type(e).__name__} {e}".lower()
    return any(s in text for s in ("gated", "401", "403", "awaiting a review", "access to dataset"))


def list_units(source: str, config: str, revision: str, workers: int = 8) -> list[Unit]:
    pq, api, fs = _hub()
    repo = SOURCES[source]["repo"]
    files = [
        f
        for f in api.list_repo_tree(
            repo, path_in_repo=config, repo_type="dataset", recursive=True, revision=revision
        )
        if getattr(f, "path", "").endswith(".parquet")
    ]
    if source == "indicsynth":
        files = [f for f in files if "/train" in f.path]

    def footer(path: str) -> list[Unit]:
        def read() -> list[Unit]:
            with fs.open(f"datasets/{repo}@{revision}/{path}", "rb", block_size=1 << 16) as fh:
                md = pq.ParquetFile(fh).metadata
                return [
                    Unit(source, path, i, md.row_group(i).num_rows, md.row_group(i).total_byte_size)
                    for i in range(md.num_row_groups)
                ]

        return _retry(read, f"footer of {path}")

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(footer, sorted(f.path for f in files)))
    return [u for p in parts for u in p]


# ---------------------------------------------------------------- audio
def decode(blob: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - e.g. mp3/m4a: let ffmpeg decode it
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise
        p = subprocess.run(  # noqa: S603
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(SR),
                "pipe:1",
            ],
            input=blob,
            capture_output=True,
            check=True,
        )
        return np.frombuffer(p.stdout, dtype="<f4").copy(), SR
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    from scipy.signal import resample_poly

    g = np.gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def _gender(v: Any) -> str:  # noqa: ANN401
    s = str(v or "").strip().lower()
    return {"f": "female", "female": "female", "m": "male", "male": "male"}.get(s, "unknown")


def _speaker(iso: str, v: Any) -> str:  # noqa: ANN401
    try:
        return f"kb_{iso}_{int(float(v))}"
    except (TypeError, ValueError):
        return f"kb_{iso}_{str(v).strip() or 'unknown'}"


def convert_record(
    rec: dict[str, Any], source: str, iso: str, unit: Unit, i: int, data_root: Path
) -> tuple[ManifestRow, dict[str, Any]] | None:
    """One hub record -> FLAC on disk + manifest row + provenance (None if unusable)."""
    import soundfile as sf

    audio = rec["audio"] if source == "indicsynth" else rec["audio_filepath"]
    blob = (audio or {}).get("bytes")
    if not blob:
        return None
    x, sr = decode(blob)
    x = to_16k(x, sr)
    if len(x) < MIN_SECONDS * SR or not np.isfinite(x).all():
        return None
    if source == "indicsynth":
        key = hashlib.sha256(f"{unit.path}:{unit.rg}:{i}".encode()).hexdigest()[:12]
        utt = f"isyn_{iso}_{key}"
        family = str(rec["Generative Model"]).strip().lower()
        speaker = _speaker(iso, rec["Target Speaker ID"])
        src = rec.get("Source Speaker_ID")
        prov = {"source_speaker": _speaker(iso, src) if src is not None else None}
    else:
        utt = f"kb_{iso}_{Path(str(rec['fname'])).stem}"
        family, speaker, prov = None, _speaker(iso, rec["speaker_id"]), {}
    rel = Path("indic") / iso / source / f"{utt}.flac"
    out = data_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x)))
    sf.write(out, x / peak * 0.99 if peak > 1.0 else x, SR, subtype="PCM_16")
    row = ManifestRow(
        utt_id=utt,
        path=rel.as_posix(),
        label="spoof" if family else "bona_fide",
        generator_family=family,
        generator_name=f"indicsynth/{family}" if family else None,
        language=iso,
        gender=_gender(rec.get("Gender") if source == "indicsynth" else rec.get("gender")),  # type: ignore[arg-type]
        speaker_id=speaker,
        source_corpus=source,
        license=SOURCES[source]["license"],
        commercial_use=False,
        duration_s=round(len(x) / SR, 3),
        sample_rate=SR,
    )
    return row, {**prov, "original_sr": sr}


# ---------------------------------------------------------------- splits
def assign_splits(
    rows: Iterable[ManifestRow], fracs: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Speaker-disjoint and clip-balanced *per label*.

    Speakers are visited in a seeded-hash order; each goes to the split whose remaining
    bona fide / spoof quota it fills best (dev and eval first, the rest to train). With
    only tens of speakers per language this keeps both classes in every split far more
    reliably than a per-speaker coin flip.
    """
    rows = list(rows)
    per_spk: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        per_spk[r.speaker_id][r.label] += 1
    total = Counter(r.label for r in rows)
    quota = {s: {lab: fracs[s] * total[lab] for lab in total} for s in fracs if s != "train"}
    have = {s: Counter() for s in quota}
    order = sorted(per_spk, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    where: dict[str, str] = {}
    for spk in order:
        c = per_spk[spk]
        best, gain = "train", 0.0
        for s in quota:
            g = sum(
                min(c[lab], max(0.0, quota[s][lab] - have[s][lab])) / max(total[lab], 1)
                for lab in total
            )
            over = sum(max(0.0, have[s][lab] + c[lab] - 1.5 * quota[s][lab]) for lab in total)
            if over == 0 and g > gain:
                best, gain = s, g
        where[spk] = best
        if best != "train":
            have[best].update(c)
    out: dict[str, list[str]] = {s: [] for s in ("train", "dev", "eval")}
    for r in sorted(rows, key=lambda r: r.utt_id):
        out[where[r.speaker_id]].append(r.utt_id)
    return out


# ---------------------------------------------------------------- import
def import_language(
    lang: str,
    n_spoof: int,
    n_bona: int,
    seed: int = 1337,
    fracs: dict[str, float] | None = None,
    data_root: Path = ROOT / "data",
    manifests: Path = ROOT / "data/manifests",
    splits_dir: Path = ROOT / "splits",
    workers: int = 4,
) -> dict[str, Any]:
    synth_cfg, kb_cfg, iso = language(lang)
    fracs = fracs or {"train": 0.7, "dev": 0.1, "eval": 0.2}
    work = data_root / "indic" / iso
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    free = shutil.disk_usage(work).free
    if free < 15e9:
        raise SystemExit(f"only {free / 1e9:.0f} GB free on {work.anchor}; need ~15 GB")

    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (plan["targets"], plan["seed"]) != ({"indicsynth": n_spoof, "kathbath": n_bona}, seed):
            raise SystemExit(
                f"{plan_path} was made with other targets/seed {plan['targets']}, {plan['seed']}; "
                "delete it (and data/indic/<iso>/parts) to re-plan"
            )
    else:
        _, api, _ = _hub()
        plan = {
            "language": lang,
            "iso": iso,
            "seed": seed,
            "fracs": fracs,
            "targets": {"indicsynth": n_spoof, "kathbath": n_bona},
            "revisions": {},
            "units": {},
        }
        for source, config, target in (
            ("indicsynth", synth_cfg, n_spoof),
            ("kathbath", kb_cfg, n_bona),
        ):
            repo = SOURCES[source]["repo"]
            try:
                rev = api.dataset_info(repo).sha
                units = list_units(source, config, rev)
            except Exception as e:  # noqa: BLE001
                if _is_gated(e):
                    raise SystemExit(
                        f"{repo} is gated: open https://huggingface.co/datasets/{repo}, accept the "
                        "terms, then run  huggingface-cli login  once, and re-run this command"
                    ) from e
                raise
            chosen = plan_units(units, target, seed)
            plan["revisions"][source] = rev
            plan["units"][source] = [asdict(u) for u in chosen]
            mb = sum(u.nbytes for u in chosen) / 1e6
            print(
                f"{source}: {len(units)} row groups in {config}; sampling {len(chosen)} "
                f"(~{sum(u.rows for u in chosen)} rows, ~{mb:.0f} MB to download)",
                flush=True,
            )
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")

    parts = work / "parts"
    parts.mkdir(exist_ok=True)
    todo = [
        (source, k, Unit(**u))
        for source in ("indicsynth", "kathbath")
        for k, u in enumerate(plan["units"][source])
        if not (parts / f"{source}_{k:05d}.jsonl").exists()
    ]
    n_units = sum(len(v) for v in plan["units"].values())
    print(f"{iso}: {n_units - len(todo)}/{n_units} row groups already imported", flush=True)
    pq, _, fs = _hub() if todo else (None, None, None)

    def work_unit(source: str, k: int, u: Unit) -> int:
        repo, rev = SOURCES[source]["repo"], plan["revisions"][source]
        target = plan["targets"][source]
        cap = max(50, target // 60)

        def read() -> list[dict[str, Any]]:
            with fs.open(f"datasets/{repo}@{rev}/{u.path}", "rb", block_size=8 << 20) as fh:
                pf = pq.ParquetFile(fh)
                return pf.read_row_group(u.rg, columns=SOURCES[source]["columns"]).to_pylist()

        recs = _retry(read, f"{u.path} row group {u.rg}")
        lines = []
        for i in rows_to_keep(u, cap, seed):
            got = convert_record(recs[i], source, iso, u, i, data_root)
            if got is not None:
                lines.append(json.dumps({"row": got[0].model_dump(), "prov": got[1]}))
        tmp = parts / f"{source}_{k:05d}.tmp"
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp.replace(parts / f"{source}_{k:05d}.jsonl")
        return len(lines)

    t0, done_rows = time.time(), 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = [ex.submit(work_unit, s, k, u) for s, k, u in todo]
        for j, f in enumerate(as_completed(futs), 1):
            done_rows += f.result()
            if j % 5 == 0 or j == len(futs):
                rate = done_rows / max(time.time() - t0, 1e-6)
                eta = (len(futs) - j) * (time.time() - t0) / j / 60
                print(
                    f"  {j}/{len(futs)} row groups, {done_rows} clips, {rate:.0f} clips/s, "
                    f"ETA {eta:.0f} min",
                    flush=True,
                )
    return finalize(plan, parts, manifests, splits_dir, work)


def finalize(
    plan: dict[str, Any], parts: Path, manifests: Path, splits_dir: Path, work: Path
) -> dict[str, Any]:
    iso = plan["iso"]
    rows: list[ManifestRow] = []
    prov: dict[str, dict[str, Any]] = {}
    for p in sorted(parts.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows.append(ManifestRow.model_validate(d["row"]))
                prov[rows[-1].utt_id] = d["prov"]
    rows = list({r.utt_id: r for r in rows}.values())  # a re-imported part must not duplicate
    splits = assign_splits(rows, plan["fracs"], plan["seed"])
    write_manifest(sorted(rows, key=lambda r: r.utt_id), manifests / f"indic_{iso}.jsonl")
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / f"indic_{iso}.json").write_text(
        json.dumps(
            {
                "source": f"IndicSynth + Kathbath ({plan['language']}), speaker-disjoint",
                "revisions": plan["revisions"],
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    split_of = {u: s for s, ids in splits.items() for u in ids}
    by = {r.utt_id: r for r in rows}
    report: dict[str, Any] = {"language": plan["language"], "iso": iso, "n": len(rows)}
    for s, ids in splits.items():
        rs = [by[u] for u in ids]
        report[s] = {
            "bona_fide": sum(r.label == "bona_fide" for r in rs),
            "spoof": sum(r.label == "spoof" for r in rs),
            "speakers": len({r.speaker_id for r in rs}),
            "generators": dict(Counter(r.generator_family for r in rs if r.generator_family)),
            "gender": dict(Counter(r.gender for r in rs)),
        }
    report["original_sample_rates"] = {
        src: dict(
            Counter(str(prov[r.utt_id].get("original_sr")) for r in rows if r.source_corpus == src)
        )
        for src in SOURCES
    }
    report["mean_duration_s"] = {
        src: round(float(np.mean([r.duration_s for r in rows if r.source_corpus == src] or [0])), 2)
        for src in SOURCES
    }
    spk_split = {r.speaker_id: split_of[r.utt_id] for r in rows}
    vc_cross = sum(
        1
        for r in rows
        if (src := prov[r.utt_id].get("source_speaker")) is not None
        and src in spk_split
        and spk_split[src] != split_of[r.utt_id]
    )
    report["vc_rows_whose_source_speaker_is_in_another_split"] = vc_cross
    (work / "import_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    rates = report["original_sample_rates"]
    if set(rates["indicsynth"]) != set(rates["kathbath"]):
        print(
            "NOTE: fake and real audio arrive at different sample rates "
            f"({rates}). Everything is resampled to 16 kHz, but bandwidth can still differ; "
            "the augmented view (narrowband codecs for both classes) and the per-language "
            "eval are there to catch a model that learns the recording source instead of the fake."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--language", required=True, help=", ".join(LANGS))
    ap.add_argument("--spoof", type=int, default=12000)
    ap.add_argument("--bona", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    import_language(args.language, args.spoof, args.bona, args.seed, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
