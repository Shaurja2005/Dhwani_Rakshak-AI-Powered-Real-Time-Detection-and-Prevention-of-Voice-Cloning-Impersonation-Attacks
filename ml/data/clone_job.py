"""ml.data.clone_job — batch cloning over consented bona fide utterances (B3-T06).

For each sampled bona fide reference utterance and each generator, run the
generator container and write a matched spoof manifest row with provenance.
Clean spoof audio is produced here; channel degradation happens afterwards
for both classes in ml/data/degrade.py (invariant I3).

Usage:
    python -m ml.data.clone_job --bona-fide data/manifests/cv_hi.jsonl \
        --texts data/prompts/hi.txt --generators xtts_v2,openvoice_v2 \
        --per-generator 200 --out data/manifests/clones_hi.jsonl [--execute]

Without --execute this is a dry run that prints the docker commands.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess  # noqa: S404 - fixed argv, no shell
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ml.data.consent import ConsentRegister, load_consent
from ml.data.manifest import ManifestRow, read_manifest, write_manifest

GENERATORS_PATH = Path(__file__).with_name("generators.yaml")


class ConsentViolation(RuntimeError):  # noqa: N818
    pass


@dataclass(frozen=True)
class Generator:
    name: str
    family: str
    kind: str  # tts | vc
    license: str
    commercial_use: bool

    @property
    def image(self) -> str:
        return f"vg-gen-{self.name}"


@dataclass(frozen=True)
class CloneTask:
    utt_id: str
    generator: Generator
    ref: ManifestRow
    text: str | None
    source: ManifestRow | None  # VC source utterance
    out_path: str
    seed: int


def load_generators(path: Path = GENERATORS_PATH) -> dict[str, Generator]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {g["name"]: Generator(**g) for g in raw["generators"]}


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:4], "big")


def plan(
    bona_fide: list[ManifestRow],
    generators: list[Generator],
    texts: list[str],
    per_generator: int,
    out_dir: str,
    consent: ConsentRegister,
    seed: int = 1337,
) -> list[CloneTask]:
    refused = [r.utt_id for r in bona_fide if not consent.allows(r.source_corpus, r.speaker_id)]
    if refused:
        raise ConsentViolation(
            f"{len(refused)} reference utterances lack consent (I9), e.g. {refused[:3]}. "
            "Add the corpus or speaker to ml/data/consent_register.yaml."
        )
    refs = [r for r in bona_fide if r.label == "bona_fide"]
    if not refs:
        raise ValueError("no bona fide reference utterances")
    rng = np.random.default_rng(seed)
    tasks: list[CloneTask] = []
    for g in generators:
        if g.kind == "tts" and not texts:
            raise ValueError(f"{g.name} is a TTS generator and needs --texts")
        picks = rng.choice(len(refs), size=min(per_generator, len(refs)), replace=False)
        for i in picks:
            ref = refs[int(i)]
            utt_id = f"{g.name}__{ref.utt_id}"
            text = texts[int(rng.integers(len(texts)))] if g.kind == "tts" else None
            source = None
            if g.kind == "vc":
                others = [r for r in refs if r.speaker_id != ref.speaker_id]
                if not others:
                    raise ValueError("voice conversion needs at least two speakers")
                source = others[int(rng.integers(len(others)))]
            tasks.append(
                CloneTask(
                    utt_id=utt_id,
                    generator=g,
                    ref=ref,
                    text=text,
                    source=source,
                    out_path=f"{out_dir}/{g.name}/{utt_id}.wav",
                    seed=_seed(str(seed), utt_id),
                )
            )
    return tasks


def docker_command(t: CloneTask, data_root: str = "data", gpus: bool = True) -> list[str]:
    root = str(Path(data_root).resolve())
    cmd = ["docker", "run", "--rm", "-v", f"{root}:/data"]
    if gpus:
        cmd += ["--gpus", "all"]
    cmd += [t.generator.image, "clone", "--ref-audio", f"/data/{t.ref.path}"]
    if t.text is not None:
        cmd += ["--text", t.text, "--lang", t.ref.language]
    if t.source is not None:
        cmd += ["--source-audio", f"/data/{t.source.path}"]
    cmd += ["--out", f"/data/{t.out_path}", "--seed", str(t.seed)]
    return cmd


def row_for(t: CloneTask, duration_s: float, sample_rate: int) -> ManifestRow:
    return ManifestRow(
        utt_id=t.utt_id,
        path=t.out_path,
        label="spoof",
        generator_family=t.generator.family,
        generator_name=t.generator.name,
        language=t.ref.language,
        accent=t.ref.accent,
        gender=t.ref.gender,
        # Cloned voice belongs to the reference speaker: keeps splits speaker-disjoint.
        speaker_id=t.ref.speaker_id,
        source_corpus="vg_indic_telephony",
        license=f"{t.ref.license} + {t.generator.license}",
        commercial_use=t.ref.commercial_use and t.generator.commercial_use,
        codec_chain=[],
        duration_s=duration_s,
        sample_rate=sample_rate,
    )


def execute(
    tasks: list[CloneTask], data_root: str = "data", gpus: bool = True
) -> list[ManifestRow]:
    import soundfile as sf

    rows: list[ManifestRow] = []
    for t in tasks:
        p = subprocess.run(docker_command(t, data_root, gpus), check=False)  # noqa: S603
        if p.returncode != 0:
            print(f"FAILED {t.utt_id} (exit {p.returncode})", file=sys.stderr)
            continue
        info = sf.info(str(Path(data_root) / t.out_path))
        rows.append(row_for(t, info.duration, info.samplerate))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bona-fide", type=Path, required=True)
    ap.add_argument("--texts", type=Path)
    ap.add_argument("--generators", required=True)
    ap.add_argument("--per-generator", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio-dir", default="clones")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    zoo = load_generators()
    gens = [zoo[n] for n in args.generators.split(",")]
    texts = (
        [s.strip() for s in args.texts.read_text(encoding="utf-8").splitlines() if s.strip()]
        if args.texts
        else []
    )
    tasks = plan(
        list(read_manifest(args.bona_fide)),
        gens,
        texts,
        args.per_generator,
        args.audio_dir,
        load_consent(),
        args.seed,
    )
    if not args.execute:
        for t in tasks[:10]:
            print(" ".join(docker_command(t, args.data_root, not args.no_gpu)))
        print(f"dry run: {len(tasks)} tasks planned. Re-run with --execute.")
        return 0
    rows = execute(tasks, args.data_root, not args.no_gpu)
    n = write_manifest(rows, args.out)
    print(f"wrote {n}/{len(tasks)} rows -> {args.out}")
    return 0 if n == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
