"""Train the detector on Indic languages, one at a time (continual fine-tuning, B4-T13).

    python scripts/train_indic_language.py status                 # every language so far
    python scripts/train_indic_language.py bengali status         # one language: done / next
    python scripts/train_indic_language.py bengali import         # 1. sample + download (~20-40 min)
    python scripts/train_indic_language.py bengali cache          # 2. XLS-R features (~40-70 min)
    python scripts/train_indic_language.py bengali train          # 3. fine-tune (~30-60 min)
    python scripts/train_indic_language.py bengali replay         # 4. keep a replay sample (~5 min)
    python scripts/train_indic_language.py bengali eval           # 5. before/after, all languages (~30-60 min)
    python scripts/train_indic_language.py compare                # 6. per-language table, forgetting check
    python scripts/train_indic_language.py bengali all            # 1-5, skipping finished steps
    python scripts/train_indic_language.py queue hindi marathi    # several languages in a row

The first language you run becomes step 01, the next step 02, and so on
(runs/indic/ledger.json). Language k starts from language k-1's model and trains on its
own data (clean + augmented) plus replay samples of ASVspoof 5 and every earlier
language. Stop any step with Ctrl+C; the same command continues where it stopped.
Guide: TRAINING_INDIC.md
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data.importers.indic import LANGS, language  # noqa: E402
from scripts.train_base_model import ROOT, cache_done, run, train_done  # noqa: E402

CONFIG = ROOT / "ml/training/configs/indic_continual.yaml"
LEDGER = ROOT / "runs/indic/ledger.json"
CFG_DIR = ROOT / "runs/indic/configs"
RUNS_DIR = ROOT / "docs/benchmarks/runs"
STEPS = ("import", "cache", "train", "replay", "eval")


# ---------------------------------------------------------------- ledger
def load_cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_ledger() -> dict[str, Any]:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"order": []}


def entry(lang: str, create: bool = False) -> dict[str, Any] | None:
    _, _, iso = language(lang)
    led = load_ledger()
    for e in led["order"]:
        if e["iso"] == iso:
            return e
    if not create:
        return None
    k = len(led["order"]) + 1
    name = next(n for n, v in LANGS.items() if v[2] == iso)
    e = {
        "language": name,
        "iso": iso,
        "index": k,
        "run": f"runs/indic_{k:02d}_{iso}",
        "version": f"0.3.{k}",
    }
    led["order"].append(e)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")
    print(f"{name} is language step {k:02d} (model {after_version(e)})")
    return e


def model_version(c: dict[str, Any], version: str) -> str:
    """The name HeadAModel gives itself (``A@<front-end>-<back-end>-v<version>``)."""
    from ml.training.train_head_a_cached import SHORT_NAMES

    fe = str(c["frontend"])
    return f"A@{SHORT_NAMES.get(fe, fe.split('/')[-1])}-{c['backend']}-v{version}"


def after_version(e: dict[str, Any]) -> str:
    return model_version(load_cfg(), e["version"])


def earlier(e: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in load_ledger()["order"] if x["index"] < e["index"]]


def init_model(e: dict[str, Any], cfg: dict[str, Any]) -> tuple[Path, str]:
    """(checkpoint, model version) the language starts from."""
    prev = earlier(e)
    if prev:
        p = prev[-1]
        return ROOT / p["run"] / "best.pt", after_version(p)
    ck = ROOT / cfg["init_from"]
    cfg_path = ck.parent / "config.yaml"  # written by the trainer next to best.pt
    if not cfg_path.exists():
        return ck, f"unknown ({cfg['init_from']})"
    c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return ck, model_version(c, str(c["version"]))


# ---------------------------------------------------------------- paths + status
def cache_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["cache"]["root"]


def lang_splits(iso: str) -> list[str]:
    return [f"indic_{iso}/{s}" for s in ("train", "dev", "train_aug1", "dev_aug1")]


def tag(e: dict[str, Any]) -> str:
    return f"indic_{e['index']:02d}"


def run_file(version: str, t: str) -> Path:
    return RUNS_DIR / f"{version.replace('@', '_')}__{t}.json"


def status(e: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    iso, root = e["iso"], cache_root(cfg)
    cfg_path = CFG_DIR / f"{e['index']:02d}_{iso}.yaml"
    _, before = init_model(e, cfg)
    return {
        "import": (ROOT / f"data/manifests/indic_{iso}.jsonl").exists()
        and (ROOT / f"splits/indic_{iso}.json").exists(),
        "cache": all(cache_done(s, root) for s in lang_splits(iso)),
        "train": cfg_path.exists() and train_done(ROOT / e["run"], str(cfg_path.relative_to(ROOT))),
        "replay": all(cache_done(f"replay/{iso}_{s}", root) for s in ("train", "dev")),
        "eval": run_file(after_version(e), tag(e)).exists() and run_file(before, tag(e)).exists(),
    }


# ---------------------------------------------------------------- configs
def split_counts(iso: str, split: str) -> tuple[int, Counter[str]]:
    from ml.data.manifest import read_manifest

    ids = set(
        json.loads((ROOT / f"splits/indic_{iso}.json").read_text(encoding="utf-8"))["splits"][split]
    )
    rows = [r for r in read_manifest(ROOT / f"data/manifests/indic_{iso}.jsonl") if r.utt_id in ids]
    return sum(r.label == "bona_fide" for r in rows), Counter(
        r.generator_family for r in rows if r.label == "spoof"
    )


def write_train_config(e: dict[str, Any], cfg: dict[str, Any]) -> Path:
    iso, k = e["iso"], e["index"]
    frac = float(cfg["cache"].get("aug_fraction", 0.5))
    augment = yaml.safe_load((ROOT / cfg["augment_from"]).read_text(encoding="utf-8"))["augment"]
    cache: dict[str, Any] = {
        "dir": cfg["cache"]["root"],
        "batch": cfg["cache"]["batch"],
        "workers": cfg["cache"]["workers"],
    }
    for split in ("train", "dev"):
        n_bona, fams = split_counts(iso, split)
        cache[f"indic_{iso}/{split}"] = {
            "source": split,
            "max_bona": None,
            "spoof_per_family": None,
        }
        cache[f"indic_{iso}/{split}_aug1"] = {
            "source": split,
            "view": 1,
            "max_bona": max(1, round(frac * n_bona)),
            "spoof_per_family": max(1, math.ceil(frac * sum(fams.values()) / max(1, len(fams)))),
        }
    prev = earlier(e)
    init, _ = init_model(e, cfg)
    train = dict(cfg["train"])
    train["train_sets"] = [f"indic_{iso}/train", f"indic_{iso}/train_aug1", "replay/asv5_train"]
    train["train_sets"] += [f"replay/{p['iso']}_train" for p in prev]
    train["dev_sets"] = [f"indic_{iso}/dev", f"indic_{iso}/dev_aug1", "replay/asv5_dev"]
    train["dev_sets"] += [f"replay/{p['iso']}_dev" for p in prev]
    train["init_from"] = str(init.relative_to(ROOT)).replace("\\", "/")
    out = {
        "run_name": Path(e["run"]).name,
        "lineage": cfg["lineage"],
        "allow_noncommercial": cfg["allow_noncommercial"],
        "seed": cfg["seed"],
        "out_dir": "runs",
        **{
            k2: cfg[k2]
            for k2 in (
                "frontend",
                "frontend_revision",
                "frontend_layers",
                "freeze_frontend",
                "backend",
                "emb_dim",
                "loss",
            )
        },
        "version": e["version"],
        "data": {
            "manifests": f"data/manifests/indic_{iso}.jsonl",
            "splits": f"splits/indic_{iso}.json",
            "data_root": "data",
        },
        "cache": cache,
        "augment": augment,
        "train": train,
    }
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{k:02d}_{iso}.yaml"
    path.write_text(
        f"# generated by scripts/train_indic_language.py from {CONFIG.relative_to(ROOT)}; "
        "edit that file, not this one\n" + yaml.safe_dump(out, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_eval_config(e: dict[str, Any], cfg: dict[str, Any]) -> Path:
    ev = cfg["eval"]
    sets = [
        {
            "name": f"indic_{p['iso']}_test",
            "manifests": f"data/manifests/indic_{p['iso']}.jsonl",
            "split": {"file": f"splits/indic_{p['iso']}.json", "name": "eval"},
        }
        for p in [*earlier(e), e]
    ]
    sets.append(
        {
            "name": "asvspoof5_eval",
            "manifests": "data/manifests/asvspoof5_eval.jsonl",
            "split": {"file": "splits/asvspoof5.json", "name": "eval"},
            "sample": int(ev.get("asvspoof5_sample", 10000)),
        }
    )
    out: dict[str, Any] = {
        "lineage": cfg["lineage"],
        "allow_noncommercial": cfg["allow_noncommercial"],
        "data_root": "data",
        "sample_seed": 0,
        "eval_sets": sets,
        "train_datasets": ["asvspoof5", "indicsynth", "kathbath"],
        "seen_families": [],
        "operating_point": {"mode": "eer"},
        "fairness": {"attributes": ["gender", "language"], "max_gap": None, "min_group_n": 30},
        "bootstrap": int(ev.get("bootstrap", 200)),
    }
    if ev.get("full"):
        base = yaml.safe_load(
            (ROOT / "ml/eval/configs/base_asvspoof5.yaml").read_text(encoding="utf-8")
        )
        out["sweeps"], out["adversarial"] = base["sweeps"], base["adversarial"]
    path = CFG_DIR / f"{e['index']:02d}_{e['iso']}_eval.yaml"
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- steps
def ensure_asv5_replay(cfg: dict[str, Any]) -> int:
    from ml.training.replay_cache import build_replay

    root = cache_root(cfg)
    for split in ("train", "dev"):
        out = root / f"replay/asv5_{split}"
        if cache_done(f"replay/asv5_{split}", root):
            continue
        srcs = [root / s for s in (split, f"{split}_aug1") if cache_done(s, root)]
        if not srcs:
            print(
                f"The ASVspoof 5 feature cache ({root / split}) is gone, so the English replay "
                "sample cannot be built. It is needed to keep the model good at English attacks."
            )
            return 2
        n = cfg["replay"]["asvspoof5"][split]
        build_replay(srcs, out, int(n["bona"]), int(n["spoof"]), int(cfg["seed"]))
    return 0


def step(e: dict[str, Any], name: str, cfg: dict[str, Any]) -> int:
    iso, root = e["iso"], cache_root(cfg)
    if name == "import":
        imp = cfg["import"]
        return run(
            "-m",
            "ml.data.importers.indic",
            "--language",
            e["language"],
            "--spoof",
            str(imp["spoof"]),
            "--bona",
            str(imp["bona"]),
            "--seed",
            str(cfg["seed"]),
            "--workers",
            str(imp.get("workers", 4)),
        )
    if not status(e, cfg)["import"]:
        print(f"{e['language']}: run the import step first")
        return 2
    if name == "cache":
        path = write_train_config(e, cfg)
        rc = run(
            "-m",
            "ml.training.feature_cache",
            "--config",
            str(path.relative_to(ROOT)),
            "--splits",
            *lang_splits(iso),
        )
        return rc or ensure_asv5_replay(cfg)
    if name == "train":
        for p in earlier(e):
            st = status(p, cfg)
            if not (st["train"] and st["replay"]):
                print(
                    f"{p['language']} (step {p['index']:02d}) is not finished: train it and build "
                    f"its replay first (python scripts/train_indic_language.py {p['language']} all)"
                )
                return 2
        init, _ = init_model(e, cfg)
        if not init.exists():
            print(f"starting model {init} not found: set init_from in {CONFIG.relative_to(ROOT)}")
            return 2
        if not status(e, cfg)["cache"]:
            print(f"{e['language']}: run the cache step first")
            return 2
        rc = ensure_asv5_replay(cfg)
        if rc:
            return rc
        path = write_train_config(e, cfg)
        return run("-m", "ml.training.train_head_a_cached", "--config", str(path.relative_to(ROOT)))
    if name == "replay":
        from ml.training.replay_cache import build_replay

        if not status(e, cfg)["train"]:
            print(f"{e['language']}: train first (the replay is built after training)")
            return 2
        for split in ("train", "dev"):
            n = cfg["replay"]["language"][split]
            build_replay(
                [root / f"indic_{iso}/{split}", root / f"indic_{iso}/{split}_aug1"],
                root / f"replay/{iso}_{split}",
                int(n["bona"]),
                int(n["spoof"]),
                int(cfg["seed"]),
            )
        big = root / f"indic_{iso}"
        gb = sum(f.stat().st_size for f in big.rglob("feats.npy")) / 1e9
        print(
            f"\nThe replay sample of {e['language']} is saved in {root / 'replay'}.\n"
            f"You can now delete the folder  {big}  by hand to free ~{gb:.0f} GB\n"
            "(Shift+Delete in File Explorer). Keep the replay folder and data\\indic\\"
            f"{iso} (the audio, needed for evaluation)."
        )
        return 0
    if name == "eval":
        if not status(e, cfg)["train"]:
            print(f"{e['language']}: train first")
            return 2
        path = str(write_eval_config(e, cfg).relative_to(ROOT))
        before_ck, before = init_model(e, cfg)
        after = after_version(e)
        for ck, version in ((before_ck, before), (ROOT / e["run"] / "best.pt", after)):
            if run_file(version, tag(e)).exists():
                print(f"--- eval of {version} on {tag(e)} already done")
                continue
            rc = run(
                "ml/eval/run_eval.py", "--model", f"head_a:{ck}", "--config", path, "--tag", tag(e)
            )
            if rc:
                return rc
        return compare()
    print(__doc__)
    return 2


# ---------------------------------------------------------------- compare
def eers(path: Path) -> dict[str, float | None]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {r["slice"]: r.get("eer") for r in d["rows"] if r["protocol"] == "language"}


def compare() -> int:
    cfg = load_cfg()
    tol = cfg.get("compare", {}).get("max_regression_eer")
    led = load_ledger()["order"]
    if not led:
        print("no language started yet")
        return 0
    print(
        "EER per language (lower is better); 'en' = ASVspoof 5. before = previous model, "
        "after = the model trained on the new language, on the same test sets.\n"
    )
    worst = 0.0
    for e in led:
        _, before = init_model(e, cfg)
        after = after_version(e)
        fb, fa = run_file(before, tag(e)), run_file(after, tag(e))
        if not (fb.exists() and fa.exists()):
            print(f"step {e['index']:02d} {e['language']}: not evaluated yet")
            continue
        b, a = eers(fb), eers(fa)
        print(f"step {e['index']:02d} {e['language']} ({before} -> {after}), rows in REPORT.md:")
        for lang in sorted(set(b) | set(a), key=lambda x: (x != e["iso"], x)):
            vb, va = b.get(lang), a.get(lang)
            if vb is None or va is None:
                continue
            delta = va - vb
            new = lang == e["iso"]
            if not new:
                worst = max(worst, delta)
            flag = ""
            if not new and tol is not None and delta > float(tol):
                flag = "  <-- REGRESSION beyond max_regression_eer"
            print(
                f"   {lang:4s} {'(new)' if new else '     '}  {vb:7.2%} -> {va:7.2%}  "
                f"({delta:+.2%}){flag}"
            )
        print()
    if tol is None:
        print(
            f"largest increase on an earlier language / ASVspoof 5: {worst:+.2%}. "
            "No pass/fail limit is set yet (compare.max_regression_eer, PROJECT_STATUS Q13)."
        )
        return 0
    return 1 if worst > float(tol) else 0


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    cfg = load_cfg()
    if args and args[0] == "queue":
        # several languages back to back (e.g. overnight); re-running skips finished ones
        names = args[1:]
        if not names:
            print("usage: python scripts/train_indic_language.py queue hindi marathi telugu ...")
            return 2
        for lang in names:
            language(lang)  # fail fast on a typo, before hours of work
        for lang in names:
            print(f"\n=== {lang} ===", flush=True)
            rc = main([lang, "all"])
            if rc:
                print(
                    f"--- queue stopped at {lang}; fix the message above, then re-run the same queue"
                )
                return rc
        return compare()
    if not args or args[0] in ("status", "compare"):
        if args and args[0] == "compare":
            return compare()
        led = load_ledger()["order"]
        if not led:
            print(
                "No language started yet. Start with e.g.:  "
                "python scripts/train_indic_language.py bengali import"
            )
        for e in led:
            print(f"step {e['index']:02d} {e['language']:10s} {json.dumps(status(e, cfg))}")
        return 0
    lang, name = args[0], (args[1] if len(args) > 1 else "status")
    if name not in (*STEPS, "all", "status"):
        print(__doc__)
        return 2
    e = entry(lang, create=name != "status")
    if e is None:
        print(
            f"{lang}: not started. First step:  python scripts/train_indic_language.py {lang} import"
        )
        return 0
    if name == "status":
        st = status(e, cfg)
        print(json.dumps(st, indent=2))
        nxt = next((s for s in STEPS if not st[s]), None)
        print(
            f"next step: python scripts/train_indic_language.py {e['language']} {nxt}"
            if nxt
            else f"{e['language']}: all steps done"
        )
        return 0
    if name == "all":
        for s in STEPS:
            if status(e, cfg)[s]:
                print(f"--- {s}: already done, skipping")
                continue
            rc = step(e, s, cfg)
            if rc:
                print(
                    f"--- {s} failed (exit {rc}); fix the message above, then re-run the same command"
                )
                return rc
        return 0
    return step(e, name, cfg)


if __name__ == "__main__":
    sys.exit(main())
