"""One command for the base model — every step resumable (docs/TRAINING_GUIDE.md).

    python scripts/train_base_model.py status          # what is done, what is next
    python scripts/train_base_model.py check           # 1. environment + GPU
    python scripts/train_base_model.py extract         # 2. unpack ASVspoof 5 (~20-40 min)
    python scripts/train_base_model.py manifests       # 3. protocols -> manifests (~5-10 min)
    python scripts/train_base_model.py cache           # 4. XLS-R features on the GPU (~1-2 h)
    python scripts/train_base_model.py train           # 5. back-end training (~30-60 min)
    python scripts/train_base_model.py eval            # 6. B15 evaluation -> REPORT.md
    python scripts/train_base_model.py all             # 2-6 in order, skipping finished steps

Stop any step with Ctrl+C; running the same command again continues where it stopped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIG = "ml/training/configs/head_a_base.yaml"
EVAL_CONFIG = "ml/eval/configs/base_asvspoof5.yaml"
EXTRACTED = ROOT / "data/asvspoof5/extracted"
CACHE = ROOT / "data/features/xlsr300m_L5-9"
RUN = ROOT / "runs/head_a_base"


def run(*args: str) -> int:
    print(f"\n>>> {' '.join(args)}\n", flush=True)
    return subprocess.call([PY, *args], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})


def cache_done(split: str, cache: Path = CACHE) -> bool:
    d = cache / split / "done.npy"
    if not d.exists():
        return False
    import numpy as np

    return bool(np.load(d).all())


def train_done(run_dir: Path = RUN, config: str = CONFIG) -> bool:
    # best.pt appears after the first epoch, so it alone does not mean training finished
    if not (run_dir / "best.pt").exists() or not (run_dir / "last.pt").exists():
        return False
    import torch
    import yaml

    st = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)  # noqa: S614
    t = yaml.safe_load((ROOT / config).read_text(encoding="utf-8"))["train"]
    return st.get("bad", 0) >= int(t.get("patience", 6)) or st["epoch"] + 1 >= int(t["epochs"])


def status() -> dict[str, bool]:
    tars = list((ROOT / "data/asvspoof5").glob("flac_*.tar"))
    markers = list(EXTRACTED.glob("flac_*.tar.done"))
    st = {
        # the tars may be deleted after a verified extraction: the .done markers remain
        "extract": (
            all((EXTRACTED / f"{t.name}.done").exists() for t in tars)
            if tars
            else len(markers) >= 18
        ),
        "manifests": all(
            (ROOT / f"data/manifests/asvspoof5_{s}.jsonl").exists()
            for s in ("train", "dev", "eval")
        ),
        "cache": cache_done("train") and cache_done("dev"),
        "train": train_done(),
        "eval": any((ROOT / "docs/benchmarks/runs").glob("A_xlsr300m-nes2net-v0.1.0*.json")),
    }
    return st


STEPS = {
    "check": lambda: run("scripts/setup/check_gpu.py", "--bench"),
    "extract": lambda: run("scripts/setup/extract_asvspoof5.py"),
    "manifests": lambda: run("-m", "ml.data.importers.asvspoof5"),
    "cache": lambda: run("-m", "ml.training.feature_cache", "--config", CONFIG),
    "train": lambda: run("-m", "ml.training.train_head_a_cached", "--config", CONFIG),
    "eval": lambda: run(
        "ml/eval/run_eval.py", "--model", f"head_a:{RUN / 'best.pt'}", "--config", EVAL_CONFIG
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    step = args[0] if args else "status"
    if step == "status":
        st = status()
        print(json.dumps(st, indent=2))
        nxt = next((k for k, v in st.items() if not v), None)
        print(f"next step: python scripts/train_base_model.py {nxt}" if nxt else "all steps done")
        return 0
    if step == "all":
        st = status()
        for name in ("extract", "manifests", "cache", "train", "eval"):
            if st[name] and name != "cache":
                print(f"--- {name}: already done, skipping")
                continue
            rc = STEPS[name]()
            if rc != 0:
                print(
                    f"--- {name} failed (exit {rc}); fix the message above, then re-run the same command"
                )
                return rc
        return 0
    if step not in STEPS:
        print(__doc__)
        return 2
    return STEPS[step]()


if __name__ == "__main__":
    sys.exit(main())
