"""One command for the augmented model v0.2 — every step resumable (TRAINING_AUGMENTED.md).

Needs the base model's clean feature cache (``python scripts/train_base_model.py status``
shows ``"cache": true``); it is reused, not recomputed.

    python scripts/train_augmented_model.py status     # what is done, what is next
    python scripts/train_augmented_model.py check      # 1. codecs + augmentation speed (~1 min)
    python scripts/train_augmented_model.py cache      # 2. augmented XLS-R features (~1-2 h)
    python scripts/train_augmented_model.py train      # 3. train on clean + augmented (~1 h)
    python scripts/train_augmented_model.py eval       # 4. B15 evaluation -> REPORT.md (~1-2 h)
    python scripts/train_augmented_model.py all        # 2-4 in order, skipping finished steps

Stop any step with Ctrl+C; running the same command again continues where it stopped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_base_model import CACHE, ROOT, cache_done, run, train_done  # noqa: E402

CONFIG = "ml/training/configs/head_a_aug.yaml"
EVAL_CONFIG = "ml/eval/configs/base_asvspoof5.yaml"
RUN = ROOT / "runs/head_a_aug"
AUG_SPLITS = ("train_aug1", "dev_aug1")


def status() -> dict[str, bool]:
    return {
        "base_cache": cache_done("train") and cache_done("dev"),
        "cache": all(cache_done(s) for s in AUG_SPLITS),
        "train": train_done(RUN, CONFIG),
        "eval": any((ROOT / "docs/benchmarks/runs").glob("A_xlsr300m-nes2net-v0.2.0*.json")),
    }


STEPS = {
    "check": lambda: run("-m", "ml.training.channel_aug", "--config", CONFIG),
    "cache": lambda: run(
        "-m", "ml.training.feature_cache", "--config", CONFIG, "--splits", *AUG_SPLITS
    ),
    "train": lambda: run("-m", "ml.training.train_head_a_cached", "--config", CONFIG),
    "eval": lambda: run(
        "ml/eval/run_eval.py", "--model", f"head_a:{RUN / 'best.pt'}", "--config", EVAL_CONFIG
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    step = args[0] if args else "status"
    st = status()
    if not st["base_cache"] and step not in ("status", "check"):
        print(
            f"The clean base cache is missing or incomplete ({CACHE}). "
            "Finish the base model first: python scripts/train_base_model.py status"
        )
        return 2
    if step == "status":
        print(json.dumps(st, indent=2))
        nxt = next((k for k, v in st.items() if not v), None)
        if nxt == "base_cache":
            print("next: finish the base model first (python scripts/train_base_model.py status)")
        else:
            print(
                f"next step: python scripts/train_augmented_model.py {nxt}"
                if nxt
                else "all steps done"
            )
        return 0
    if step == "all":
        if run("-m", "ml.training.channel_aug", "--config", CONFIG, "-n", "12") != 0:
            print("--- check failed; fix the message above, then re-run the same command")
            return 1
        for name in ("cache", "train", "eval"):
            if st[name]:
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
