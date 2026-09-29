"""B18-T03 headline chart: per-language EER before vs after Indic + channel fine-tuning.

    python scripts/make_headline_chart.py \
        --before docs/benchmarks/runs/A_xlsr300m-nes2net-v0.1.0.json \
        --after  docs/benchmarks/runs/A_xlsr300m-nes2net-v0.2.0.json \
        --out docs/demo/before_after_indic_eer.png

Both inputs are B15 run records, so every bar traces to a REPORT.md row
(``<run_id>.language.<lang>``). Synthetic smoke runs are refused, and so is a pair
evaluated on different eval sets (the comparison would not be apples to apples).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.vg_eval.report import RunRecord, plot_before_after  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/demo/before_after_indic_eer.png"))
    ap.add_argument("--allow-synthetic", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    before, after = RunRecord.load(args.before), RunRecord.load(args.after)
    if (before.synthetic or after.synthetic) and not args.allow_synthetic:
        print("refused: synthetic runs cannot make the headline chart", file=sys.stderr)
        return 2
    if before.data.get("manifest_sha256") != after.data.get("manifest_sha256"):
        print(
            "refused: the two runs were evaluated on different data (manifest sha256 differs)",
            file=sys.stderr,
        )
        return 2
    png = plot_before_after(before, after, args.out)
    print(f"{png}  (rows: {before.run_id}.language.* -> {after.run_id}.language.*)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
