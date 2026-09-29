"""Model card + data statement generator (B16-T09).

    python -m services.privacy.model_card --version A@xlsr300m-nes2net-v0.3.1

Combines the model registry entry (sha256, lineage, status, eval run id), the
evaluation run record (overall / unseen-family / cross-dataset / per-language rows
and the fairness gate) and the data registry (licences, commercial use) into
``docs/model_cards/<version>.md``. Numbers are copied with their REPORT.md row ids
so the card never becomes a second, divergent source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from packages.vg_eval.report import RunRecord, row_id
from packages.vg_models.registry import ModelEntry, ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs/model_cards/TEMPLATE.md"


def _find_run(run_id: str | None, runs_dir: Path) -> RunRecord | None:
    if not run_id:
        return None
    for p in sorted(runs_dir.glob("*.json")):
        rec = RunRecord.load(p)
        if rec.run_id == run_id:
            return rec
    return None


def _eval_section(rec: RunRecord | None) -> tuple[str, str]:
    if rec is None:
        return (
            "_No evaluation run recorded — this model must not be promoted (B17 registry)._",
            "_Not evaluated._",
        )
    lines = [
        "| Row id | Protocol | Slice | EER | minDCF | Coverage |",
        "|---|---|---|---:|---:|---:|",
    ]
    keep = {"overall", "logo", "cross_dataset", "language"}
    for r in rec.rows:
        if r["protocol"] not in keep or r.get("eer") is None:
            continue
        if r["protocol"] == "logo" and not r["slice"].startswith("ALL_"):
            continue
        lines.append(
            f"| `{row_id(rec.run_id, r['protocol'], r['slice'])}` | {r['protocol']} | {r['slice']} | "
            f"{100 * r['eer']:.2f}% | {r['min_dcf']:.3f} | {100 * r['coverage']:.1f}% |"
        )
    fair = [f"Release gate: **{rec.gate.get('status', '—')}** — {rec.gate.get('detail', '')}", ""]
    for f in rec.fairness:
        worst = max(f["groups"], key=lambda g: g["fpr"]) if f["groups"] else None
        fair.append(
            f"- {f['attribute']}: FPR gap {100 * f['gap']:.2f} pts"
            + (f" (highest: {worst['group']} {100 * worst['fpr']:.2f}%)" if worst else "")
        )
    return "\n".join(lines), "\n".join(fair)


def _data_section(rec: RunRecord | None, entry: ModelEntry) -> tuple[str, str]:
    reg = yaml.safe_load((ROOT / "ml/data/registry.yaml").read_text(encoding="utf-8"))
    by_name = {d["name"]: d for d in reg.get("datasets", [])}
    names = list(rec.data.get("train_datasets", [])) if rec else []
    if not names:
        return (
            "_Training datasets not recorded in the evaluation run — fill in by hand._",
            f"Lineage: {entry.lineage}.",
        )
    rows = ["| Dataset | Role | Licence | Commercial use |", "|---|---|---|---|"]
    lic = []
    for n in names:
        d: dict[str, Any] = by_name.get(n, {})
        rows.append(
            f"| {n} | {d.get('role', '?')} | {d.get('license', '?')} | {d.get('commercial_use', '?')} |"
        )
        if d and not d.get("commercial_use", False):
            lic.append(
                f"- {n} ({d.get('license')}) is **non-commercial**: research lineage only (I7)."
            )
    return "\n".join(rows), "\n".join(lic) or "All training data permits commercial use."


def render(entry: ModelEntry, rec: RunRecord | None) -> str:
    evaluation, fairness = _eval_section(rec)
    data, licences = _data_section(rec, entry)
    return TEMPLATE.read_text(encoding="utf-8").format(
        version=entry.version,
        kind=entry.kind,
        lineage=entry.lineage,
        commercial_ok="yes" if entry.lineage == "commercial" else "no",
        sha256=entry.sha256,
        status=entry.status,
        eval_run=entry.eval_run_id or "none",
        fairness_gate=entry.fairness_gate or "not evaluated",
        training_data=data,
        evaluation=evaluation,
        fairness=fairness,
        licences=licences,
        extra_failure_modes="(add deployment-specific observations from shadow mode here)",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "docs/benchmarks/runs")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/model_cards")
    args = ap.parse_args(argv)
    reg = ModelRegistry(args.registry) if args.registry else ModelRegistry()
    entry = reg.get(args.version)
    card = render(entry, _find_run(entry.eval_run_id, args.runs_dir))
    out = args.out_dir / f"{entry.version.replace('@', '_').replace('/', '_')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
