"""Model registry CLI: register, attach an evaluation, promote (blue/green), roll back (B17-T07).

    python scripts/model_rollout.py register --version A@xlsr300m-nes2net-v0.3.1 --kind head_a \
        --path runs/head_a_stage2/best.pt --lineage research
    python scripts/model_rollout.py attach-eval --version A@... --run-id r3f9a1c02
    python scripts/model_rollout.py promote --version A@... --actor alice
    python scripts/model_rollout.py rollback --kind head_a --actor alice
    python scripts/model_rollout.py status

Gateways running with ``VG_INFERENCE_BACKEND=registry`` pick up the active Head A
on their next inference batch — promote and rollback need no restart. The fairness
gate status is read from the evaluation run record, so promotion cannot skip it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.vg_eval.report import RunRecord  # noqa: E402
from packages.vg_models.registry import ModelRegistry, RegistryError  # noqa: E402


def _audit(action: str, actor: str, details: dict[str, object]) -> None:
    data = Path(os.getenv("VG_DATA_DIR", "var"))
    if not data.exists():
        return
    from services.privacy.audit import AuditLog

    AuditLog(data / "audit.db").record(
        action,
        actor,
        os.getenv("VG_OPERATOR_TENANT", "platform"),
        "model_management",
        details=details,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--registry", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--version", required=True)
    r.add_argument("--kind", required=True)
    r.add_argument("--path", required=True)
    r.add_argument("--lineage", required=True, choices=["research", "commercial"])
    r.add_argument("--notes", default="")
    a = sub.add_parser("attach-eval")
    a.add_argument("--version", required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--runs-dir", type=Path, default=Path("docs/benchmarks/runs"))
    p = sub.add_parser("promote")
    p.add_argument("--version", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--commercial", action="store_true", help="deployment is commercial (I7 check)")
    b = sub.add_parser("rollback")
    b.add_argument("--kind", required=True)
    b.add_argument("--actor", required=True)
    sub.add_parser("status")
    args = ap.parse_args(argv)

    reg = ModelRegistry(args.registry) if args.registry else ModelRegistry()
    try:
        if args.cmd == "register":
            e = reg.register(args.version, args.kind, args.path, args.lineage, notes=args.notes)
            print(
                f"registered {e.version} ({e.kind}, {e.lineage}) sha256={e.sha256[:12]} as staging"
            )
        elif args.cmd == "attach-eval":
            rec = next(
                (
                    RunRecord.load(x)
                    for x in sorted(args.runs_dir.glob("*.json"))
                    if RunRecord.load(x).run_id == args.run_id
                ),
                None,
            )
            if rec is None:
                raise RegistryError(f"run {args.run_id} not found in {args.runs_dir}")
            if rec.synthetic:
                raise RegistryError("synthetic smoke-test runs cannot back a promotion")
            gate = rec.gate.get("status", "not_evaluated")
            reg.attach_eval(args.version, args.run_id, gate)
            print(f"{args.version}: eval {args.run_id}, fairness gate {gate}")
        elif args.cmd == "promote":
            reg.promote(args.version, commercial_deployment=args.commercial, actor=args.actor)
            _audit("model_promoted", args.actor, {"version": args.version})
            print(f"{args.version} is now active (previous kept for instant rollback)")
        elif args.cmd == "rollback":
            e = reg.rollback(args.kind, actor=args.actor)
            _audit("model_rolled_back", args.actor, {"kind": args.kind, "to": e.version})
            print(f"{args.kind}: rolled back to {e.version}")
        else:
            for e in reg.entries:
                print(
                    f"{e.kind:8s} {e.status:9s} {e.version:40s} {e.lineage:10s} "
                    f"eval={e.eval_run_id or '-'} gate={e.fairness_gate or '-'}"
                )
    except RegistryError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
