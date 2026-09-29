"""Privacy service (B16): retention job, DSAR, key rotation, consent lookups.

CLI (run by cron / a k8s CronJob, B17):

    python -m services.privacy.main retention --tenants bank-a,bank-b [--dry-run]
    python -m services.privacy.main dsar-export --tenant bank-a --customer cust-42
    python -m services.privacy.main dsar-erase  --tenant bank-a --caller +9198xxxxxxx [--dry-run]
    python -m services.privacy.main rotate-key  --tenant bank-a
    python -m services.privacy.main serve       # admin HTTP API, internal network only
    python -m services.privacy.main retention-loop --tenants bank-a --every-hours 24   # sidecar

Stores are opened from ``VG_DATA_DIR`` (default ``var/``): evidence.db,
timeline.db, feedback.db, voiceprints.db, flagged_audio.db, audit.db, keys.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from services.privacy.audit import AuditLog
from services.privacy.keys import TenantKeyManager
from services.privacy.policy import ConsentMatrix, TenantPrivacy
from services.privacy.raw_retention import FlaggedAudioVault
from services.privacy.retention import DSAR, RetentionJob, Stores, Subject


class SubjectIn(BaseModel):
    claimed_identity_id: str | None = None
    caller_number: str | None = None
    speaker_id: str | None = None
    actor: str
    dry_run: bool = False


def open_stores(data_dir: Path | str | None = None) -> tuple[Stores, AuditLog, TenantKeyManager]:
    from packages.vg_models.heads.head_d_speaker.vault import VoiceprintVault
    from services.fusion.persistence import SQLiteTimelineStore
    from services.policy.evidence import EvidenceStore
    from services.policy.feedback import FeedbackStore

    d = Path(data_dir or os.getenv("VG_DATA_DIR", "var"))
    d.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(d / "audit.db")
    keys = TenantKeyManager(d / "keys.json")
    evidence = EvidenceStore(d / "evidence.db")
    stores = Stores(
        evidence=evidence,
        timeline=SQLiteTimelineStore(d / "timeline.db"),
        feedback=FeedbackStore(evidence, d / "feedback.db"),
        voiceprints=VoiceprintVault(d / "voiceprints.db", key_provider=keys.provider),
        flagged_audio=FlaggedAudioVault(
            keys.provider, ConsentMatrix.load(), audit, d / "flagged_audio.db"
        ),
    )
    return stores, audit, keys


def create_app(stores: Stores, audit: AuditLog, keys: TenantKeyManager) -> Any:  # noqa: ANN401
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="VoiceGuard privacy (internal)", version="0.1.0")
    dsar = DSAR(stores, audit)

    def subject(b: SubjectIn) -> Subject:
        s = Subject(b.claimed_identity_id, b.caller_number, b.speaker_id)
        if not (s.claimed_identity_id or s.caller_number or s.speaker_id):
            raise HTTPException(422, "give at least one subject identifier")
        return s

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "audit_chain_ok": audit.verify(),
            "evidence_chain_ok": stores.evidence.verify_chain(),
        }

    @app.get("/v1/consent-matrix")
    def matrix() -> dict[str, Any]:
        return {k: asdict(v) for k, v in ConsentMatrix.load().purposes.items()}

    @app.get("/v1/tenants/{tenant_id}/privacy")
    def tenant_policy(tenant_id: str) -> dict[str, Any]:
        return asdict(TenantPrivacy.load(tenant_id))

    @app.post("/v1/tenants/{tenant_id}/dsar/export")
    def export(tenant_id: str, body: SubjectIn) -> dict[str, Any]:
        return dsar.export(tenant_id, subject(body), body.actor)

    @app.post("/v1/tenants/{tenant_id}/dsar/erase")
    def erase(tenant_id: str, body: SubjectIn) -> dict[str, Any]:
        return dsar.erase(tenant_id, subject(body), body.actor, body.dry_run)

    @app.post("/v1/retention/run")
    def retention(tenants: list[str] = Body(...), dry_run: bool = False) -> list[dict[str, Any]]:
        return [asdict(r) for r in RetentionJob(stores, audit).run(tenants, dry_run=dry_run)]

    @app.get("/v1/tenants/{tenant_id}/audit")
    def audit_events(tenant_id: str) -> list[dict[str, Any]]:
        return audit.events(tenant_id)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retention")
    r.add_argument("--tenants", required=True)
    r.add_argument("--dry-run", action="store_true")
    for name in ("dsar-export", "dsar-erase"):
        p = sub.add_parser(name)
        p.add_argument("--tenant", required=True)
        p.add_argument("--customer")
        p.add_argument("--caller")
        p.add_argument("--speaker")
        p.add_argument("--actor", default="cli")
        p.add_argument("--dry-run", action="store_true")
    k = sub.add_parser("rotate-key")
    k.add_argument("--tenant", required=True)
    sub.add_parser("serve")
    lp = sub.add_parser("retention-loop")
    lp.add_argument("--tenants", required=True)
    lp.add_argument("--every-hours", type=float, default=24.0)
    args = ap.parse_args(argv)

    stores, audit, keys = open_stores()
    if args.cmd == "retention":
        reps = RetentionJob(stores, audit).run(args.tenants.split(","), dry_run=args.dry_run)
        print(json.dumps([asdict(x) for x in reps], indent=2))
    elif args.cmd in ("dsar-export", "dsar-erase"):
        s = Subject(args.customer, args.caller, args.speaker)
        d = DSAR(stores, audit)
        out = (
            d.export(args.tenant, s, args.actor)
            if args.cmd == "dsar-export"
            else d.erase(args.tenant, s, args.actor, args.dry_run)
        )
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "rotate-key":
        v = keys.rotate(args.tenant, [stores.voiceprints, stores.flagged_audio])
        audit.record("key_rotated", "cli", args.tenant, "key_management", details={"version": v})
        print(f"{args.tenant}: now on key version {v}")
    elif args.cmd == "retention-loop":
        import time

        while True:  # container restart policy handles crashes; each run is audited
            reps = RetentionJob(stores, audit).run(args.tenants.split(","))
            print(json.dumps([asdict(x) for x in reps]), flush=True)
            time.sleep(args.every_hours * 3600)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            create_app(stores, audit, keys),
            host=os.getenv("VG_PRIVACY_HOST", "127.0.0.1"),
            port=int(os.getenv("VG_PRIVACY_PORT", "8095")),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
