"""Immutable, content-hashed evidence bundles (B11-T04).

A bundle records everything needed to justify (or audit) a decision: the policy
decision and full rationale, the session risk and timeline, context signals,
per-window fused and per-head scores, call metadata, model versions, and the
exact threshold profile applied. **No audio** (I5).

``bundle_id`` = SHA-256 of the canonical JSON (sorted keys, no whitespace) of
the bundle with ``bundle_id`` and ``policy_decision.evidence_bundle_id`` blanked
— so the decision can reference its own bundle and both remain verifiable.

``EvidenceStore`` is append-only: rows are never updated, and each row stores
the hash of the previous row (a hash chain), so deleting or altering a bundle is
detectable with ``verify_chain``. Bundles are validated against
``schemas/evidence_bundle.schema.json`` before being stored.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from packages.vg_core.models import (
    CallMetadata,
    ContextSignals,
    FusedWindowScore,
    HeadScore,
    PolicyDecision,
    SessionRisk,
)

SCHEMA_VERSION = "evidence_bundle/1.0"
SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


class EvidenceIntegrityError(RuntimeError):
    pass


def canonical(obj: Any) -> bytes:  # noqa: ANN401 - any JSON-serialisable value
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()


def _hash_body(bundle: dict[str, Any]) -> str:
    body = copy.deepcopy(bundle)
    body["bundle_id"] = ""
    body["policy_decision"]["evidence_bundle_id"] = ""
    return "sha256:" + hashlib.sha256(canonical(body)).hexdigest()


def _validator() -> Draft202012Validator:
    resources = []
    for f in SCHEMAS.glob("*.schema.json"):
        schema = json.loads(f.read_text(encoding="utf-8"))
        res = Resource.from_contents(schema)
        resources += [(schema["$id"], res), (f.name, res)]
    registry = Registry().with_resources(resources)
    root = json.loads((SCHEMAS / "evidence_bundle.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(root, registry=registry)


_VALIDATOR: Draft202012Validator | None = None


def validate(bundle: dict[str, Any]) -> None:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = _validator()
    errors = sorted(_VALIDATOR.iter_errors(bundle), key=lambda e: list(e.path))
    if errors:
        raise EvidenceIntegrityError("; ".join(f"{list(e.path)}: {e.message}" for e in errors[:5]))


def build_bundle(
    decision: PolicyDecision,
    risk: SessionRisk,
    profile_snapshot: dict[str, Any],
    context: ContextSignals | None = None,
    window_scores: list[FusedWindowScore] | None = None,
    head_scores: list[HeadScore] | None = None,
    call_metadata: CallMetadata | None = None,
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "bundle_id": "",
        "session_id": decision.session_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "policy_decision": json.loads(decision.model_dump_json()),
        "session_risk": json.loads(risk.model_dump_json()),
        "model_versions": dict(risk.model_versions),
        "threshold_profile_snapshot": profile_snapshot,
        "schema_version": SCHEMA_VERSION,
    }
    if context is not None:
        bundle["context_signals"] = json.loads(context.model_dump_json())
    if window_scores:
        bundle["window_scores"] = [json.loads(w.model_dump_json()) for w in window_scores]
    if head_scores:
        bundle["head_scores"] = [json.loads(h.model_dump_json()) for h in head_scores]
    if call_metadata is not None:
        bundle["call_metadata"] = json.loads(call_metadata.model_dump_json())
    bid = _hash_body(bundle)
    bundle["bundle_id"] = bid
    bundle["policy_decision"]["evidence_bundle_id"] = bid
    validate(bundle)
    return bundle


def verify_bundle(bundle: dict[str, Any]) -> bool:
    return (
        bool(bundle.get("bundle_id"))
        and _hash_body(bundle)
        == bundle["bundle_id"]
        == bundle["policy_decision"]["evidence_bundle_id"]
    )


class EvidenceStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS evidence (seq INTEGER PRIMARY KEY AUTOINCREMENT, bundle_id TEXT UNIQUE, "
            "session_id TEXT, created_at TEXT, body BLOB, prev_hash TEXT, row_hash TEXT)"
        )
        # Append-only at the database level too.
        self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence "
            "BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END"
        )
        self._db.commit()

    def append(self, bundle: dict[str, Any]) -> str:
        if not verify_bundle(bundle):
            raise EvidenceIntegrityError("bundle hash does not match its content")
        body = canonical(bundle)
        with self._lock:
            last = self._db.execute(
                "SELECT row_hash FROM evidence ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev = last[0] if last else "genesis"
            row_hash = hashlib.sha256(prev.encode() + body).hexdigest()
            self._db.execute(
                "INSERT INTO evidence (bundle_id, session_id, created_at, body, prev_hash, row_hash) VALUES (?,?,?,?,?,?)",
                (
                    bundle["bundle_id"],
                    bundle["session_id"],
                    bundle["created_at"],
                    body,
                    prev,
                    row_hash,
                ),
            )
            self._db.commit()
        return str(bundle["bundle_id"])

    def get(self, bundle_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT body FROM evidence WHERE bundle_id=?", (bundle_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT body FROM evidence WHERE session_id=? ORDER BY seq", (session_id,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def verify_chain(self) -> bool:
        with self._lock:
            rows = self._db.execute(
                "SELECT body, prev_hash, row_hash FROM evidence ORDER BY seq"
            ).fetchall()
        prev = "genesis"
        for body, prev_hash, row_hash in rows:
            if prev_hash != prev or hashlib.sha256(prev.encode() + body).hexdigest() != row_hash:
                return False
            if not verify_bundle(json.loads(body)):
                return False
            prev = row_hash
        return True
