"""Immutable, content-hashed evidence bundles (B11-T04).

A bundle records everything needed to justify (or audit) a decision: the policy
decision and full rationale, the session risk and timeline, context signals,
per-window fused and per-head scores, call metadata, model versions, and the
exact threshold profile applied. **No audio** (I5).

``bundle_id`` = SHA-256 of the canonical JSON (sorted keys, no whitespace) of
the bundle with ``bundle_id`` and ``policy_decision.evidence_bundle_id`` blanked
— so the decision can reference its own bundle and both remain verifiable.

``EvidenceStore`` is append-only: chain rows are never updated, and each row
stores the hash of the previous row (a hash chain), so deleting or altering a
bundle is detectable with ``verify_chain``. Retention / erasure (B16) remove only
the body and leave a verifiable erasure record. Bundles are validated against
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
    """Append-only hash chain over bundle *hashes*, with erasable bodies (B11-T04 + B16-T06).

    * ``evidence`` (the chain) is never updated: each row stores the SHA-256 of the
      bundle body and the previous row hash, so deleting or altering history is
      detectable.
    * ``evidence_body`` holds the bundle JSON. Retention and data-subject erasure
      delete the body only, recording why in ``evidence_erasure`` — the chain still
      verifies and proves *what* existed and *that* it was erased, without keeping
      the personal data.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(evidence)").fetchall()}
        if "body" in cols and "body_sha" not in cols:
            self._migrate_v1()
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS evidence (seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " bundle_id TEXT UNIQUE, session_id TEXT, tenant_id TEXT, created_at TEXT,"
            " body_sha TEXT, prev_hash TEXT, row_hash TEXT);"
            "CREATE TABLE IF NOT EXISTS evidence_body (bundle_id TEXT PRIMARY KEY, body BLOB);"
            "CREATE TABLE IF NOT EXISTS evidence_erasure (bundle_id TEXT PRIMARY KEY,"
            " erased_at TEXT, reason TEXT);"
            # Append-only at the database level too.
            "CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence "
            "BEGIN SELECT RAISE(ABORT, 'evidence is append-only'); END;"
        )
        self._db.commit()

    def _migrate_v1(self) -> None:
        """B11 v1 schema (body inside the chain row) -> v2 (erasable bodies).

        The v1 chain is verified first; a broken chain is never migrated (that would
        launder tampering). The v2 chain is rebuilt in the same order and the v1 head
        hash is recorded in ``evidence_migration`` so the two histories stay linked.
        """
        rows = self._db.execute(
            "SELECT bundle_id, session_id, created_at, body, prev_hash, row_hash FROM evidence ORDER BY seq"
        ).fetchall()
        prev = "genesis"
        for _bid, _sid, _ts, body, prev_hash, row_hash in rows:
            if prev_hash != prev or hashlib.sha256(prev.encode() + body).hexdigest() != row_hash:
                raise EvidenceIntegrityError("v1 evidence chain is broken; refusing to migrate")
            prev = row_hash
        v1_head = prev
        self._db.executescript(
            "ALTER TABLE evidence RENAME TO evidence_v1;"
            "DROP TRIGGER IF EXISTS evidence_no_update;"
            "CREATE TABLE evidence (seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " bundle_id TEXT UNIQUE, session_id TEXT, tenant_id TEXT, created_at TEXT,"
            " body_sha TEXT, prev_hash TEXT, row_hash TEXT);"
            "CREATE TABLE IF NOT EXISTS evidence_body (bundle_id TEXT PRIMARY KEY, body BLOB);"
            "CREATE TABLE IF NOT EXISTS evidence_migration (migrated_at TEXT, from_schema TEXT,"
            " rows INTEGER, v1_head TEXT);"
        )
        prev = "genesis"
        for bid, sid, ts, body, _p, _r in rows:
            body_sha = hashlib.sha256(body).hexdigest()
            tenant = (json.loads(body).get("call_metadata") or {}).get("tenant_id")
            row_hash = hashlib.sha256((prev + body_sha).encode()).hexdigest()
            self._db.execute(
                "INSERT INTO evidence (bundle_id, session_id, tenant_id, created_at, body_sha, prev_hash, row_hash) VALUES (?,?,?,?,?,?,?)",
                (bid, sid, tenant, ts, body_sha, prev, row_hash),
            )
            self._db.execute("INSERT INTO evidence_body VALUES (?, ?)", (bid, body))
            prev = row_hash
        self._db.execute(
            "INSERT INTO evidence_migration VALUES (?, 'v1', ?, ?)",
            (datetime.now(tz=UTC).isoformat(), len(rows), v1_head),
        )
        self._db.execute("DROP TABLE evidence_v1")
        self._db.commit()

    def append(self, bundle: dict[str, Any]) -> str:
        if not verify_bundle(bundle):
            raise EvidenceIntegrityError("bundle hash does not match its content")
        body = canonical(bundle)
        body_sha = hashlib.sha256(body).hexdigest()
        tenant = (bundle.get("call_metadata") or {}).get("tenant_id")
        with self._lock:
            last = self._db.execute(
                "SELECT row_hash FROM evidence ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev = last[0] if last else "genesis"
            row_hash = hashlib.sha256((prev + body_sha).encode()).hexdigest()
            self._db.execute(
                "INSERT INTO evidence (bundle_id, session_id, tenant_id, created_at, body_sha, prev_hash, row_hash) VALUES (?,?,?,?,?,?,?)",
                (
                    bundle["bundle_id"],
                    bundle["session_id"],
                    tenant,
                    bundle["created_at"],
                    body_sha,
                    prev,
                    row_hash,
                ),
            )
            self._db.execute("INSERT INTO evidence_body VALUES (?, ?)", (bundle["bundle_id"], body))
            self._db.commit()
        return str(bundle["bundle_id"])

    def get(self, bundle_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT body FROM evidence_body WHERE bundle_id=?", (bundle_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def status(self, bundle_id: str) -> str | None:
        """``present`` | ``erased`` | None (never existed)."""
        with self._lock:
            if not self._db.execute(
                "SELECT 1 FROM evidence WHERE bundle_id=?", (bundle_id,)
            ).fetchone():
                return None
            erased = self._db.execute(
                "SELECT 1 FROM evidence_erasure WHERE bundle_id=?", (bundle_id,)
            ).fetchone()
        return "erased" if erased else "present"

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT b.body FROM evidence e JOIN evidence_body b ON b.bundle_id = e.bundle_id"
                " WHERE e.session_id=? ORDER BY e.seq",
                (session_id,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def iter_bodies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT b.body FROM evidence e JOIN evidence_body b ON b.bundle_id = e.bundle_id"
        args: tuple[Any, ...] = ()
        if tenant_id is not None:
            q += " WHERE e.tenant_id=?"
            args = (tenant_id,)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY e.seq", args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def expired(self, before_iso: str, tenant_id: str | None = None) -> list[tuple[str, str]]:
        """(bundle_id, session_id) of present bundles created before ``before_iso``."""
        q = (
            "SELECT e.bundle_id, e.session_id FROM evidence e JOIN evidence_body b"
            " ON b.bundle_id = e.bundle_id WHERE e.created_at < ?"
        )
        args: tuple[Any, ...] = (before_iso,)
        if tenant_id is not None:
            q += " AND (e.tenant_id=? OR e.tenant_id IS NULL)"
            args = (before_iso, tenant_id)
        with self._lock:
            return [(r[0], r[1]) for r in self._db.execute(q, args).fetchall()]

    def erase(self, bundle_id: str, reason: str) -> bool:
        """Delete the bundle body (personal data); keep the chain row and an erasure record."""
        with self._lock:
            cur = self._db.execute("DELETE FROM evidence_body WHERE bundle_id=?", (bundle_id,))
            if cur.rowcount:
                self._db.execute(
                    "INSERT OR REPLACE INTO evidence_erasure VALUES (?, ?, ?)",
                    (bundle_id, datetime.now(tz=UTC).isoformat(), reason[:200]),
                )
            self._db.commit()
        return cur.rowcount > 0

    def verify_chain(self) -> bool:
        with self._lock:
            rows = self._db.execute(
                "SELECT e.bundle_id, e.body_sha, e.prev_hash, e.row_hash, b.body,"
                " x.bundle_id IS NOT NULL FROM evidence e"
                " LEFT JOIN evidence_body b ON b.bundle_id = e.bundle_id"
                " LEFT JOIN evidence_erasure x ON x.bundle_id = e.bundle_id ORDER BY e.seq"
            ).fetchall()
        prev = "genesis"
        for _bid, body_sha, prev_hash, row_hash, body, erased in rows:
            if (
                prev_hash != prev
                or hashlib.sha256((prev + body_sha).encode()).hexdigest() != row_hash
            ):
                return False
            if body is None:
                if not erased:
                    return False  # body vanished without an erasure record: tampering
            elif hashlib.sha256(body).hexdigest() != body_sha or not verify_bundle(
                json.loads(body)
            ):
                return False
            prev = row_hash
        return True
