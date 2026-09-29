"""Append-only, hash-chained audit log for privacy-relevant actions (B16-T08).

Who did what to whose data, when, and on which lawful basis: retention runs,
DSAR exports and erasures, key rotations, consent refusals, raw-audio
retention, model promotions. Events are validated against
``services/privacy/audit_event.schema.json`` and chained like the evidence
store, so a removed or edited entry is detectable.

Audit events never contain the personal data itself — only identifiers
(hashed where they are direct identifiers such as phone numbers).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA = json.loads(
    (Path(__file__).with_name("audit_event.schema.json")).read_text(encoding="utf-8")
)
_VALIDATOR = Draft202012Validator(SCHEMA)


def hash_identifier(value: str, salt: str = "vg-audit") -> str:
    return "h:" + hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:24]


class AuditLog:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT,"
            " prev_hash TEXT, row_hash TEXT);"
            "CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit "
            "BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;"
            "CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit "
            "BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;"
        )
        self._db.commit()

    def record(
        self,
        action: str,
        actor: str,
        tenant_id: str,
        purpose: str,
        lawful_basis: str | None = None,
        subject: str | None = None,
        outcome: str = "ok",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ev = {
            "schema_version": "audit_event/1.0",
            "at": dt.datetime.now(dt.UTC).isoformat(),
            "action": action,
            "actor": actor,
            "tenant_id": tenant_id,
            "purpose": purpose,
            "lawful_basis": lawful_basis,
            "subject_ref": subject,
            "outcome": outcome,
            "details": details or {},
        }
        errors = sorted(_VALIDATOR.iter_errors(ev), key=lambda e: e.path)
        if errors:
            raise ValueError(f"invalid audit event: {errors[0].message}")
        body = json.dumps(ev, sort_keys=True, separators=(",", ":"))
        with self._lock:
            last = self._db.execute(
                "SELECT row_hash FROM audit ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev = last[0] if last else "genesis"
            row_hash = hashlib.sha256((prev + body).encode()).hexdigest()
            self._db.execute(
                "INSERT INTO audit (body, prev_hash, row_hash) VALUES (?,?,?)",
                (body, prev, row_hash),
            )
            self._db.commit()
        return ev

    def events(
        self, tenant_id: str | None = None, action: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT body FROM audit ORDER BY seq").fetchall()
        out = [json.loads(r[0]) for r in rows]
        return [
            e
            for e in out
            if (tenant_id is None or e["tenant_id"] == tenant_id)
            and (action is None or e["action"] == action)
        ]

    def verify(self) -> bool:
        with self._lock:
            rows = self._db.execute(
                "SELECT body, prev_hash, row_hash FROM audit ORDER BY seq"
            ).fetchall()
        prev = "genesis"
        for body, p, h in rows:
            if p != prev or hashlib.sha256((prev + body).encode()).hexdigest() != h:
                return False
            prev = h
        return True
