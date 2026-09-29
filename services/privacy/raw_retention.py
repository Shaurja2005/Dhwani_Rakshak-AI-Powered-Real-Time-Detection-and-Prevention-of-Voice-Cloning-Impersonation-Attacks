"""Short-window retention of raw audio for FLAGGED calls only (B16-T02).

Off by default: the default path keeps no raw audio at all (I5). A tenant can
enable it by setting ``retention.raw_audio_hours > 0`` in its privacy policy,
and even then audio is kept only when

* the consent matrix permits ``raw_audio_retention`` for the call's lawful basis,
* the call was flagged (policy band ELEVATED/HIGH), and
* the audio is encrypted with the tenant's current key (AES-GCM) and expires
  after ``raw_audio_hours``; ``purge_expired`` deletes it (retention job).

Every store and purge is written to the audit log.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from services.privacy.audit import AuditLog
from services.privacy.policy import ConsentError, ConsentMatrix, TenantPrivacy

FLAGGED_BANDS = {"ELEVATED", "HIGH"}


class FlaggedAudioVault:
    def __init__(
        self,
        key_provider: Callable[[str], bytes],
        matrix: ConsentMatrix,
        audit: AuditLog,
        path: str | Path = ":memory:",
    ) -> None:
        self._keys = key_provider
        self._matrix = matrix
        self._audit = audit
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS flagged_audio (session_id TEXT PRIMARY KEY, tenant_id TEXT,"
            " expires_at TEXT, nonce BLOB, ciphertext BLOB, sample_rate INTEGER)"
        )
        self._db.commit()

    def maybe_retain(
        self,
        session_id: str,
        tenant: TenantPrivacy,
        lawful_basis: str,
        band: str,
        pcm: np.ndarray,
        sample_rate: int = 16000,
    ) -> bool:
        hours = tenant.retention.raw_audio_hours
        if hours <= 0 or band not in FLAGGED_BANDS:
            return False
        try:
            self._matrix.require("raw_audio_retention", lawful_basis)
        except ConsentError as exc:
            self._audit.record(
                "consent_refused",
                "flagged_audio_vault",
                tenant.tenant_id,
                "raw_audio_retention",
                str(getattr(lawful_basis, "value", lawful_basis)),
                session_id,
                "denied",
                {"reason": str(exc)},
            )
            return False
        expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=hours)
        nonce = os.urandom(12)
        aad = f"vg-raw:{tenant.tenant_id}:{session_id}".encode()
        ct = AESGCM(self._keys(tenant.tenant_id)).encrypt(
            nonce, np.asarray(pcm, dtype="<f4").tobytes(), aad
        )
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO flagged_audio VALUES (?,?,?,?,?,?)",
                (session_id, tenant.tenant_id, expires.isoformat(), nonce, ct, sample_rate),
            )
            self._db.commit()
        self._audit.record(
            "raw_audio_retained",
            "flagged_audio_vault",
            tenant.tenant_id,
            "raw_audio_retention",
            str(getattr(lawful_basis, "value", lawful_basis)),
            session_id,
            details={
                "band": band,
                "expires_at": expires.isoformat(),
                "seconds": round(len(pcm) / sample_rate, 1),
            },
        )
        return True

    def get(self, session_id: str, tenant_id: str) -> np.ndarray | None:
        with self._lock:
            row = self._db.execute(
                "SELECT nonce, ciphertext, expires_at FROM flagged_audio WHERE session_id=? AND tenant_id=?",
                (session_id, tenant_id),
            ).fetchone()
        if row is None or dt.datetime.fromisoformat(row[2]) < dt.datetime.now(dt.UTC):
            return None
        aad = f"vg-raw:{tenant_id}:{session_id}".encode()
        return np.frombuffer(
            AESGCM(self._keys(tenant_id)).decrypt(row[0], row[1], aad), dtype="<f4"
        )

    def delete(self, session_ids: list[str]) -> int:
        with self._lock:
            n = sum(
                self._db.execute("DELETE FROM flagged_audio WHERE session_id=?", (s,)).rowcount
                for s in session_ids
            )
            self._db.commit()
        return n

    def purge_expired(self, now: dt.datetime | None = None) -> int:
        t = (now or dt.datetime.now(dt.UTC)).isoformat()
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, tenant_id FROM flagged_audio WHERE expires_at < ?", (t,)
            ).fetchall()
            self._db.execute("DELETE FROM flagged_audio WHERE expires_at < ?", (t,))
            self._db.commit()
        for sid, tenant in rows:
            self._audit.record(
                "raw_audio_purged",
                "retention_job",
                tenant,
                "raw_audio_retention",
                None,
                sid,
                details={"reason": "retention window expired"},
            )
        return len(rows)

    def rotate_tenant_key(self, tenant_id: str, old_key: bytes, new_key: bytes) -> int:
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, nonce, ciphertext FROM flagged_audio WHERE tenant_id=?",
                (tenant_id,),
            ).fetchall()
            for sid, nonce, ct in rows:
                aad = f"vg-raw:{tenant_id}:{sid}".encode()
                pt = AESGCM(old_key).decrypt(nonce, ct, aad)
                n2 = os.urandom(12)
                self._db.execute(
                    "UPDATE flagged_audio SET nonce=?, ciphertext=? WHERE session_id=?",
                    (n2, AESGCM(new_key).encrypt(n2, pt, aad), sid),
                )
            self._db.commit()
        return len(rows)
