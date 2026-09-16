"""Encrypted voiceprint vault (B7-T03).

Voiceprints are biometric data (DPDP 2023 / GDPR / CCPA; invariant I5, B16).

* Stores **embeddings only** — raw enrolment audio never reaches this module.
* AES-256-GCM per record, with a **per-tenant data key**. Keys come from a
  ``KeyProvider``; the default reads ``VG_VAULT_KEY_<TENANT>`` (base64, 32 bytes)
  from the environment — replace with a KMS/HSM provider in production (B16-T07).
* Tenant id and speaker id are bound into the AEAD associated data, so a record
  cannot be copied to another tenant/speaker and still decrypt.
* Every record requires an explicit consent reference (voiceprinting always
  needs explicit consent, B16-T04).
* ``delete_speaker`` performs erasure (DSAR). ``rotate_tenant_key`` re-encrypts.

Storage backend is SQLite (file or ``:memory:``); only ciphertext is stored.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KeyProvider = Callable[[str], bytes]


class VaultError(RuntimeError):
    pass


def env_key_provider(tenant_id: str) -> bytes:
    var = f"VG_VAULT_KEY_{tenant_id.upper().replace('-', '_')}"
    raw = os.getenv(var)
    if not raw:
        raise VaultError(
            f"no vault key for tenant {tenant_id!r} (set {var} or configure a KMS provider)"
        )
    key = base64.b64decode(raw)
    if len(key) != 32:
        raise VaultError(f"{var} must decode to 32 bytes (AES-256)")
    return key


@dataclass
class EnrollmentSession:
    session_ref: str
    recorded_at: str
    voiced_seconds: float
    snr_db: float
    channel: str  # e.g. "wideband", "narrowband"


@dataclass
class Voiceprint:
    tenant_id: str
    speaker_id: str
    embedder: str
    consent_ref: str
    # condition ("clean", "narrowband_g711", ...) -> list of per-session embeddings
    embeddings: dict[str, list[list[float]]] = field(default_factory=dict)
    sessions: list[EnrollmentSession] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None

    def centroid(self, condition: str | None = None) -> np.ndarray | None:
        conds = [condition] if condition and condition in self.embeddings else list(self.embeddings)
        vecs = [np.array(e) for c in conds for e in self.embeddings.get(c, [])]
        if not vecs:
            return None
        m = np.mean(vecs, axis=0)
        return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)

    def expired(self, now: dt.datetime | None = None) -> bool:
        if not self.expires_at:
            return False
        return (now or dt.datetime.now(dt.UTC)) > dt.datetime.fromisoformat(self.expires_at)


class VoiceprintVault:
    def __init__(
        self, path: str | Path = ":memory:", key_provider: KeyProvider = env_key_provider
    ) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._keys = key_provider
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS voiceprints ("
            " tenant_id TEXT, speaker_id TEXT, nonce BLOB, ciphertext BLOB, key_version INTEGER,"
            " PRIMARY KEY (tenant_id, speaker_id))"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cohorts (embedder TEXT PRIMARY KEY, dim INTEGER, data BLOB)"
        )
        self._db.commit()

    # ------------------------------------------------------------------ crypto
    @staticmethod
    def _aad(tenant_id: str, speaker_id: str) -> bytes:
        return f"vg-voiceprint:{tenant_id}:{speaker_id}".encode()

    def _encrypt(self, vp: Voiceprint, key: bytes) -> tuple[bytes, bytes]:
        nonce = os.urandom(12)
        blob = json.dumps(asdict(vp)).encode()
        return nonce, AESGCM(key).encrypt(nonce, blob, self._aad(vp.tenant_id, vp.speaker_id))

    def _decrypt(
        self, tenant_id: str, speaker_id: str, nonce: bytes, ct: bytes, key: bytes
    ) -> Voiceprint:
        try:
            raw = json.loads(AESGCM(key).decrypt(nonce, ct, self._aad(tenant_id, speaker_id)))
        except Exception as exc:  # InvalidTag and friends: never leak details
            raise VaultError("voiceprint failed integrity check") from exc
        raw["sessions"] = [EnrollmentSession(**s) for s in raw["sessions"]]
        return Voiceprint(**raw)

    # ------------------------------------------------------------------ API
    def put(self, vp: Voiceprint) -> None:
        if not vp.consent_ref:
            raise VaultError("explicit consent reference is required to store a voiceprint")
        now = dt.datetime.now(dt.UTC).isoformat()
        vp.created_at = vp.created_at or now
        vp.updated_at = now
        nonce, ct = self._encrypt(vp, self._keys(vp.tenant_id))
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO voiceprints VALUES (?, ?, ?, ?, ?)",
                (vp.tenant_id, vp.speaker_id, nonce, ct, 1),
            )
            self._db.commit()

    def get(self, tenant_id: str, speaker_id: str) -> Voiceprint | None:
        with self._lock:
            row = self._db.execute(
                "SELECT nonce, ciphertext FROM voiceprints WHERE tenant_id=? AND speaker_id=?",
                (tenant_id, speaker_id),
            ).fetchone()
        if row is None:
            return None
        return self._decrypt(tenant_id, speaker_id, row[0], row[1], self._keys(tenant_id))

    def delete_speaker(self, tenant_id: str, speaker_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM voiceprints WHERE tenant_id=? AND speaker_id=?",
                (tenant_id, speaker_id),
            )
            self._db.commit()
        return cur.rowcount > 0

    def list_speakers(self, tenant_id: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT speaker_id FROM voiceprints WHERE tenant_id=? ORDER BY speaker_id",
                (tenant_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def rotate_tenant_key(self, tenant_id: str, old_key: bytes, new_key: bytes) -> int:
        with self._lock:
            rows = self._db.execute(
                "SELECT speaker_id, nonce, ciphertext FROM voiceprints WHERE tenant_id=?",
                (tenant_id,),
            ).fetchall()
            for speaker_id, nonce, ct in rows:
                vp = self._decrypt(tenant_id, speaker_id, nonce, ct, old_key)
                n2, c2 = self._encrypt(vp, new_key)
                self._db.execute(
                    "UPDATE voiceprints SET nonce=?, ciphertext=?, key_version=key_version+1"
                    " WHERE tenant_id=? AND speaker_id=?",
                    (n2, c2, tenant_id, speaker_id),
                )
            self._db.commit()
        return len(rows)

    # Cohort embeddings are non-identifying public-corpus impostor embeddings, stored per embedder.
    def set_cohort(self, embedder: str, cohort: np.ndarray) -> None:
        arr = np.asarray(cohort, dtype=np.float32)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cohorts VALUES (?, ?, ?)",
                (embedder, arr.shape[1], arr.tobytes()),
            )
            self._db.commit()

    def get_cohort(self, embedder: str) -> np.ndarray | None:
        with self._lock:
            row = self._db.execute(
                "SELECT dim, data FROM cohorts WHERE embedder=?", (embedder,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[1], dtype=np.float32).reshape(-1, row[0])
