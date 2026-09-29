"""Per-tenant data keys with versioning and rotation (B16-T07).

Envelope encryption: each tenant has 32-byte data-encryption keys (DEKs), one
*current* version plus older versions kept only until everything is re-wrapped.
DEKs are stored wrapped (AES-GCM) under a key-encryption key (KEK) that comes
from a KMS/HSM in production (``VG_KEK_B64`` env var or a ``kek_provider``
callback here). Nothing in this file ever writes a DEK in the clear.

``TenantKeyManager.provider`` plugs straight into ``VoiceprintVault`` (B7) and
``FlaggedAudioVault``; ``rotate`` creates a new version and re-encrypts the
voiceprint vault with it.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class TenantKeyError(RuntimeError):
    pass


def env_kek() -> bytes:
    raw = os.getenv("VG_KEK_B64")
    if not raw:
        raise TenantKeyError("no key-encryption key: set VG_KEK_B64 or configure a KMS provider")
    kek = base64.b64decode(raw)
    if len(kek) != 32:
        raise TenantKeyError("VG_KEK_B64 must decode to 32 bytes")
    return kek


class TenantKeyManager:
    def __init__(self, path: Path | str, kek_provider: Callable[[], bytes] = env_kek) -> None:
        self.path = Path(path)
        self._kek = kek_provider
        self._lock = threading.Lock()
        self._state: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )

    # ------------------------------------------------------------------ wrapping
    def _wrap(self, tenant_id: str, version: int, dek: bytes) -> dict[str, str]:
        nonce = os.urandom(12)
        aad = f"vg-dek:{tenant_id}:{version}".encode()
        ct = AESGCM(self._kek()).encrypt(nonce, dek, aad)
        return {"nonce": base64.b64encode(nonce).decode(), "ct": base64.b64encode(ct).decode()}

    def _unwrap(self, tenant_id: str, version: int, w: dict[str, str]) -> bytes:
        aad = f"vg-dek:{tenant_id}:{version}".encode()
        try:
            return AESGCM(self._kek()).decrypt(
                base64.b64decode(w["nonce"]), base64.b64decode(w["ct"]), aad
            )
        except Exception as exc:  # noqa: BLE001 - never leak crypto details
            raise TenantKeyError(f"cannot unwrap key {tenant_id} v{version}") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ API
    def ensure(self, tenant_id: str) -> int:
        with self._lock:
            t = self._state.get(tenant_id)
            if t is None:
                t = {
                    "current": 1,
                    "versions": {"1": self._wrap(tenant_id, 1, os.urandom(32))},
                    "created": dt.datetime.now(dt.UTC).isoformat(),
                    "rotations": [],
                }
                self._state[tenant_id] = t
                self._save()
            return int(t["current"])

    def key(self, tenant_id: str, version: int | None = None) -> bytes:
        self.ensure(tenant_id)
        t = self._state[tenant_id]
        v = int(version or t["current"])
        w = t["versions"].get(str(v))
        if w is None:
            raise TenantKeyError(f"tenant {tenant_id} has no key version {v} (retired)")
        return self._unwrap(tenant_id, v, w)

    def provider(self, tenant_id: str) -> bytes:
        """Current DEK — the ``key_provider`` signature used by the vaults."""
        return self.key(tenant_id)

    def rotate(self, tenant_id: str, vaults: list[Any] | None = None) -> int:
        """New key version; re-encrypt every vault, then retire the old version."""
        old_v = self.ensure(tenant_id)
        old = self.key(tenant_id, old_v)
        with self._lock:
            t = self._state[tenant_id]
            new_v = old_v + 1
            new = os.urandom(32)
            t["versions"][str(new_v)] = self._wrap(tenant_id, new_v, new)
            t["current"] = new_v
            self._save()
        rewrapped = 0
        for v in vaults or []:
            rewrapped += int(v.rotate_tenant_key(tenant_id, old, new))
        with self._lock:
            t = self._state[tenant_id]
            t["versions"].pop(str(old_v), None)  # crypto-shred the old key
            t["rotations"].append(
                {
                    "at": dt.datetime.now(dt.UTC).isoformat(),
                    "from": old_v,
                    "to": new_v,
                    "records": rewrapped,
                }
            )
            self._save()
        return new_v

    def shred(self, tenant_id: str) -> None:
        """Destroy every key of a tenant (offboarding): all its ciphertext becomes unreadable."""
        with self._lock:
            self._state.pop(tenant_id, None)
            self._save()

    def versions(self, tenant_id: str) -> list[int]:
        t = self._state.get(tenant_id, {})
        return sorted(int(v) for v in t.get("versions", {}))
