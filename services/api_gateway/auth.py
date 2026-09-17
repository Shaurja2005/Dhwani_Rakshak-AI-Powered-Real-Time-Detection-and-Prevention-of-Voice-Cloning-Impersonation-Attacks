"""AuthN / AuthZ, rate limits and quotas (B12-T04).

* **API keys** per tenant: ``vg_<env>_<32 random bytes, urlsafe>``. Only the
  SHA-256 is stored; the raw key is shown once at creation. Keys carry scopes.
* **Scopes**: stream · analyze · enroll · evidence:read · feedback:write ·
  liveness · context:write · admin · webhooks:admin
* **Tenant isolation**: a key can only touch ``/v1/tenants/{its tenant}/...``
  and sessions created by its tenant.
* **Rate limit**: token bucket per key (requests per minute, burst = 1 minute).
* **Quotas**: requests/day and audio-minutes/day per key, reset at UTC midnight.
* **mTLS**: terminated at the ingress for REST (B17); the gRPC server can require
  client certificates directly (``grpc_server.server_credentials``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

SCOPES = {
    "stream",
    "analyze",
    "enroll",
    "evidence:read",
    "feedback:write",
    "liveness",
    "context:write",
    "admin",
    "webhooks:admin",
}


class AuthError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class ApiKey:
    key_id: str
    tenant_id: str
    key_hash: str
    scopes: set[str]
    requests_per_minute: int = 600
    requests_per_day: int = 200_000
    audio_minutes_per_day: float = 10_000.0
    revoked: bool = False
    # runtime counters
    tokens: float = field(default=0.0, repr=False)
    last_refill: float = field(default_factory=time.monotonic, repr=False)
    day: str = field(default="", repr=False)
    requests_today: int = field(default=0, repr=False)
    audio_minutes_today: float = field(default=0.0, repr=False)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class KeyStore:
    def __init__(self) -> None:
        self._by_hash: dict[str, ApiKey] = {}
        self._lock = threading.Lock()

    def create(
        self,
        tenant_id: str,
        scopes: set[str],
        env: str = "live",
        raw: str | None = None,
        **limits: float,
    ) -> tuple[str, ApiKey]:
        """Create a key. ``raw`` pins a caller-chosen key (dev bootstrap only); normally generated."""
        bad = scopes - SCOPES
        if bad:
            raise ValueError(f"unknown scopes {bad}")
        if raw is not None and len(raw) < 24:
            raise ValueError("fixed API keys must be at least 24 characters")
        raw = raw or f"vg_{env}_{secrets.token_urlsafe(32)}"
        key = ApiKey(
            key_id=f"key_{secrets.token_hex(6)}",
            tenant_id=tenant_id,
            key_hash=_hash(raw),
            scopes=set(scopes),
            **{k: v for k, v in limits.items()},
        )  # type: ignore[arg-type]
        key.tokens = key.requests_per_minute
        with self._lock:
            self._by_hash[key.key_hash] = key
        return raw, key

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            for k in self._by_hash.values():
                if k.key_id == key_id:
                    k.revoked = True
                    return True
        return False

    def authenticate(self, raw: str | None) -> ApiKey:
        if not raw:
            raise AuthError(401, "missing API key")
        h = _hash(raw)
        with self._lock:
            key = self._by_hash.get(h)
        if key is None or not hmac.compare_digest(key.key_hash, h) or key.revoked:
            raise AuthError(401, "invalid API key")
        return key

    def _roll_day(self, key: ApiKey) -> None:
        today = dt.datetime.now(dt.UTC).date().isoformat()
        if key.day != today:
            key.day, key.requests_today, key.audio_minutes_today = today, 0, 0.0

    def charge_request(self, key: ApiKey) -> None:
        with self._lock:
            now = time.monotonic()
            key.tokens = min(
                key.requests_per_minute,
                key.tokens + (now - key.last_refill) * key.requests_per_minute / 60,
            )
            key.last_refill = now
            if key.tokens < 1:
                raise AuthError(429, "rate limit exceeded")
            self._roll_day(key)
            if key.requests_today >= key.requests_per_day:
                raise AuthError(429, "daily request quota exceeded")
            key.tokens -= 1
            key.requests_today += 1

    def charge_audio(self, key: ApiKey, seconds: float) -> None:
        with self._lock:
            self._roll_day(key)
            if key.audio_minutes_today + seconds / 60 > key.audio_minutes_per_day:
                raise AuthError(429, "daily audio quota exceeded")
            key.audio_minutes_today += seconds / 60


# (method or "*", path regex, scope). First match wins.
ROUTE_SCOPES: list[tuple[str, re.Pattern[str], str]] = [
    ("*", re.compile(r"^/v1/stream$"), "stream"),
    ("*", re.compile(r"^/v1/analyze/"), "analyze"),
    ("*", re.compile(r"^/v1/sessions$"), "stream"),
    ("POST", re.compile(r"^/v1/sessions/[^/]+/audio$"), "stream"),
    ("POST", re.compile(r"^/v1/sessions/[^/]+/close$"), "stream"),
    ("GET", re.compile(r"^/v1/sessions/[^/]+/risk$"), "stream"),
    ("POST", re.compile(r"^/v1/sessions/[^/]+/transcript$"), "context:write"),
    ("*", re.compile(r"^/v1/sessions/[^/]+/challenges"), "liveness"),
    ("GET", re.compile(r"^/v1/sessions/[^/]+/(timeline|evidence)$"), "evidence:read"),
    ("GET", re.compile(r"^/v1/evidence/[^/]+$"), "evidence:read"),
    ("POST", re.compile(r"^/v1/evidence/[^/]+/feedback$"), "feedback:write"),
    ("*", re.compile(r"^/v1/tenants/[^/]+/speakers"), "enroll"),
    ("*", re.compile(r"^/v1/tenants/[^/]+/webhooks"), "webhooks:admin"),
    ("*", re.compile(r"^/v1/tenants/[^/]+/(profile|shadow|keys)"), "admin"),
]
PUBLIC = re.compile(r"^/(healthz|docs|redoc|openapi\.json)$")
TENANT_PATH = re.compile(r"^/v1/tenants/([^/]+)")


def required_scope(method: str, path: str) -> str | None:
    if PUBLIC.match(path):
        return None
    for m, pat, scope in ROUTE_SCOPES:
        if (m == "*" or m == method) and pat.match(path):
            return scope
    return "admin"  # default-deny: anything unlisted needs admin


def authorize(key: ApiKey, method: str, path: str) -> None:
    scope = required_scope(method, path)
    if scope is None:
        return
    if scope not in key.scopes and "admin" not in key.scopes:
        raise AuthError(403, f"API key lacks scope '{scope}'")
    m = TENANT_PATH.match(path)
    if m and m.group(1) != key.tenant_id:
        raise AuthError(403, "API key belongs to a different tenant")


def extract_key(headers: dict[str, str], query: dict[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key") or query.get("api_key")
