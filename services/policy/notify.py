"""Multi-channel notification with signed SIEM webhooks (B11-T05).

Channels:
* ``websocket``     — in-process pub/sub the agent UI (B13) subscribes to per tenant
* ``send_sms`` / ``send_email`` / ``push_notification`` — provider interfaces;
  the default providers only record what would be sent (no external calls in dev)
* ``siem_webhook``  — HTTPS POST, body signed with HMAC-SHA256
  (``X-VoiceGuard-Signature: t=<unix>,v1=<hex>`` over ``"<t>.<body>"``), retried
  with exponential backoff on network errors and 5xx. B12-T03 reuses this.

In shadow mode nothing is dispatched; the decision is only recorded (B11-T07).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from packages.vg_core.logging import get_logger

log = get_logger(__name__)


def sign(secret: str, body: bytes, timestamp: int | None = None) -> str:
    t = int(timestamp if timestamp is not None else time.time())
    mac = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


def verify_signature(
    secret: str, body: bytes, header: str, tolerance_s: int = 300, now: float | None = None
) -> bool:
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        t = int(parts["t"])
    except (KeyError, ValueError):
        return False
    if abs((now or time.time()) - t) > tolerance_s:
        return False
    expected = sign(secret, body, t).split("v1=")[1]
    return hmac.compare_digest(expected, parts.get("v1", ""))


@dataclass
class WebhookResult:
    delivered: bool
    attempts: int
    status: int | None
    error: str | None = None


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    secret: str,
    max_attempts: int = 5,
    base_delay_s: float = 0.5,
    post: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> WebhookResult:
    import httpx

    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    poster = post or httpx.post
    status = None
    err = None
    for attempt in range(1, max_attempts + 1):
        headers = {"Content-Type": "application/json", "X-VoiceGuard-Signature": sign(secret, body)}
        try:
            r = poster(url, content=body, headers=headers, timeout=5.0)
            status = r.status_code
            if 200 <= status < 300:
                return WebhookResult(True, attempt, status)
            if 400 <= status < 500 and status != 429:
                return WebhookResult(False, attempt, status, "client error; not retried")
            err = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - network errors are retried
            err = type(exc).__name__
        if attempt < max_attempts:
            sleep(base_delay_s * 2 ** (attempt - 1))
    return WebhookResult(False, max_attempts, status, err)


class Broadcaster:
    """Tenant-scoped in-process pub/sub for the agent UI WebSocket."""

    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue[dict[str, Any]]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, tenant_id: str) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs[tenant_id].append(q)
        return q

    def unsubscribe(self, tenant_id: str, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subs[tenant_id]:
                self._subs[tenant_id].remove(q)

    def publish(self, tenant_id: str, message: dict[str, Any]) -> int:
        with self._lock:
            subs = list(self._subs[tenant_id])
        for q in subs:
            try:
                q.put_nowait(message)
            except queue.Full:
                log.warning("ws_subscriber_backlog", tenant_id=tenant_id)
        return len(subs)


@dataclass
class RecordingProvider:
    """Dev provider for SMS / email / push: records instead of sending."""

    sent: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def send(self, channel: str, message: dict[str, Any]) -> None:
        self.sent.append((channel, message))


@dataclass
class WebhookTarget:
    url: str
    secret: str


class Notifier:
    def __init__(
        self,
        broadcaster: Broadcaster | None = None,
        provider: RecordingProvider | None = None,
        webhooks: dict[str, WebhookTarget] | None = None,
        post: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.broadcaster = broadcaster or Broadcaster()
        self.provider = provider or RecordingProvider()
        self.webhooks = webhooks or {}
        self._post = post
        self._sleep = sleep

    def dispatch(
        self, tenant_id: str, channels: list[str], message: dict[str, Any], shadow: bool
    ) -> dict[str, Any]:
        if shadow:
            return {"dispatched": False, "reason": "shadow mode"}
        report: dict[str, Any] = {"dispatched": True}
        for ch in dict.fromkeys(channels):
            if ch == "websocket":
                report[ch] = self.broadcaster.publish(tenant_id, message)
            elif ch == "siem_webhook":
                target = self.webhooks.get(tenant_id)
                if target is None:
                    report[ch] = "not configured"
                    continue
                res = deliver_webhook(
                    target.url, message, target.secret, post=self._post, sleep=self._sleep
                )
                report[ch] = {
                    "delivered": res.delivered,
                    "attempts": res.attempts,
                    "status": res.status,
                }
            elif ch in ("send_sms", "send_email", "push_notification"):
                self.provider.send(ch, message)
                report[ch] = "queued"
        return report
