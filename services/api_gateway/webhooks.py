"""Tenant webhook subscriptions with signed, retried, asynchronous delivery (B12-T03).

Signing and retry logic live in ``services.policy.notify`` (HMAC-SHA256,
``X-VoiceGuard-Signature: t=<unix>,v1=<hex>``, exponential backoff, no retry on
4xx except 429). This module adds per-tenant subscriptions (URL, secret, event
types) and a background delivery pool so a slow receiver never blocks the call
path (I10). Only HTTPS URLs are accepted.
"""

from __future__ import annotations

import concurrent.futures as cf
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from packages.vg_core.logging import get_logger
from services.policy.notify import WebhookResult, deliver_webhook

log = get_logger(__name__)
EVENT_TYPES = {"policy_decision", "session_risk", "context_signals", "window_score"}


@dataclass
class Subscription:
    sub_id: str
    tenant_id: str
    url: str
    secret: str
    events: set[str] = field(default_factory=lambda: {"policy_decision"})
    active: bool = True


class WebhookRegistry:
    def __init__(self, max_workers: int = 4, **delivery_kw: Any) -> None:
        self._subs: dict[str, Subscription] = {}
        self._lock = threading.Lock()
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="webhook")
        self._delivery_kw = delivery_kw

    def subscribe(self, tenant_id: str, url: str, events: set[str] | None = None) -> Subscription:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("webhook URL must be https")
        ev = events or {"policy_decision"}
        if ev - EVENT_TYPES:
            raise ValueError(f"unknown event types {ev - EVENT_TYPES}")
        sub = Subscription(
            f"wh_{secrets.token_hex(6)}", tenant_id, url, secrets.token_urlsafe(32), set(ev)
        )
        with self._lock:
            self._subs[sub.sub_id] = sub
        return sub

    def list(self, tenant_id: str) -> list[Subscription]:
        with self._lock:
            return [s for s in self._subs.values() if s.tenant_id == tenant_id and s.active]

    def remove(self, tenant_id: str, sub_id: str) -> bool:
        with self._lock:
            s = self._subs.get(sub_id)
            if s is None or s.tenant_id != tenant_id:
                return False
            s.active = False
            return True

    def publish(
        self, tenant_id: str, event_type: str, payload: dict[str, Any]
    ) -> list[cf.Future[WebhookResult]]:
        futures = []
        for s in self.list(tenant_id):
            if event_type in s.events:
                body = {"type": event_type, "tenant_id": tenant_id, "data": payload}
                futures.append(
                    self._pool.submit(deliver_webhook, s.url, body, s.secret, **self._delivery_kw)
                )
        return futures
