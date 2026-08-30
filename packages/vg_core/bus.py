"""vg_core.bus — Internal message bus abstraction (Redis Streams).

All cross-service communication goes through this bus using the contracts in
SOURCE_OF_TRUTH.md §4.  Services NEVER import from each other directly.

This module provides a thin async wrapper around Redis Streams so that:
- Services can publish and subscribe to typed message streams.
- The stream key naming is canonical and consistent.
- Tests can inject a fake bus without touching Redis.

Stream key naming convention::

    vg:{tenant_id}:{message_type}  e.g.  vg:demo:analysis_window
    vg:system:{message_type}       e.g.  vg:system:session_risk (for broadcast)

Usage::

    from packages.vg_core.bus import get_bus, BusMessage

    bus = await get_bus()
    await bus.publish("vg:demo:analysis_window", window.model_dump_json())

    async for msg in bus.subscribe("vg:demo:analysis_window", group="conditioner"):
        window = AnalysisWindow.model_validate_json(msg.data)
        ...
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from packages.vg_core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BusMessage:
    stream: str
    message_id: str
    data: str          # JSON-serialised payload
    headers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract bus interface (for testing / DI)
# ---------------------------------------------------------------------------


class AbstractBus(ABC):
    @abstractmethod
    async def publish(self, stream: str, data: str, headers: Optional[dict[str, str]] = None) -> str:
        """Publish a message.  Returns the message ID."""
        ...

    @abstractmethod
    async def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str = "default",
        block_ms: int = 1000,
    ) -> AsyncIterator[BusMessage]:
        """Yield messages from the given stream."""
        yield  # type: ignore[misc]

    @abstractmethod
    async def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Redis Streams implementation
# ---------------------------------------------------------------------------


class RedisBus(AbstractBus):
    """Redis Streams-backed bus.  Requires redis[hiredis]."""

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: Any = None  # redis.asyncio.Redis

    async def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "redis[hiredis] is required for RedisBus.  "
                    "pip install redis[hiredis]"
                ) from exc
            self._client = await aioredis.from_url(self._url, decode_responses=True)
        return self._client

    async def publish(
        self,
        stream: str,
        data: str,
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        client = await self._ensure_client()
        fields: dict[str, str] = {"data": data}
        if headers:
            fields["headers"] = json.dumps(headers)
        msg_id: str = await client.xadd(stream, fields)
        return msg_id

    async def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str = "default",
        block_ms: int = 1000,
    ) -> AsyncIterator[BusMessage]:
        client = await self._ensure_client()
        # Create consumer group if it doesn't exist
        try:
            await client.xgroup_create(stream, group, id="$", mkstream=True)
        except Exception:  # noqa: BLE001
            pass  # group already exists

        while True:
            results = await client.xreadgroup(
                group, consumer, {stream: ">"}, count=10, block=block_ms
            )
            if not results:
                await asyncio.sleep(0)
                continue
            for _stream, messages in results:
                for msg_id, fields in messages:
                    raw_headers = fields.get("headers")
                    headers_dict: dict[str, str] = (
                        json.loads(raw_headers) if raw_headers else {}
                    )
                    yield BusMessage(
                        stream=stream,
                        message_id=msg_id,
                        data=fields.get("data", ""),
                        headers=headers_dict,
                    )
                    await client.xack(stream, group, msg_id)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# In-memory bus for testing
# ---------------------------------------------------------------------------


class InMemoryBus(AbstractBus):
    """Fake bus backed by asyncio.Queue.  Use this in unit tests."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[BusMessage]] = {}
        self._msg_counter = 0

    def _get_queue(self, stream: str) -> asyncio.Queue[BusMessage]:
        if stream not in self._queues:
            self._queues[stream] = asyncio.Queue()
        return self._queues[stream]

    async def publish(
        self,
        stream: str,
        data: str,
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        self._msg_counter += 1
        msg_id = str(self._msg_counter)
        msg = BusMessage(stream=stream, message_id=msg_id, data=data, headers=headers or {})
        await self._get_queue(stream).put(msg)
        return msg_id

    async def subscribe(
        self,
        stream: str,
        group: str = "default",
        consumer: str = "default",
        block_ms: int = 1000,
    ) -> AsyncIterator[BusMessage]:
        q = self._get_queue(stream)
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=block_ms / 1000)
                yield msg
            except asyncio.TimeoutError:
                return

    async def close(self) -> None:
        self._queues.clear()


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_bus_instance: Optional[AbstractBus] = None


async def get_bus() -> AbstractBus:
    """Return the singleton bus instance, initialising it on first call."""
    global _bus_instance
    if _bus_instance is None:
        from packages.vg_core.config import settings
        if settings.env == "test":
            _bus_instance = InMemoryBus()
        else:
            _bus_instance = RedisBus(settings.redis.url)
        log.info("bus_initialised", bus_type=type(_bus_instance).__name__)
    return _bus_instance


def set_bus(bus: AbstractBus) -> None:
    """Override the singleton bus (use in tests)."""
    global _bus_instance
    _bus_instance = bus
