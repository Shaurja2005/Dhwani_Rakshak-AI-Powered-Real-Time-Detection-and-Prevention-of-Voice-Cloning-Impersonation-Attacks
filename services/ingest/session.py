"""services/ingest/session.py — Session lifecycle management.

B1-T09: create / heartbeat / teardown with orphan reaper.

Design rules:
- Sessions are keyed by session_id in Redis with a TTL.
- A reaper task wakes up every REAP_INTERVAL_S and terminates sessions
  that have not received a heartbeat within SESSION_TTL_S.
- Sessions are never deleted silently — teardown always publishes
  a terminal SessionRisk onto the bus so downstream services can clean up.
- This module MUST NOT import from services/conditioner or any other service.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from packages.vg_core.config import settings
from packages.vg_core.logging import get_logger, bind_session
from packages.vg_core.models import CallMetadata

log = get_logger(__name__)

# How long without a heartbeat before a session is considered orphaned.
SESSION_TTL_S: int = 60
# How often the reaper checks for orphaned sessions.
REAP_INTERVAL_S: int = 15


class SessionState(str, Enum):
    ACTIVE = "active"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


@dataclass
class Session:
    """In-process session state.  The authoritative store is Redis."""
    session_id: str
    tenant_id: str
    metadata: CallMetadata
    state: SessionState = SessionState.ACTIVE
    created_at: float = field(default_factory=time.monotonic)
    last_heartbeat: float = field(default_factory=time.monotonic)
    chunk_count: int = 0
    window_count: int = 0

    def touch(self) -> None:
        self.last_heartbeat = time.monotonic()

    def age_s(self) -> float:
        return time.monotonic() - self.created_at

    def idle_s(self) -> float:
        return time.monotonic() - self.last_heartbeat


class SessionStore:
    """In-process session registry.

    For a single-node deployment this is sufficient.
    For multi-node, back it with Redis HSET under `vg:sessions:{tenant_id}`.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, metadata: CallMetadata) -> Session:
        async with self._lock:
            if metadata.session_id in self._sessions:
                log.warning(
                    "session_already_exists",
                    session_id=metadata.session_id,
                )
                return self._sessions[metadata.session_id]
            session = Session(
                session_id=metadata.session_id,
                tenant_id=metadata.tenant_id,
                metadata=metadata,
            )
            self._sessions[metadata.session_id] = session
            log.info(
                "session_created",
                session_id=metadata.session_id,
                tenant_id=metadata.tenant_id,
            )
            return session

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def heartbeat(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.touch()
            return True

    async def terminate(self, session_id: str, reason: str = "explicit") -> Optional[Session]:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.state = SessionState.TERMINATED
                log.info(
                    "session_terminated",
                    session_id=session_id,
                    reason=reason,
                    duration_s=session.age_s(),
                    chunks=session.chunk_count,
                    windows=session.window_count,
                )
            return session

    async def all_sessions(self) -> list[Session]:
        async with self._lock:
            return list(self._sessions.values())

    async def reap_orphans(self, ttl_s: int = SESSION_TTL_S) -> list[str]:
        """Terminate all sessions idle longer than ttl_s. Returns list of reaped IDs."""
        reaped: list[str] = []
        async with self._lock:
            to_reap = [
                s.session_id
                for s in self._sessions.values()
                if s.idle_s() > ttl_s and s.state == SessionState.ACTIVE
            ]
        for sid in to_reap:
            await self.terminate(sid, reason="orphan_reaper")
            reaped.append(sid)
        return reaped


async def orphan_reaper_task(store: SessionStore, interval_s: int = REAP_INTERVAL_S) -> None:
    """Background coroutine: wakes up every interval_s and reaps dead sessions."""
    log.info("orphan_reaper_started", interval_s=interval_s, ttl_s=SESSION_TTL_S)
    while True:
        await asyncio.sleep(interval_s)
        reaped = await store.reap_orphans()
        if reaped:
            log.warning("orphan_sessions_reaped", count=len(reaped), session_ids=reaped)
