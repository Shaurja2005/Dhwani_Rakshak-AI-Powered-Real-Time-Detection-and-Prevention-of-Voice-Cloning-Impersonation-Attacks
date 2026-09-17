"""Score timeline persistence + forensic replay (B9-T08).

``SQLiteTimelineStore`` for dev/tests; production uses the same schema on
PostgreSQL + TimescaleDB (``TIMESCALE_DDL``, hypertable on ``ts``), which B17
provisions. Only scores and versions are stored — never audio (I5).

``create_router`` exposes ``GET /v1/sessions/{session_id}/timeline`` for the
post-call forensics view (B13-T04); the API gateway (B12) adds auth.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

TIMESCALE_DDL = """
CREATE TABLE IF NOT EXISTS score_timeline (
    session_id           UUID        NOT NULL,
    window_id            INTEGER     NOT NULL,
    ts                   TIMESTAMPTZ NOT NULL,
    p_window             DOUBLE PRECISION,
    window_state         TEXT        NOT NULL,
    p_session            DOUBLE PRECISION NOT NULL,
    p_hmm                DOUBLE PRECISION NOT NULL,
    p_segment_max        DOUBLE PRECISION NOT NULL,
    state                TEXT        NOT NULL,
    risk_score           SMALLINT    NOT NULL,
    contributions        JSONB       NOT NULL,
    heads_abstained      JSONB       NOT NULL,
    fusion_version       TEXT        NOT NULL,
    calibration_versions JSONB       NOT NULL,
    model_versions       JSONB       NOT NULL,
    PRIMARY KEY (session_id, window_id, ts)
);
SELECT create_hypertable('score_timeline', 'ts', if_not_exists => TRUE);
"""

_JSON = ("contributions", "heads_abstained", "calibration_versions", "model_versions")


@dataclass
class TimelineRow:
    session_id: str
    window_id: int
    ts: str
    p_window: float | None
    window_state: str
    p_session: float
    p_hmm: float
    p_segment_max: float
    state: str
    risk_score: int
    contributions: dict[str, float] = field(default_factory=dict)
    heads_abstained: list[str] = field(default_factory=list)
    fusion_version: str = ""
    calibration_versions: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)


class TimelineStore(Protocol):
    def append(self, row: TimelineRow) -> None: ...

    def session(self, session_id: str) -> list[TimelineRow]: ...


class SQLiteTimelineStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        cols = ", ".join(f"{k} TEXT" for k in TimelineRow.__dataclass_fields__)
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS score_timeline ({cols})"
        )  # noqa: S608 - fixed columns
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_tl ON score_timeline (session_id, window_id)"
        )
        self._db.commit()

    def append(self, row: TimelineRow) -> None:
        d = asdict(row)
        vals = [
            json.dumps(d[k]) if k in _JSON else (None if d[k] is None else str(d[k])) for k in d
        ]
        with self._lock:
            placeholders = ", ".join("?" * len(vals))
            sql = f"INSERT INTO score_timeline VALUES ({placeholders})"  # noqa: S608 - "?" only
            self._db.execute(sql, vals)
            self._db.commit()

    def session(self, session_id: str) -> list[TimelineRow]:
        with self._lock:
            cur = self._db.execute(
                "SELECT * FROM score_timeline WHERE session_id=? "
                "ORDER BY CAST(window_id AS INTEGER)",
                (session_id,),
            )
            names = [c[0] for c in cur.description]
            rows = cur.fetchall()
        types = {k: f.type for k, f in TimelineRow.__dataclass_fields__.items()}
        out = []
        for r in rows:
            d: dict[str, object] = {}
            for k, v in zip(names, r, strict=True):
                if k in _JSON:
                    d[k] = json.loads(v)
                elif v is None:
                    d[k] = None
                elif "int" in str(types[k]):
                    d[k] = int(v)
                elif "float" in str(types[k]):
                    d[k] = float(v)
                else:
                    d[k] = v
            out.append(TimelineRow(**d))  # type: ignore[arg-type]
        return out


def create_router(store: TimelineStore):  # type: ignore[no-untyped-def]  # noqa: ANN201
    from fastapi import APIRouter, HTTPException

    r = APIRouter(tags=["forensics"])

    @r.get("/v1/sessions/{session_id}/timeline")
    def timeline(session_id: str) -> dict[str, object]:
        rows = store.session(session_id)
        if not rows:
            raise HTTPException(404, "no timeline for session")
        return {"session_id": session_id, "windows": [asdict(x) for x in rows]}

    return r
