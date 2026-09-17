"""fusion service entrypoint.

Serves the forensic timeline replay API. The streaming path (consume HeadScores
from the bus, run one RiskEngine per session, publish FusedWindowScore /
SessionRisk) is wired by the API gateway / pipeline integration in B12; the
engine itself is ``services.fusion.engine.RiskEngine``.

Env:
    VG_TIMELINE_DB   SQLite path for dev (default data/timeline/timeline.db)
    VG_FUSION_PORT   default 8020
"""

from __future__ import annotations

import os
from pathlib import Path

from services.fusion.persistence import SQLiteTimelineStore, create_router


def main() -> None:
    import uvicorn
    from fastapi import FastAPI

    path = Path(os.getenv("VG_TIMELINE_DB", "data/timeline/timeline.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="VoiceGuard Fusion", version="0.1.0")
    app.include_router(create_router(SQLiteTimelineStore(path)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VG_FUSION_PORT", "8020")))  # noqa: S104


if __name__ == "__main__":
    main()
