"""context service entrypoint (B10).

REST (behind the API gateway, B12):
    POST /v1/sessions                              start a context session with CallMetadata
    POST /v1/sessions/{session_id}/transcript      append ASR segments (redacted on ingest)
    GET  /v1/sessions/{session_id}/context         ContextSignals + explanations

Env:
    VG_CONTEXT_LLM_BASE_URL   local Ollama / vLLM (OpenAI-compatible) endpoint
    VG_CONTEXT_LLM_MODEL      e.g. qwen2.5:7b-instruct
    VG_CONTEXT_LLM_ENABLED    1 to enable the LLM classifier (rules always on)
    VG_CONTEXT_PORT           default 8030
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from packages.vg_core.config import settings
from packages.vg_core.logging import get_logger
from packages.vg_core.models import CallMetadata
from services.context.asr import Segment
from services.context.engine import ContextEngine
from services.context.intent import LLMIntentClassifier, NonLocalEndpointError

log = get_logger(__name__)


class SegmentIn(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None
    final: bool = True
    words: list[tuple[str, int, int]] = []


def build_llm() -> LLMIntentClassifier | None:
    if os.getenv("VG_CONTEXT_LLM_ENABLED") != "1":
        return None
    try:
        return LLMIntentClassifier(
            settings.context.llm_base_url,
            settings.context.llm_model,
            timeout_s=settings.context.max_latency_ms / 1000,
        )
    except NonLocalEndpointError as exc:
        log.error("intent_llm_rejected", error=str(exc))
        return None


def create_app(llm: LLMIntentClassifier | None = None) -> FastAPI:
    app = FastAPI(title="VoiceGuard Context", version="0.1.0")
    engines: dict[str, ContextEngine] = {}

    @app.post("/v1/sessions", status_code=201)
    def start(meta: CallMetadata) -> dict[str, str]:
        engines[meta.session_id] = ContextEngine(meta, llm=llm)
        return {"session_id": meta.session_id}

    @app.post("/v1/sessions/{session_id}/transcript", status_code=204)
    def transcript(session_id: str, segments: list[SegmentIn]) -> None:
        eng = engines.get(session_id)
        if eng is None:
            raise HTTPException(404, "unknown session")
        eng.add_segments(
            [
                Segment(s.start_ms, s.end_ms, s.text, s.language, s.final, 1.0, list(s.words))
                for s in segments
            ]
        )

    @app.get("/v1/sessions/{session_id}/context")
    def context(session_id: str) -> dict[str, object]:
        eng = engines.get(session_id)
        if eng is None:
            raise HTTPException(404, "unknown session")
        rep = eng.report()
        return {
            "signals": rep.signals.model_dump(mode="json"),
            "explanations": rep.explanations,
            "transaction_tier": rep.transaction_tier,
        }

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(build_llm()), host="0.0.0.0", port=int(os.getenv("VG_CONTEXT_PORT", "8030"))  # noqa: S104
    )


if __name__ == "__main__":
    main()
