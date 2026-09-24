"""Inference service entrypoint (B14-T04) — CPU-tier stand-in for Triton.

Serves Head A through the batching server over HTTP so that several gateway
replicas can share one model host. Internal network only: audio never leaves
the deployment (I6) and nothing is persisted (I5).

    VG_INFERENCE_BACKEND=torch|torch-int8|onnx|proxy python -m services.inference.main

POST /v1/infer/head_a   {"wav_f32_base64": "...", "timeout_ms": 400}
                        → {"raw_score": float, "model_version": str, "trained": bool}
                        503 when the queue is full (caller abstains, I10)
GET  /healthz, /v1/stats
"""

from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.inference.backends import InferenceBackend, backend_from_env
from services.inference.batching import BatchingInferenceServer, OverloadedError


class InferRequest(BaseModel):
    wav_f32_base64: str
    timeout_ms: float = 400.0


def create_app(backend: InferenceBackend | None = None, **batch_kwargs: float) -> FastAPI:
    server = BatchingInferenceServer(backend or backend_from_env(), **batch_kwargs)  # type: ignore[arg-type]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        server.close()

    app = FastAPI(title="VoiceGuard inference", version="0.1.0", lifespan=lifespan)
    app.state.server = server

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "ok": True,
            "backend": server.backend.name,
            "model_version": server.backend.model_version,
        }

    @app.get("/v1/stats")
    def stats() -> dict[str, object]:
        s = server.stats
        return {
            "batches": s.batches,
            "items": s.items,
            "mean_batch": round(s.mean_batch, 2),
            "rejected": s.rejected,
            "expired": s.expired,
            "ema_batch_ms": round(s.ema_batch_ms, 2),
            "queue_depth": server.queue_depth,
        }

    @app.post("/v1/infer/head_a")
    def infer(req: InferRequest) -> dict[str, object]:
        try:
            pcm = np.frombuffer(base64.b64decode(req.wav_f32_base64, validate=True), dtype="<f4")
        except ValueError as exc:
            raise HTTPException(400, "wav_f32_base64 must be base64 little-endian float32") from exc
        if pcm.size < 1600:
            raise HTTPException(400, "need at least 0.1 s of 16 kHz audio")
        try:
            raw = server.infer(pcm, timeout_ms=req.timeout_ms)
        except (OverloadedError, TimeoutError) as exc:
            raise HTTPException(503, f"overloaded: {exc}") from exc
        return {
            "raw_score": raw,
            "model_version": server.backend.model_version,
            "trained": server.backend.trained,
        }

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("VG_INFERENCE_HOST", "127.0.0.1"),
        port=int(os.getenv("VG_INFERENCE_PORT", "8600")),
    )


if __name__ == "__main__":
    main()
