"""Enrollment REST API (B7-T02).

    POST   /v1/tenants/{tenant_id}/speakers/{speaker_id}/enrollments   add a session (or replace)
    GET    /v1/tenants/{tenant_id}/speakers/{speaker_id}               status (no embeddings)
    DELETE /v1/tenants/{tenant_id}/speakers/{speaker_id}               erasure (DSAR)
    GET    /v1/tenants/{tenant_id}/speakers                             list enrolled speaker ids

Audio is posted as base64 (WAV file bytes, or raw PCM s16le with a sample rate).
AuthN/AuthZ and rate limits are added by the API gateway (B12-T04); gRPC
``Enroll`` streaming is exposed there too.
"""

from __future__ import annotations

import base64
import io
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.enrollment.core import EnrollmentService


class EnrollRequest(BaseModel):
    audio_base64: str
    encoding: Literal["wav", "pcm_s16le"] = "wav"
    sample_rate: int | None = Field(None, ge=8000, le=48000, description="required for pcm_s16le")
    consent_ref: str = Field(
        ..., min_length=1, description="reference to the signed consent record"
    )
    session_ref: str = Field(..., min_length=1)
    replace: bool = False


class EnrollResponse(BaseModel):
    accepted: bool
    speaker_id: str
    reasons: list[str]
    voiced_seconds: float
    snr_db: float
    sessions: int
    conditions: list[str]
    consistency_cosine: float | None


def _decode(req: EnrollRequest) -> tuple[np.ndarray, int]:
    try:
        raw = base64.b64decode(req.audio_base64, validate=True)
    except Exception as exc:
        raise HTTPException(400, "audio_base64 is not valid base64") from exc
    if req.encoding == "wav":
        import soundfile as sf

        try:
            pcm, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:
            raise HTTPException(400, "could not decode WAV audio") from exc
        return pcm, int(sr)
    if not req.sample_rate:
        raise HTTPException(400, "sample_rate is required for pcm_s16le")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, req.sample_rate


def create_app(service: EnrollmentService) -> FastAPI:
    app = FastAPI(title="VoiceGuard Enrollment", version="0.1.0")

    @app.post(
        "/v1/tenants/{tenant_id}/speakers/{speaker_id}/enrollments", response_model=EnrollResponse
    )
    def enroll(tenant_id: str, speaker_id: str, req: EnrollRequest) -> EnrollResponse:
        pcm, sr = _decode(req)
        res = service.enroll(
            tenant_id, speaker_id, pcm, sr, req.consent_ref, req.session_ref, req.replace
        )
        del pcm  # raw audio never outlives the request (I5)
        body = EnrollResponse(**res.__dict__)
        if not res.accepted:
            raise HTTPException(422, detail=body.model_dump())
        return body

    @app.get("/v1/tenants/{tenant_id}/speakers/{speaker_id}")
    def status(tenant_id: str, speaker_id: str) -> dict[str, object]:
        vp = service.vault.get(tenant_id, speaker_id)
        if vp is None:
            raise HTTPException(404, "speaker not enrolled")
        return {
            "speaker_id": vp.speaker_id,
            "embedder": vp.embedder,
            "sessions": [s.__dict__ for s in vp.sessions],
            "conditions": sorted(vp.embeddings),
            "created_at": vp.created_at,
            "updated_at": vp.updated_at,
            "expires_at": vp.expires_at,
            "expired": vp.expired(),
        }

    @app.delete("/v1/tenants/{tenant_id}/speakers/{speaker_id}", status_code=204)
    def erase(tenant_id: str, speaker_id: str) -> None:
        if not service.vault.delete_speaker(tenant_id, speaker_id):
            raise HTTPException(404, "speaker not enrolled")

    @app.get("/v1/tenants/{tenant_id}/speakers")
    def list_speakers(tenant_id: str) -> dict[str, list[str]]:
        return {"speakers": service.vault.list_speakers(tenant_id)}

    return app
