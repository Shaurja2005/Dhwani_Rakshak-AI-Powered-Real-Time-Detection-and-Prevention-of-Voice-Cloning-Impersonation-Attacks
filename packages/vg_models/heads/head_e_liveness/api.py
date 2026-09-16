"""Operator-triggered challenge hook (B8-T03).

REST endpoints the agent UI (B13) calls to issue a challenge mid-call, and that
B10's ASR stream (or the agent, manually) uses to submit the caller's answer:

    POST .../challenges                          issue; returns prompt text for the agent
    POST .../challenges/{challenge_id}/prompt-end  when the prompt finished (call ms)
    POST .../challenges/{challenge_id}/response    ASR tokens of the answer
    GET  .../challenges/active                   current challenge + verification

    (all under /v1/sessions/{session_id})

Mounted by the API gateway (B12), which adds authentication.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.vg_models.heads.head_e_liveness.challenge import ChallengeRegistry, generate
from packages.vg_models.heads.head_e_liveness.verifier import verify


class IssueRequest(BaseModel):
    kind: Literal["digits_codeswitch", "nonce_phrase", "reverse_order"] | None = None
    languages: list[str] = Field(default_factory=lambda: ["en", "hi"])


class PromptEnd(BaseModel):
    prompt_end_ms: int = Field(..., ge=0)


class ResponseToken(BaseModel):
    text: str
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)


class ResponseBody(BaseModel):
    tokens: list[ResponseToken]


def create_router(registry: ChallengeRegistry) -> APIRouter:
    r = APIRouter(prefix="/v1/sessions/{session_id}/challenges", tags=["liveness"])

    @r.post("")
    def issue(session_id: str, req: IssueRequest) -> dict[str, object]:
        ch = registry.issue(generate(session_id, req.kind, tuple(req.languages)))
        return {
            "challenge_id": ch.challenge_id,
            "kind": ch.kind,
            "prompt_text": ch.prompt_text,
            "expires_in_s": ch.ttl_s,
        }

    @r.post("/{challenge_id}/prompt-end", status_code=204)
    def prompt_end(session_id: str, challenge_id: str, body: PromptEnd) -> None:
        ch = registry.active(session_id)
        if ch is None or ch.challenge_id != challenge_id:
            raise HTTPException(404, "no such active challenge")
        registry.mark_prompt_end(session_id, body.prompt_end_ms)

    @r.post("/{challenge_id}/response")
    def response(session_id: str, challenge_id: str, body: ResponseBody) -> dict[str, object]:
        tokens = [(t.text, t.start_ms, t.end_ms) for t in body.tokens]
        if not registry.submit_response(session_id, challenge_id, tokens):
            raise HTTPException(404, "no such active challenge")
        ch = registry.active(session_id)
        assert ch is not None
        v = verify(ch, tokens)
        return {"content_match": v.content_match, "latency_ms": v.latency_ms, "reasons": v.reasons}

    @r.get("/active")
    def active(session_id: str) -> dict[str, object]:
        ch = registry.active(session_id)
        if ch is None:
            raise HTTPException(404, "no active challenge")
        resp = registry.response(ch.challenge_id)
        out: dict[str, object] = {
            "challenge_id": ch.challenge_id,
            "kind": ch.kind,
            "prompt_text": ch.prompt_text,
            "expired": ch.expired,
            "answered": resp is not None,
        }
        if resp is not None:
            v = verify(ch, resp.tokens)
            out.update(content_match=v.content_match, latency_ms=v.latency_ms, reasons=v.reasons)
        return out

    return r
