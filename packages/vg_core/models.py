"""vg_core.models — Pydantic v2 models for all VoiceGuard data contracts.

These models are the authoritative Python representation of the JSON Schemas
in ``schemas/`` and the protobuf messages in ``proto/voiceguard.proto``.

Field names, types, and semantics are defined in SOURCE_OF_TRUTH.md §4.
Do NOT rename fields without an ADR.

Generated-code note:  In production these could be generated with
``datamodel-code-generator``.  During the bootstrap phase they are hand-written
to avoid a hard dependency on the code-gen tool in CI.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations  (canonical vocabulary — SOURCE_OF_TRUTH §3)
# ---------------------------------------------------------------------------


class RiskState(str, Enum):
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    ABSTAIN = "ABSTAIN"


class Channel(str, Enum):
    PSTN = "pstn"
    VOIP = "voip"
    WEBRTC = "webrtc"
    MOBILE = "mobile"
    FILE = "file"


class AudioEncoding(str, Enum):
    PCM_S16LE = "pcm_s16le"
    MULAW = "mulaw"
    ALAW = "alaw"


class ConsentBasis(str, Enum):
    LEGITIMATE_USE = "legitimate_use"
    EXPLICIT_CONSENT = "explicit_consent"
    NONE = "none"


class HeadID(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class AbstainReason(str, Enum):
    NO_ENROLLMENT = "no_enrollment"
    INSUFFICIENT_SPEECH = "insufficient_speech"
    TIMEOUT = "timeout"
    QUALITY_GATE = "quality_gate"
    NO_CHALLENGE = "no_challenge"
    UNTRAINED = "untrained"  # ADR 0006: head has no trained weights loaded


# ---------------------------------------------------------------------------
# §4.1 CallMetadata
# ---------------------------------------------------------------------------


class CallMetadata(BaseModel):
    """One per session, emitted at session start.  SOURCE_OF_TRUTH §4.1"""

    session_id: str = Field(..., description="UUIDv7")
    tenant_id: str
    direction: str = Field(..., pattern="^(inbound|outbound)$")
    started_at: datetime
    caller_number: Optional[str] = None
    callee_number: Optional[str] = None
    claimed_identity_id: Optional[str] = Field(
        None, description="Keys the voiceprint lookup"
    )
    channel: Channel
    codec_hint: str = Field(
        ...,
        pattern="^(g711u|g711a|g729|amrnb|opus|evs|pcm|unknown)$",
    )
    source_sample_rate: int = Field(..., ge=8000, le=96000)
    trunk_id: Optional[str] = None
    source_asn: Optional[str] = None
    sip_headers: dict[str, str] = Field(default_factory=dict)
    language_hint: Optional[str] = None
    consent_basis: ConsentBasis
    shadow_mode: bool = True


# ---------------------------------------------------------------------------
# §4.2 AudioChunk
# ---------------------------------------------------------------------------


class AudioChunk(BaseModel):
    """Transport-level audio packet.  SOURCE_OF_TRUTH §4.2"""

    session_id: str
    seq: int = Field(..., ge=0)
    ts_ms: int = Field(..., ge=0)
    sample_rate: int = Field(..., ge=8000)
    encoding: AudioEncoding
    payload: bytes


# ---------------------------------------------------------------------------
# §4.3 AnalysisWindow
# ---------------------------------------------------------------------------


class AnalysisWindow(BaseModel):
    """Output of B2; input to every detection head.  SOURCE_OF_TRUTH §4.3"""

    session_id: str
    window_id: int = Field(..., ge=0)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    sample_rate: Literal[16000] = 16000  # always 16 kHz
    samples_ref: str = Field(
        ..., description="shm:// reference — never persisted (I5)"
    )
    voiced_ms: int = Field(..., ge=0)
    snr_db: float
    clipping_ratio: float = Field(..., ge=0.0, le=1.0)
    quality_ok: bool
    quality_flags: list[str] = Field(default_factory=list)
    original_sample_rate: int = Field(
        ..., description="Source SR before resampling — a feature, not just plumbing"
    )
    detected_codec: Optional[str] = None

    @model_validator(mode="after")
    def end_after_start(self) -> "AnalysisWindow":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


# ---------------------------------------------------------------------------
# §4.4 HeadScore
# ---------------------------------------------------------------------------


class HeadScore(BaseModel):
    """Exactly one per head per analysis window.  SOURCE_OF_TRUTH §4.4"""

    session_id: str
    window_id: int = Field(..., ge=0)
    head_id: HeadID
    raw_score: float
    p_spoof: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Calibrated probability; null when abstain=True",
    )
    abstain: bool
    abstain_reason: Optional[AbstainReason] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: int = Field(..., ge=0)
    model_version: str
    calibration_version: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def p_spoof_required_when_not_abstain(self) -> "HeadScore":
        if not self.abstain and self.p_spoof is None:
            raise ValueError("p_spoof must be set when abstain is False")
        return self


# ---------------------------------------------------------------------------
# §4.5 FusedWindowScore
# ---------------------------------------------------------------------------


class FusedWindowScore(BaseModel):
    """Fusion output per window.  SOURCE_OF_TRUTH §4.5"""

    session_id: str
    window_id: int = Field(..., ge=0)
    p_spoof: float = Field(..., ge=0.0, le=1.0)
    state: RiskState
    contributions: dict[str, float] = Field(
        default_factory=dict,
        description="head_id -> contribution weight",
    )
    heads_abstained: list[str] = Field(default_factory=list)
    fusion_version: str


# ---------------------------------------------------------------------------
# §4.6 ContextSignals
# ---------------------------------------------------------------------------


class TranscriptSnippet(BaseModel):
    start_ms: int
    text: str
    label: str


class ContextSignals(BaseModel):
    """Output of B10 context/intent enrichment.  SOURCE_OF_TRUTH §4.6"""

    session_id: str
    as_of_ms: int = Field(..., ge=0)
    metadata_risk: float = Field(..., ge=0.0, le=1.0)
    transaction_risk: float = Field(..., ge=0.0, le=1.0)
    intent_risk: float = Field(..., ge=0.0, le=1.0)
    intent_labels: list[str] = Field(default_factory=list)
    transcript_snippets: list[TranscriptSnippet] = Field(default_factory=list)
    language_detected: str
    code_switched: bool


# ---------------------------------------------------------------------------
# §4.7 SessionRisk
# ---------------------------------------------------------------------------


class RiskDriver(BaseModel):
    factor: str
    weight: float
    detail: str


class WindowSummary(BaseModel):
    window_id: int
    p_spoof: float
    state: RiskState


class SessionRisk(BaseModel):
    """Headline session-level output.  SOURCE_OF_TRUTH §4.7"""

    session_id: str
    updated_at: datetime
    risk_score: int = Field(..., ge=0, le=100)
    state: RiskState
    p_spoof_session_max: float = Field(
        ..., ge=0.0, le=1.0, description="Any-segment trigger"
    )
    p_spoof_session_mean: float = Field(..., ge=0.0, le=1.0)
    drivers: list[RiskDriver] = Field(default_factory=list)
    timeline: list[WindowSummary] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    abstain_ratio: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# §4.8 PolicyDecision
# ---------------------------------------------------------------------------


VALID_ACTIONS = {
    "agent_banner",
    "force_callback",
    "hold_transaction",
    "supervisor_escalate",
    "soc_ticket",
    "flag_recording",
    "send_sms",
    "send_email",
    "push_notification",
    "siem_webhook",
    "step_up_mfa",
    "dual_approval",
}


class PolicyDecision(BaseModel):
    """Action output from B11.  SOURCE_OF_TRUTH §4.8"""

    session_id: str
    decided_at: datetime
    threshold_profile: str
    actions: list[str]
    rationale: str
    evidence_bundle_id: str = Field(..., description="SHA256 content hash")
    shadow_mode: bool
    suppressed: bool

    @field_validator("actions")
    @classmethod
    def valid_action_names(cls, v: list[str]) -> list[str]:
        unknown = set(v) - VALID_ACTIONS
        if unknown:
            raise ValueError(f"Unknown action(s): {unknown}")
        return v


# ---------------------------------------------------------------------------
# §4.9 SessionContext — passed to heads alongside AnalysisWindow
# ---------------------------------------------------------------------------


class SessionContext(BaseModel):
    """Per-session context passed to every head alongside the window.

    Heads use this for speaker lookup, tenant configuration, etc.
    It is NOT a data-contract message; it is an internal execution context.
    """

    session_id: str
    tenant_id: str
    claimed_identity_id: Optional[str] = None
    shadow_mode: bool = True
    call_metadata: Optional[CallMetadata] = None
    model_config = {"arbitrary_types_allowed": True}
