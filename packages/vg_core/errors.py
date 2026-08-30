"""vg_core.errors — Error taxonomy for VoiceGuard.

All VoiceGuard exceptions derive from VGError.  The taxonomy is:

    VGError
    ├── ConfigError       — bad configuration or missing required env var
    ├── ContractError     — schema/contract violation (a bug, not a user error)
    ├── AudioError
    │   ├── BadAudioFormat   — unrecognised encoding or sample rate
    │   └── QualityGateError — audio rejected by the quality gate
    ├── HeadError
    │   ├── HeadTimeoutError — head exceeded its budget_ms (MUST abstain, not raise)
    │   └── HeadNotFoundError
    ├── EnrollmentError
    │   ├── NoEnrollmentError
    │   └── InsufficientEnrollmentAudioError
    └── PolicyError

Design rule: heads MUST NOT propagate exceptions up the audio path.  A head
that encounters an error should catch it internally and return HeadScore with
``abstain=True`` (invariant I2, I10).  VGError subclasses are for non-audio
control paths (config, enrollment, policy) where a proper exception is fine.
"""
from __future__ import annotations

from typing import Optional


class VGError(Exception):
    """Base class for all VoiceGuard-specific exceptions."""

    code: str = "VG_ERROR"

    def __init__(self, message: str, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.detail = detail

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": str(self), "detail": self.detail}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(VGError):
    """Raised when required configuration is missing or invalid."""

    code = "VG_CONFIG_ERROR"


# ---------------------------------------------------------------------------
# Contract violations
# ---------------------------------------------------------------------------


class ContractError(VGError):
    """Raised when a service produces output that violates a schema contract.
    This is always a programmer error, not a user error.
    """

    code = "VG_CONTRACT_ERROR"


# ---------------------------------------------------------------------------
# Audio pipeline
# ---------------------------------------------------------------------------


class AudioError(VGError):
    """Base for audio-related errors."""

    code = "VG_AUDIO_ERROR"


class BadAudioFormatError(AudioError):
    """Unrecognised encoding, sample rate, or channel count."""

    code = "VG_BAD_AUDIO_FORMAT"


class QualityGateError(AudioError):
    """Audio was rejected by the quality gate; head should abstain."""

    code = "VG_QUALITY_GATE"


# ---------------------------------------------------------------------------
# Detection heads
# ---------------------------------------------------------------------------


class HeadError(VGError):
    """Base for detection head errors."""

    code = "VG_HEAD_ERROR"


class HeadTimeoutError(HeadError):
    """Head exceeded its latency budget.  The head MUST catch this and abstain."""

    code = "VG_HEAD_TIMEOUT"

    def __init__(self, head_id: str, budget_ms: int, elapsed_ms: float) -> None:
        super().__init__(
            f"Head {head_id} exceeded budget {budget_ms} ms (elapsed {elapsed_ms:.0f} ms)",
            detail=f"head_id={head_id}, budget_ms={budget_ms}, elapsed_ms={elapsed_ms:.1f}",
        )
        self.head_id = head_id
        self.budget_ms = budget_ms
        self.elapsed_ms = elapsed_ms


class HeadNotFoundError(HeadError):
    """Requested head ID is not registered in the HeadRegistry."""

    code = "VG_HEAD_NOT_FOUND"


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


class EnrollmentError(VGError):
    """Base for enrollment / voiceprint errors."""

    code = "VG_ENROLLMENT_ERROR"


class NoEnrollmentError(EnrollmentError):
    """No voiceprint found for the given speaker_id.  Heads must abstain (I2)."""

    code = "VG_NO_ENROLLMENT"


class InsufficientEnrollmentAudioError(EnrollmentError):
    """Not enough voiced audio was provided for enrollment."""

    code = "VG_INSUFFICIENT_ENROLLMENT_AUDIO"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PolicyError(VGError):
    """Raised by the policy engine for configuration problems."""

    code = "VG_POLICY_ERROR"
