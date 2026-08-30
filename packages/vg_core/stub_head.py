"""vg_core.stub_head — StubHead: deterministic pseudo-random scorer.

This head produces a deterministic score from a seed so that integration tests
are reproducible.  It does NOT load any model and has zero external dependencies.

**This is what unblocks B9, B11, and B13 on day one.**  Every downstream block
can build and test against the stub stream without waiting for a real model.

The score sequence for a given (head_id, session_id, window_id) is fully
determined by ``STUB_SEED`` so tests are repeatable across machines.

Usage in tests / replay::

    from packages.vg_core.stub_head import StubHead
    from packages.vg_core.head_api import HeadRegistry

    registry = HeadRegistry.get_instance()
    for head_id in "ABCDEF":
        registry.register(StubHead(head_id=head_id))  # type: ignore[arg-type]
"""
from __future__ import annotations

import hashlib
import time
from typing import ClassVar

from packages.vg_core.head_api import BaseDetectionHead
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore, SessionContext

# Default seed for the pseudo-random number generator.
# Override via STUB_SEED environment variable for reproducibility.
import os as _os

_DEFAULT_SEED: int = int(_os.getenv("STUB_SEED", "42"))


def _deterministic_float(head_id: str, session_id: str, window_id: int, seed: int) -> float:
    """Return a float in [0, 1] that is fully determined by the inputs."""
    digest = hashlib.sha256(
        f"{seed}:{head_id}:{session_id}:{window_id}".encode()
    ).digest()
    # Use the first 4 bytes as an unsigned int, normalise to [0, 1]
    raw = int.from_bytes(digest[:4], "big")
    return raw / 0xFFFF_FFFF


class StubHead(BaseDetectionHead):
    """Deterministic pseudo-random head for dev / integration testing.

    Parameters
    ----------
    head_id:          The head ID this stub impersonates ("A".."F").
    seed:             Integer seed for score generation (default: STUB_SEED env / 42).
    abstain_fraction: Fraction [0, 1] of windows this head will abstain on.
                      Determined by the same deterministic function.
    """

    head_id: ClassVar[str] = "stub"
    model_version: ClassVar[str] = "stub-v0.1.0"
    budget_ms: ClassVar[int] = 10

    def __init__(
        self,
        head_id: str = "stub",
        seed: int = _DEFAULT_SEED,
        abstain_fraction: float = 0.05,
    ) -> None:
        # Override class-level head_id for per-instance use
        # (entry-point loading will use class attr; direct instantiation can override)
        self._head_id = head_id
        self._seed = seed
        self._abstain_fraction = abstain_fraction

    @property
    def head_id(self) -> str:  # type: ignore[override]
        return self._head_id

    def warmup(self) -> None:
        """No-op for the stub."""
        pass

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:
        t0 = time.monotonic()

        p = _deterministic_float(self._head_id, window.session_id, window.window_id, self._seed)
        abstain_p = _deterministic_float(
            self._head_id + "_abstain", window.session_id, window.window_id, self._seed
        )

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Respect quality gate — if the conditioner says quality is bad, abstain
        should_abstain = (not window.quality_ok) or (abstain_p < self._abstain_fraction)

        if should_abstain:
            return HeadScore(
                session_id=window.session_id,
                window_id=window.window_id,
                head_id=HeadID(self._head_id),  # type: ignore[arg-type]
                raw_score=0.0,
                p_spoof=None,
                abstain=True,
                abstain_reason=AbstainReason.QUALITY_GATE if not window.quality_ok
                else AbstainReason.INSUFFICIENT_SPEECH,
                confidence=0.0,
                latency_ms=latency_ms,
                model_version=self.model_version,
                calibration_version="stub-cal-v0.1",
                evidence={"stub": True, "seed": self._seed},
            )

        return HeadScore(
            session_id=window.session_id,
            window_id=window.window_id,
            head_id=HeadID(self._head_id),  # type: ignore[arg-type]
            raw_score=float(p * 4 - 2),     # map to roughly logit scale [-2, 2]
            p_spoof=p,
            abstain=False,
            abstain_reason=None,
            confidence=0.5 + abs(p - 0.5),  # higher confidence near 0 or 1
            latency_ms=latency_ms,
            model_version=self.model_version,
            calibration_version="stub-cal-v0.1",
            evidence={"stub": True, "seed": self._seed, "note": "deterministic pseudo-random"},
        )
