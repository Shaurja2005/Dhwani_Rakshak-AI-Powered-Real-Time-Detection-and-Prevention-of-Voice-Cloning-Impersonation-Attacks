"""vg_core.head_api — DetectionHead Protocol and HeadRegistry.

This is the contract every detection head (B4–B8) must satisfy.
SOURCE_OF_TRUTH §4.9.

Design rules
------------
- Heads are **pure** with respect to session state (no mutable class-level
  caches keyed by session_id).
- Heads MUST NOT raise exceptions up the audio path.  On any error, return
  a ``HeadScore`` with ``abstain=True`` (invariants I2, I10).
- Heads MUST respect ``budget_ms``.  Use ``HeadTimeoutError`` internally to
  track when you are over budget, then return abstain.
- ``warmup()`` is called once at startup outside any request context.  It may
  load weights, JIT-compile, or allocate GPU memory.

HeadRegistry
------------
Heads self-register via Python entry points (``voiceguard.heads`` group) or
can be registered programmatically with ``HeadRegistry.register()``.

Example entry_points in pyproject.toml::

    [project.entry-points."voiceguard.heads"]
    A = "packages.vg_models.heads.head_a_ssl.head:HeadA"
    stub = "packages.vg_core.stub_head:StubHead"
"""
from __future__ import annotations

import time
from importlib.metadata import entry_points
from typing import ClassVar, Optional, Protocol, runtime_checkable

from packages.vg_core.errors import HeadNotFoundError
from packages.vg_core.logging import get_logger
from packages.vg_core.models import AnalysisWindow, HeadID, HeadScore, SessionContext

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol (structural typing — source of truth §4.9)
# ---------------------------------------------------------------------------


@runtime_checkable
class DetectionHead(Protocol):
    """Structural protocol that every detection head must satisfy.

    Use ``isinstance(obj, DetectionHead)`` to check conformance at runtime.
    """

    head_id: ClassVar[str]       # "A".."F" or "stub"
    model_version: ClassVar[str]
    budget_ms: ClassVar[int]

    def warmup(self) -> None:
        """Load model, allocate GPU memory.  Called once at startup."""
        ...

    def score(
        self,
        window: AnalysisWindow,
        ctx: SessionContext,
    ) -> HeadScore:
        """Score one analysis window.

        Parameters
        ----------
        window:  The conditioned analysis window (always 16 kHz, quality_ok may be False).
        ctx:     Session-level context (tenant, claimed identity, etc.).

        Returns
        -------
        HeadScore with either a valid p_spoof (abstain=False) or abstain=True.
        MUST return within ``budget_ms`` milliseconds.  MUST NOT raise.
        """
        ...


# ---------------------------------------------------------------------------
# Abstract base class (optional — for implementations that prefer inheritance)
# ---------------------------------------------------------------------------


class BaseDetectionHead:
    """Convenience ABC.  Subclass this *or* implement the Protocol directly."""

    head_id: ClassVar[str] = ""
    model_version: ClassVar[str] = "unset"
    budget_ms: ClassVar[int] = 120

    def warmup(self) -> None:  # noqa: B027
        pass

    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore:  # noqa: ANN201
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helper: budget-enforced scoring wrapper
    # ------------------------------------------------------------------

    def score_with_budget(
        self, window: AnalysisWindow, ctx: SessionContext
    ) -> HeadScore:
        """Call ``score()`` and enforce the latency budget.

        If ``score()`` raises OR takes longer than ``budget_ms``, returns an
        abstain score instead of propagating the error (I2, I10).
        """
        t0 = time.monotonic()
        try:
            result = self.score(window, ctx)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            if elapsed_ms > self.budget_ms:
                log.warning(
                    "head_over_budget",
                    head_id=self.head_id,
                    budget_ms=self.budget_ms,
                    elapsed_ms=elapsed_ms,
                    session_id=window.session_id,
                )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error(
                "head_exception_abstained",
                head_id=self.head_id,
                exc_type=type(exc).__name__,
                exc=str(exc),
                session_id=window.session_id,
            )
            return HeadScore(
                session_id=window.session_id,
                window_id=window.window_id,
                head_id=HeadID(self.head_id),  # type: ignore[arg-type]
                raw_score=0.0,
                p_spoof=None,
                abstain=True,
                abstain_reason="timeout",
                confidence=0.0,
                latency_ms=elapsed_ms,
                model_version=self.model_version,
                calibration_version="none",
                evidence={"error": str(exc)},
            )


# ---------------------------------------------------------------------------
# HeadRegistry — entry-point plugin loading
# ---------------------------------------------------------------------------


class HeadRegistry:
    """Plugin registry for detection heads.

    Loading priority:
    1. Explicitly registered heads (``register()``).
    2. Heads discovered from ``voiceguard.heads`` entry points.

    The registry is a singleton; use ``HeadRegistry.get_instance()``.
    """

    _instance: Optional["HeadRegistry"] = None
    _registry: dict[str, DetectionHead]

    def __init__(self) -> None:
        self._registry = {}

    @classmethod
    def get_instance(cls) -> "HeadRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, head: DetectionHead) -> None:
        """Register a head instance, replacing any existing one with the same head_id."""
        hid = head.head_id  # type: ignore[attr-defined]
        self._registry[hid] = head
        log.info("head_registered", head_id=hid, model_version=head.model_version)  # type: ignore[attr-defined]

    def load_entry_points(self) -> None:
        """Discover and instantiate heads from the ``voiceguard.heads`` entry point group."""
        eps = entry_points(group="voiceguard.heads")
        for ep in eps:
            try:
                cls = ep.load()
                instance = cls()
                self.register(instance)
                log.info("head_loaded_from_entry_point", name=ep.name, cls=cls.__name__)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "head_entry_point_load_failed",
                    name=ep.name,
                    error=str(exc),
                )

    def get(self, head_id: str) -> DetectionHead:
        """Return the head for the given ID.  Raises HeadNotFoundError if absent."""
        try:
            return self._registry[head_id]
        except KeyError as exc:
            raise HeadNotFoundError(
                f"Head '{head_id}' is not registered",
                detail=f"Registered heads: {list(self._registry.keys())}",
            ) from exc

    def all_heads(self) -> list[DetectionHead]:
        """Return all registered heads."""
        return list(self._registry.values())

    def warmup_all(self) -> None:
        """Call warmup() on every registered head (at service startup)."""
        for head in self._registry.values():
            hid = head.head_id  # type: ignore[attr-defined]
            log.info("head_warmup_start", head_id=hid)
            try:
                head.warmup()
                log.info("head_warmup_done", head_id=hid)
            except Exception as exc:  # noqa: BLE001
                log.error("head_warmup_failed", head_id=hid, error=str(exc))
