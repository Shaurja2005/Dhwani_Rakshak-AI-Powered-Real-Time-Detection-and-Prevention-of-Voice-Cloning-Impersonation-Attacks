"""vg_core.tracing — TraceContext and distributed tracing helpers.

Every operation that crosses a service boundary carries a TraceContext.
It is propagated via gRPC metadata and HTTP headers.

Fields mirror W3C traceparent where possible so they interop with OpenTelemetry.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Immutable distributed tracing context.

    Attributes
    ----------
    trace_id:   W3C-style 128-bit hex string.
    span_id:    64-bit hex string for the current span.
    session_id: VoiceGuard session UUID (UUIDv7).
    tenant_id:  Owning tenant.
    parent_span_id: Parent span, or None for root spans.
    """

    trace_id: str
    span_id: str
    session_id: str
    tenant_id: str
    parent_span_id: Optional[str] = None

    @classmethod
    def new(cls, session_id: str, tenant_id: str) -> "TraceContext":
        """Create a fresh root span context."""
        return cls(
            trace_id=uuid.uuid4().hex + uuid.uuid4().hex,  # 128-bit
            span_id=uuid.uuid4().hex[:16],                 # 64-bit
            session_id=session_id,
            tenant_id=tenant_id,
        )

    def child(self) -> "TraceContext":
        """Return a child span context inheriting trace_id and session_id."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            parent_span_id=self.span_id,
        )

    def to_headers(self) -> dict[str, str]:
        """Serialise to HTTP/gRPC metadata headers."""
        headers = {
            "x-vg-trace-id": self.trace_id,
            "x-vg-span-id": self.span_id,
            "x-vg-session-id": self.session_id,
            "x-vg-tenant-id": self.tenant_id,
        }
        if self.parent_span_id:
            headers["x-vg-parent-span-id"] = self.parent_span_id
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional["TraceContext"]:
        """Deserialise from HTTP/gRPC metadata headers.  Returns None if incomplete."""
        try:
            return cls(
                trace_id=headers["x-vg-trace-id"],
                span_id=headers["x-vg-span-id"],
                session_id=headers["x-vg-session-id"],
                tenant_id=headers["x-vg-tenant-id"],
                parent_span_id=headers.get("x-vg-parent-span-id"),
            )
        except KeyError:
            return None


# ---------------------------------------------------------------------------
# Context variable for async propagation
# ---------------------------------------------------------------------------

_trace_ctx_var: ContextVar[Optional[TraceContext]] = ContextVar(
    "trace_ctx", default=None
)


def set_trace_context(ctx: TraceContext) -> None:
    _trace_ctx_var.set(ctx)


def get_trace_context() -> Optional[TraceContext]:
    return _trace_ctx_var.get()
