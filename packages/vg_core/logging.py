"""vg_core.logging — structured JSON logging with session_id correlation.

Every log record is a JSON object. The ``session_id`` field is automatically
injected by ``bind_session`` and propagated for the lifetime of a request via
contextvars so that all downstream calls within a session share the same ID.

Usage::

    from packages.vg_core.logging import get_logger, bind_session

    log = get_logger(__name__)
    bind_session("01900b1a-0000-7000-8000-000000000001")
    log.info("window_scored", window_id=3, p_spoof=0.87, head_id="A")
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

_session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
_tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)


def bind_session(session_id: str, tenant_id: str | None = None) -> None:
    """Bind session context for the current async task/thread."""
    _session_id_var.set(session_id)
    if tenant_id is not None:
        _tenant_id_var.set(tenant_id)


def clear_session() -> None:
    """Clear session context (call at session teardown)."""
    _session_id_var.set(None)
    _tenant_id_var.set(None)


# ---------------------------------------------------------------------------
# structlog processors
# ---------------------------------------------------------------------------


def _inject_session_context(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Inject session_id + tenant_id from context vars into every log record."""
    sid = _session_id_var.get()
    tid = _tenant_id_var.get()
    if sid:
        event_dict.setdefault("session_id", sid)
    if tid:
        event_dict.setdefault("tenant_id", tid)
    return event_dict


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Call once at application startup to configure structlog + stdlib logging."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_session_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Default configuration for when running as a library (no app startup)
# ---------------------------------------------------------------------------
import os as _os

_default_json = _os.getenv("VG_LOG_JSON", "true").lower() != "false"
_default_level = _os.getenv("VG_LOG_LEVEL", "INFO")
configure_logging(level=_default_level, json_output=_default_json)
