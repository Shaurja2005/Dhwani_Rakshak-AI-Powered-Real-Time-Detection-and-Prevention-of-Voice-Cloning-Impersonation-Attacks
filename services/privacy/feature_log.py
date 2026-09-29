"""Feature-only logging (B16-T02).

``redact_audio`` is a structlog processor that strips anything waveform- or
biometric-shaped from every log record before it is rendered: binary values,
numpy arrays, long numeric lists, base64 blobs, and audio/embedding keys. What
remains is telemetry — ``p_spoof``, codec, intent labels, latencies, versions.

``install()`` adds it to the running structlog configuration; the gateway and
every service call it at start-up (``packages/vg_core/logging.configure_logging``
stays unchanged for other blocks).
"""

from __future__ import annotations

from typing import Any

import structlog

from services.privacy.egress import audio_findings

REDACTED = "[redacted:audio]"


def _scrub(value: Any, key: str = "") -> Any:  # noqa: ANN401
    if audio_findings({key: value} if key else value):
        if isinstance(value, dict):
            return {k: _scrub(v, k) for k, v in value.items()}
        return REDACTED
    return value


def redact_audio(
    logger: Any, method: str, event_dict: structlog.types.EventDict  # noqa: ANN401
) -> structlog.types.EventDict:
    for k in list(event_dict):
        if k == "event":
            continue
        event_dict[k] = _scrub(event_dict[k], k)
    return event_dict


def install() -> None:
    cfg = structlog.get_config()
    procs = list(cfg.get("processors", []))
    if redact_audio in procs:
        return
    # run before the renderer / formatter wrapper (always the last processor)
    procs.insert(max(0, len(procs) - 1), redact_audio)
    structlog.configure(processors=procs)
