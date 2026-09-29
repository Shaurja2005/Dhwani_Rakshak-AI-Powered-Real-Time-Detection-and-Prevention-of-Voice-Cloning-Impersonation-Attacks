"""Egress guard: data residency and no-audio-leaves-the-premises (B16-T03/T05, I6).

Every outbound call a service makes (webhooks, SMS/e-mail providers, model
downloads during setup) goes through ``EgressGuard.check``:

* the destination host must be on the tenant's ``egress_allowlist`` (or the
  process-wide ``VG_EGRESS_ALLOWLIST``) — default deny;
* offshore processing endpoints are refused unless the tenant opted in to
  central processing (RBI: banking voice data stays in India);
* the payload must not carry audio or embeddings: bytes, base64 blobs, long
  numeric arrays and audio-ish keys (``pcm``, ``audio``, ``wav``, ``embedding``…)
  are rejected wherever they appear.
"""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from services.privacy.policy import TenantPrivacy

AUDIO_KEYS = {
    "pcm",
    "audio",
    "wav",
    "samples",
    "waveform",
    "audio_base64",
    "payload_base64",
    "embedding",
    "embeddings",
    "voiceprint",
    "centroid",
}
MAX_NUMERIC_LIST = 64  # a longer float list is treated as a signal / embedding
MAX_OPAQUE_B64 = 4096  # a longer base64 string is treated as a blob


class EgressDenied(PermissionError):  # noqa: N818 - reads naturally at call sites
    pass


def _host(url: str) -> str:
    u = urlparse(url)
    if u.scheme not in ("https", "wss"):
        raise EgressDenied(f"only https/wss egress is allowed, got {u.scheme or 'none'!r}")
    return (u.hostname or "").lower()


def _allowed(host: str, allow: Iterable[str]) -> bool:
    for a in allow:
        a = a.lower().strip()
        if a and (host == a or (a.startswith("*.") and host.endswith(a[1:]))):
            return True
    return False


def _empty(v: Any) -> bool:  # noqa: ANN401
    if v is None:
        return True
    if isinstance(v, str | bytes | list | tuple | dict):
        return len(v) == 0
    return False


def audio_findings(obj: Any, path: str = "$") -> list[str]:  # noqa: ANN401
    """Paths in ``obj`` that look like raw audio or embeddings."""
    out: list[str] = []
    if isinstance(obj, bytes | bytearray | memoryview):
        out.append(f"{path}: binary payload ({len(obj)} bytes)")
    elif hasattr(obj, "dtype") and hasattr(obj, "shape"):
        out.append(f"{path}: array {getattr(obj, 'shape', '')}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in AUDIO_KEYS and not _empty(v):
                out.append(f"{path}.{k}: audio/biometric field")
            out += audio_findings(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        if len(obj) > MAX_NUMERIC_LIST and all(isinstance(x, int | float) for x in obj[:256]):
            out.append(f"{path}: numeric array of length {len(obj)}")
        else:
            for i, v in enumerate(obj):
                out += audio_findings(v, f"{path}[{i}]")
    elif isinstance(obj, str) and len(obj) > MAX_OPAQUE_B64 and " " not in obj:
        try:
            base64.b64decode(obj, validate=True)
            out.append(f"{path}: base64 blob ({len(obj)} chars)")
        except (binascii.Error, ValueError):
            pass
    return out


class EgressGuard:
    def __init__(self, global_allowlist: Iterable[str] | None = None) -> None:
        env = os.getenv("VG_EGRESS_ALLOWLIST", "")
        self.global_allowlist = list(global_allowlist or []) + [h for h in env.split(",") if h]

    def check(
        self,
        url: str,
        payload: Any,
        tenant: TenantPrivacy | None = None,  # noqa: ANN401
        offshore: bool = False,
    ) -> None:
        host = _host(url)
        allow = self.global_allowlist + (tenant.egress_allowlist if tenant else [])
        if not _allowed(host, allow):
            raise EgressDenied(f"egress to {host} is not on the allowlist (default deny)")
        if offshore and not (tenant and tenant.central_processing):
            raise EgressDenied(
                f"{host} processes data outside {tenant.region if tenant else 'IN'}; "
                "tenant has not opted in to central processing (data residency)"
            )
        findings = audio_findings(payload)
        if findings:
            raise EgressDenied("payload carries audio/biometric data: " + "; ".join(findings[:5]))
