"""services/ingest/metadata.py — Metadata envelope extraction.

B1-T08: extract calling number, SIP P-Asserted-Identity, trunk ID,
source ASN, tenant ID, direction, claimed identity from all ingest paths.

Every adapter calls `build_call_metadata(...)` to produce the canonical
CallMetadata that gets published to the bus at session start.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from packages.vg_core.models import CallMetadata, Channel, ConsentBasis

# E.164 pattern: +<country_code><number>, 7-15 digits total
_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def normalise_number(raw: Optional[str]) -> Optional[str]:
    """Normalise a phone number to E.164, returning None if unparseable."""
    if not raw:
        return None
    # Strip whitespace, dashes, parens
    cleaned = re.sub(r"[\s\-\(\)\.]+", "", raw.strip())
    if not cleaned.startswith("+"):
        # Assume India (+91) if 10 digits with no country code
        if re.match(r"^\d{10}$", cleaned):
            cleaned = "+91" + cleaned
        elif re.match(r"^91\d{10}$", cleaned):
            cleaned = "+" + cleaned
        else:
            cleaned = "+" + cleaned
    return cleaned if _E164_RE.match(cleaned) else None


def extract_sip_identity(sip_headers: dict[str, str]) -> Optional[str]:
    """Extract caller identity from SIP P-Asserted-Identity or From header."""
    for header in ("P-Asserted-Identity", "p-asserted-identity", "From", "from"):
        value = sip_headers.get(header)
        if value:
            # Match sip:user@host or tel:+number
            m = re.search(r"(?:sip:|tel:)([+\w\-\.@]+)", value)
            if m:
                return m.group(1)
    return None


def detect_codec(content_type: Optional[str], sample_rate: int) -> str:
    """Infer codec_hint from content-type or sample rate."""
    if content_type:
        ct = content_type.lower()
        if "mulaw" in ct or "pcmu" in ct or "g711u" in ct:
            return "g711u"
        if "alaw" in ct or "pcma" in ct or "g711a" in ct:
            return "g711a"
        if "g729" in ct:
            return "g729"
        if "amr" in ct:
            return "amrnb"
        if "opus" in ct:
            return "opus"
        if "evs" in ct:
            return "evs"
        if "pcm" in ct or "l16" in ct or "wav" in ct:
            return "pcm"
    # Fallback: infer from sample rate
    if sample_rate == 8000:
        return "g711u"  # most common 8kHz codec
    if sample_rate == 16000:
        return "pcm"
    return "unknown"


def build_call_metadata(
    *,
    session_id: str,
    tenant_id: str,
    channel: Channel,
    source_sample_rate: int,
    direction: str = "inbound",
    caller_number: Optional[str] = None,
    callee_number: Optional[str] = None,
    claimed_identity_id: Optional[str] = None,
    trunk_id: Optional[str] = None,
    source_asn: Optional[str] = None,
    sip_headers: Optional[dict[str, str]] = None,
    content_type: Optional[str] = None,
    language_hint: Optional[str] = None,
    consent_basis: ConsentBasis = ConsentBasis.LEGITIMATE_USE,
    shadow_mode: bool = True,
) -> CallMetadata:
    """Construct a validated CallMetadata from raw ingest parameters.

    This is the single canonical factory all adapters must use (DRY + contract).
    """
    hdrs = sip_headers or {}

    # Normalise numbers
    caller_norm = normalise_number(caller_number)
    callee_norm = normalise_number(callee_number)

    # Extract identity from SIP headers if not explicitly provided
    identity = claimed_identity_id or extract_sip_identity(hdrs)

    # Detect codec
    codec = detect_codec(content_type, source_sample_rate)

    return CallMetadata(
        session_id=session_id,
        tenant_id=tenant_id,
        direction=direction,
        started_at=datetime.now(tz=timezone.utc),
        caller_number=caller_norm,
        callee_number=callee_norm,
        claimed_identity_id=identity,
        channel=channel,
        codec_hint=codec,
        source_sample_rate=source_sample_rate,
        trunk_id=trunk_id,
        source_asn=source_asn,
        sip_headers=hdrs,
        language_hint=language_hint,
        consent_basis=consent_basis,
        shadow_mode=shadow_mode,
    )
