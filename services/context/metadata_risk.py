"""Call-metadata risk scorer (B10-T04).

Signals (each with a plain-English explanation), combined by noisy-OR:
* first-time caller (no prior calls from this number to this tenant)
* calling number does not match the numbers registered for the claimed identity
* international origin presenting a domestic (+91) caller ID
* voice trunk / ASN reputation
* off-hours call (tenant business hours, tenant timezone)
* velocity — many calls from the same source in a short period

``CallHistory`` is an in-memory implementation of the lookup interface; the
production tenant backend (CRM / telephony CDRs) implements the same methods.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from packages.vg_core.models import CallMetadata

WEIGHTS = {
    "first_time_caller": 0.15,
    "cli_mismatch": 0.45,
    "international_spoofed_cli": 0.6,
    "bad_trunk_reputation": 0.4,
    "off_hours": 0.15,
    "high_velocity": 0.35,
}


@dataclass
class CallHistory:
    calls: dict[str, list[dt.datetime]] = field(default_factory=lambda: defaultdict(list))
    registered_numbers: dict[str, set[str]] = field(default_factory=dict)  # identity -> numbers
    trunk_reputation: dict[str, float] = field(
        default_factory=dict
    )  # trunk id / ASN -> 0 good .. 1 bad
    international_trunks: set[str] = field(default_factory=set)

    def record(self, number: str, at: dt.datetime) -> None:
        self.calls[number].append(at)


@dataclass
class TenantHours:
    timezone: str = "Asia/Kolkata"
    open_hour: int = 9
    close_hour: int = 19
    working_days: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # Mon-Sat


@dataclass
class MetadataRisk:
    risk: float
    signals: dict[str, str]


def _norm(num: str | None) -> str:
    digits = "".join(c for c in (num or "") if c.isdigit())
    return digits[-10:]


def score(
    meta: CallMetadata, history: CallHistory, hours: TenantHours | None = None
) -> MetadataRisk:
    hours = hours or TenantHours()
    signals: dict[str, str] = {}
    number = meta.caller_number or ""
    key = _norm(number)

    past = [t for t in history.calls.get(key, []) if t < meta.started_at]
    if meta.direction == "inbound" and key and not past:
        signals["first_time_caller"] = "First call from this number."

    if meta.claimed_identity_id and meta.claimed_identity_id in history.registered_numbers:
        registered = {_norm(n) for n in history.registered_numbers[meta.claimed_identity_id]}
        if key and key not in registered:
            signals["cli_mismatch"] = (
                "Calling number is not registered to the person the caller claims to be."
            )

    pai = meta.sip_headers.get("P-Asserted-Identity", "")
    via_international = (meta.trunk_id in history.international_trunks) or (
        meta.source_asn in history.international_trunks
    )
    if number.startswith("+91") and (
        via_international or (pai and "+91" not in pai and "+" in pai)
    ):
        signals["international_spoofed_cli"] = (
            "Indian caller ID arriving over an international route."
        )

    rep = max(
        history.trunk_reputation.get(meta.trunk_id or "", 0.0),
        history.trunk_reputation.get(meta.source_asn or "", 0.0),
    )
    if rep >= 0.5:
        signals["bad_trunk_reputation"] = f"Call route has a poor reputation score ({rep:.1f})."

    local = meta.started_at.astimezone(ZoneInfo(hours.timezone))
    if (
        local.weekday() not in hours.working_days
        or not hours.open_hour <= local.hour < hours.close_hour
    ):
        signals["off_hours"] = f"Call at {local:%a %H:%M} local time, outside business hours."

    recent = [t for t in past if meta.started_at - t <= dt.timedelta(hours=1)]
    if len(recent) >= 3:
        signals["high_velocity"] = f"{len(recent)} calls from this number in the last hour."

    p_clean = 1.0
    for k in signals:
        p_clean *= 1 - WEIGHTS[k]
    return MetadataRisk(round(1 - p_clean, 4), signals)
