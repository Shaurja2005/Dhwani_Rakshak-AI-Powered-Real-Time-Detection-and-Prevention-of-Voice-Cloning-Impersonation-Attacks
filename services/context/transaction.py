"""Transaction context connector (B10-T05).

``TransactionConnector`` is the interface to a tenant's core-banking / ERP
system; ``MockCoreBanking`` implements it in memory for demos and tests.

Risk signals for the transaction being requested during the call:
* amount far above the customer's historical pattern (robust z-score / multiple of p95)
* new (never-paid) beneficiary, or a beneficiary added very recently
* first-ever high-value transfer
* privileged-access request (limit change, password reset, adding a signatory)

Also returns a *sensitivity tier* (low / medium / high) that B11 uses to pick
tiered thresholds (B11-T03).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

Tier = Literal["low", "medium", "high"]
PRIVILEGED = {
    "limit_increase",
    "password_reset",
    "add_signatory",
    "change_registered_mobile",
    "add_beneficiary",
}


@dataclass
class PendingTransaction:
    kind: str  # "transfer", "balance_inquiry", "limit_increase", ...
    amount: float = 0.0
    currency: str = "INR"
    beneficiary_id: str | None = None


@dataclass
class CustomerProfile:
    customer_id: str
    past_amounts: list[float] = field(default_factory=list)
    beneficiaries: dict[str, dt.datetime] = field(default_factory=dict)  # id -> added at
    high_value_threshold: float = 500_000.0


class TransactionConnector(Protocol):
    def profile(self, customer_id: str) -> CustomerProfile | None: ...

    def pending(self, session_id: str) -> PendingTransaction | None: ...


class MockCoreBanking:
    def __init__(self) -> None:
        self.profiles: dict[str, CustomerProfile] = {}
        self.pending_by_session: dict[str, PendingTransaction] = {}

    def profile(self, customer_id: str) -> CustomerProfile | None:
        return self.profiles.get(customer_id)

    def pending(self, session_id: str) -> PendingTransaction | None:
        return self.pending_by_session.get(session_id)


@dataclass
class TransactionRisk:
    risk: float
    tier: Tier
    signals: dict[str, str]


def score(
    tx: PendingTransaction | None, prof: CustomerProfile | None, now: dt.datetime | None = None
) -> TransactionRisk:
    if tx is None:
        return TransactionRisk(0.0, "low", {})
    now = now or dt.datetime.now(dt.UTC)
    signals: dict[str, str] = {}
    weights: dict[str, float] = {}

    if tx.kind in PRIVILEGED:
        signals["privileged_request"] = (
            f"Caller is requesting a privileged change ({tx.kind.replace('_', ' ')})."
        )
        weights["privileged_request"] = 0.5

    if tx.amount > 0:
        hist = np.array(prof.past_amounts if prof else [], dtype=float)
        if len(hist) >= 5:
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med))) * 1.4826 or max(med * 0.1, 1.0)
            z = (tx.amount - med) / mad
            p95 = float(np.percentile(hist, 95))
            if z > 6 or tx.amount > 3 * p95:
                signals["unusual_amount"] = (
                    f"Amount {tx.amount:,.0f} {tx.currency} is "
                    f"{tx.amount / max(p95, 1):.1f}× the customer's usual maximum."
                )
                weights["unusual_amount"] = min(0.6, 0.2 + 0.05 * min(z, 8))
        else:
            signals["no_history"] = "Little transaction history to compare against."
            weights["no_history"] = 0.15
        if (
            prof
            and tx.amount >= prof.high_value_threshold
            and not (hist >= prof.high_value_threshold).any()
        ):
            signals["first_high_value"] = "First high-value transfer for this customer."
            weights["first_high_value"] = 0.35

    if tx.beneficiary_id:
        added = prof.beneficiaries.get(tx.beneficiary_id) if prof else None
        if added is None:
            signals["new_beneficiary"] = (
                "Money is going to a beneficiary this customer has never paid."
            )
            weights["new_beneficiary"] = 0.45
        elif now - added < dt.timedelta(hours=24):
            signals["recent_beneficiary"] = "Beneficiary was added less than 24 hours ago."
            weights["recent_beneficiary"] = 0.35

    p_clean = 1.0
    for w in weights.values():
        p_clean *= 1 - w
    risk = round(1 - p_clean, 4)
    high_value = prof is not None and tx.amount >= prof.high_value_threshold
    tier: Tier = (
        "high"
        if (high_value or tx.kind in PRIVILEGED or risk >= 0.5)
        else "medium" if tx.amount > 0 else "low"
    )
    return TransactionRisk(risk, tier, signals)
