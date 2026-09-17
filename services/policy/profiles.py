"""Tenant policy profiles: thresholds, tiered bands, actions, shadow mode (B11-T01, T03, T07).

Policy is *data*, not code. A profile maps (transaction sensitivity tier, risk
band) → actions. Tiered thresholds (B11-T03) let a wealth desk hold a high-value
wire at a lower risk than it would flag a balance inquiry.

Risk here is the combined ``final_risk`` (B10) on a 0–1 scale.

Shadow mode (B11-T07) is the **default for new tenants**: every decision is
computed and recorded in an evidence bundle, but nothing is dispatched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from packages.vg_core.models import VALID_ACTIONS

Band = Literal["LOW", "ELEVATED", "HIGH", "ABSTAIN"]
Tier = Literal["low", "medium", "high"]


@dataclass
class TierBands:
    elevated: float
    high: float


@dataclass
class TenantProfile:
    name: str
    tenant_id: str
    shadow_mode: bool = True  # B11-T07: new tenants start in shadow mode
    bands: dict[str, TierBands] = field(default_factory=dict)  # tier -> thresholds
    actions: dict[str, dict[str, list[str]]] = field(
        default_factory=dict
    )  # tier -> band -> actions
    # Label-triggered minimum actions, regardless of band (e.g. any OTP request -> banner).
    label_actions: dict[str, list[str]] = field(default_factory=dict)
    notify_channels: list[str] = field(default_factory=lambda: ["websocket"])
    version: str = "v1"

    def __post_init__(self) -> None:
        self.bands = {
            k: v if isinstance(v, TierBands) else TierBands(**v) for k, v in self.bands.items()
        }
        for tier, by_band in self.actions.items():
            for band, acts in by_band.items():
                bad = set(acts) - VALID_ACTIONS
                if bad:
                    raise ValueError(
                        f"profile {self.name}: unknown actions {bad} for {tier}/{band}"
                    )
        for tier, b in self.bands.items():
            if not 0 < b.elevated < b.high <= 1:
                raise ValueError(f"profile {self.name}: tier {tier} needs 0 < elevated < high <= 1")

    def band_for(self, risk: float, tier: Tier) -> Band:
        b = self.bands.get(tier) or self.bands["medium"]
        return "HIGH" if risk >= b.high else "ELEVATED" if risk >= b.elevated else "LOW"

    def snapshot(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TenantProfile:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


_BANNER_CALLBACK = ["agent_banner", "force_callback"]
_HIGH_ACTIONS = [
    "agent_banner",
    "force_callback",
    "hold_transaction",
    "supervisor_escalate",
    "flag_recording",
    "soc_ticket",
]


def bfsi_default(tenant_id: str) -> TenantProfile:
    """SOURCE_OF_TRUTH §9 bands (40 / 70) for medium-sensitivity requests; tiered around them."""
    return TenantProfile(
        name="bfsi-default-v1",
        tenant_id=tenant_id,
        bands={
            "low": {
                "elevated": 0.60,
                "high": 0.95,
            },  # balance inquiry: flag only with strong evidence
            "medium": {"elevated": 0.40, "high": 0.70},
            "high": {"elevated": 0.30, "high": 0.60},  # high-value wire: hold at lower risk
        },
        actions={
            "low": {"ELEVATED": ["agent_banner"], "HIGH": _BANNER_CALLBACK},
            "medium": {
                "ELEVATED": _BANNER_CALLBACK,
                "HIGH": ["agent_banner", "force_callback", "step_up_mfa", "flag_recording"],
            },
            "high": {
                "ELEVATED": ["agent_banner", "force_callback", "hold_transaction"],
                "HIGH": _HIGH_ACTIONS,
            },
        },
        label_actions={"credential_request": ["agent_banner"], "secrecy_demand": ["agent_banner"]},
        notify_channels=["websocket", "siem_webhook"],
    )


def wealth_desk(tenant_id: str) -> TenantProfile:
    """Private-banking desk: few calls, very high value, conservative — dual approval on doubt."""
    return TenantProfile(
        name="bfsi-wealth-v2",
        tenant_id=tenant_id,
        bands={
            "low": {"elevated": 0.40, "high": 0.80},
            "medium": {"elevated": 0.25, "high": 0.55},
            "high": {"elevated": 0.15, "high": 0.40},
        },
        actions={
            "low": {
                "ELEVATED": _BANNER_CALLBACK,
                "HIGH": _BANNER_CALLBACK + ["supervisor_escalate"],
            },
            "medium": {
                "ELEVATED": _BANNER_CALLBACK + ["dual_approval"],
                "HIGH": _HIGH_ACTIONS + ["dual_approval"],
            },
            "high": {
                "ELEVATED": _BANNER_CALLBACK + ["hold_transaction", "dual_approval"],
                "HIGH": _HIGH_ACTIONS + ["dual_approval", "siem_webhook"],
            },
        },
        label_actions={
            "credential_request": ["agent_banner", "supervisor_escalate"],
            "authority_pressure": ["agent_banner"],
        },
        notify_channels=["websocket", "siem_webhook", "send_email"],
    )


def retail_helpline(tenant_id: str) -> TenantProfile:
    """High-volume retail helpline: avoid friction for ordinary customers."""
    return TenantProfile(
        name="retail-helpline-v1",
        tenant_id=tenant_id,
        bands={
            "low": {"elevated": 0.75, "high": 0.97},
            "medium": {"elevated": 0.55, "high": 0.85},
            "high": {"elevated": 0.40, "high": 0.70},
        },
        actions={
            "low": {"ELEVATED": ["agent_banner"], "HIGH": ["agent_banner"]},
            "medium": {"ELEVATED": ["agent_banner"], "HIGH": _BANNER_CALLBACK + ["step_up_mfa"]},
            "high": {
                "ELEVATED": _BANNER_CALLBACK,
                "HIGH": _BANNER_CALLBACK + ["hold_transaction", "flag_recording"],
            },
        },
        label_actions={"credential_request": ["agent_banner"]},
    )


BUILTIN = {
    "bfsi-default-v1": bfsi_default,
    "bfsi-wealth-v2": wealth_desk,
    "retail-helpline-v1": retail_helpline,
}


class ProfileStore:
    """Per-tenant active profile. New tenants get ``bfsi-default-v1`` in shadow mode."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = Path(directory) if directory else None
        self._profiles: dict[str, TenantProfile] = {}

    def get(self, tenant_id: str) -> TenantProfile:
        if tenant_id not in self._profiles:
            path = self._dir / f"{tenant_id}.json" if self._dir else None
            self._profiles[tenant_id] = (
                TenantProfile.load(path) if path and path.exists() else bfsi_default(tenant_id)
            )
        return self._profiles[tenant_id]

    def put(self, profile: TenantProfile) -> None:
        self._profiles[profile.tenant_id] = profile
        if self._dir:
            profile.save(self._dir / f"{profile.tenant_id}.json")

    def set_shadow(self, tenant_id: str, shadow: bool) -> TenantProfile:
        prof = self.get(tenant_id)
        prof.shadow_mode = shadow
        self.put(prof)
        return prof
