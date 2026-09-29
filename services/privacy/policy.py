"""Consent matrix and per-tenant privacy policy, loaded from configuration (B16-T04/T05).

    matrix = ConsentMatrix.load()
    matrix.require("speaker_enrollment", "explicit_consent", consent_ref="C-123")

Purposes, lawful bases and bundling rules live in ``config/privacy/consent_matrix.yaml``
— changing who may do what is a config review, not a code change. Tenant policy
(region, central-processing opt-in, egress allowlist, retention windows, legal
holds) lives in ``config/privacy/tenants/<tenant_id>.yaml`` with ``default.yaml``
as the fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(
    os.getenv("VG_PRIVACY_CONFIG", str(Path(__file__).resolve().parents[2] / "config" / "privacy"))
)


class ConsentError(PermissionError):
    """Processing refused: no lawful basis for this purpose."""


@dataclass(frozen=True)
class Purpose:
    name: str
    allowed_bases: tuple[str, ...]
    privacy_notice_required: bool = True
    in_call_consent_prompt: bool = False
    bundling_allowed: bool = True
    requires_consent_record: bool = False
    requires_enrolled_consent: bool = False
    requires_tenant_policy: bool = False
    flagged_calls_only: bool = False
    disabled: bool = False
    description: str = ""


class ConsentMatrix:
    def __init__(self, purposes: dict[str, Purpose], version: int = 1) -> None:
        self.purposes = purposes
        self.version = version

    @classmethod
    def load(cls, path: Path | str | None = None) -> ConsentMatrix:
        p = Path(path) if path else CONFIG_DIR / "consent_matrix.yaml"
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        purposes = {}
        for name, d in raw["purposes"].items():
            d = dict(d)
            d["allowed_bases"] = tuple(d.get("allowed_bases", ()))
            purposes[name] = Purpose(name=name, **d)
        return cls(purposes, int(raw.get("version", 1)))

    def check(
        self,
        purpose: str,
        basis: str,
        consent_ref: str | None = None,
        bundled_with: list[str] | None = None,
    ) -> tuple[bool, str]:
        p = self.purposes.get(purpose)
        if p is None:
            return False, f"unknown purpose {purpose!r} (purpose limitation: not in the matrix)"
        if p.disabled:
            return False, f"purpose {purpose!r} is disabled in this product"
        basis = getattr(basis, "value", basis)
        if basis not in p.allowed_bases:
            return False, f"{purpose} needs one of {list(p.allowed_bases)}, got {basis!r}"
        if p.requires_consent_record and not consent_ref:
            return False, f"{purpose} needs a verifiable consent reference"
        if bundled_with and not p.bundling_allowed:
            return False, f"consent for {purpose} cannot be bundled with {bundled_with}"
        return True, "ok"

    def require(self, purpose: str, basis: str, **kw: Any) -> None:  # noqa: ANN401
        ok, why = self.check(purpose, basis, **kw)
        if not ok:
            raise ConsentError(why)


@dataclass
class RetentionPolicy:
    evidence_days: int = 365
    timeline_days: int = 90
    feedback_days: int = 365
    raw_audio_hours: int = 0
    voiceprint_inactive_days: int = 730


@dataclass
class TenantPrivacy:
    tenant_id: str
    region: str = "IN"
    central_processing: bool = False
    egress_allowlist: list[str] = field(default_factory=list)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    legal_holds: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, tenant_id: str, config_dir: Path | str | None = None) -> TenantPrivacy:
        d = Path(config_dir) if config_dir else CONFIG_DIR
        path = d / "tenants" / f"{tenant_id}.yaml"
        if not path.exists():
            path = d / "tenants" / "default.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw["tenant_id"] = tenant_id
        raw["retention"] = RetentionPolicy(**raw.get("retention", {}))
        return cls(**raw)
