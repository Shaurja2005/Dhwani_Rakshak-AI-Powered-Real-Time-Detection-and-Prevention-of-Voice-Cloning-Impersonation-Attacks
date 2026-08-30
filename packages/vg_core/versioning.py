"""vg_core.versioning — model_version and calibration_version helpers.

Every HeadScore and SessionRisk carries versioned strings (invariant I8) so
that:
- Two alerts from the same call can be compared even if the model rolled out
  between them.
- A rollback can be traced to an exact checkpoint.
- Forensics always knows what model produced a score.

Version format: ``<head_id>@<frontend>-<backend>-v<semver>``
Example:        ``A@xlsr300m-nes2net-v0.3.1``

Calibration format: ``cal-<YYYY-MM-DD>``
Example:            ``cal-2026-02-11``
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION_RE = re.compile(
    r"^(?P<head>[A-Fa-z]+)@(?P<frontend>[a-z0-9\-]+)-(?P<backend>[a-z0-9\-]+)-v(?P<semver>\d+\.\d+\.\d+)$"
)
_CAL_RE = re.compile(r"^cal-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ModelVersion:
    head: str
    frontend: str
    backend: str
    semver: str

    def __str__(self) -> str:
        return f"{self.head}@{self.frontend}-{self.backend}-v{self.semver}"

    @classmethod
    def parse(cls, version_str: str) -> "ModelVersion":
        m = _VERSION_RE.match(version_str)
        if not m:
            raise ValueError(
                f"Invalid model_version format: {version_str!r}. "
                f"Expected: <head>@<frontend>-<backend>-v<semver>"
            )
        return cls(
            head=m.group("head"),
            frontend=m.group("frontend"),
            backend=m.group("backend"),
            semver=m.group("semver"),
        )

    def is_compatible_with(self, other: "ModelVersion") -> bool:
        """True when major version matches (semver major compat)."""
        return (
            self.head == other.head
            and self.frontend == other.frontend
            and self.semver.split(".")[0] == other.semver.split(".")[0]
        )


def validate_calibration_version(cal_str: str) -> str:
    """Validate a calibration version string; raise ValueError if invalid."""
    if not _CAL_RE.match(cal_str) and cal_str not in ("stub-cal-v0.1", "none"):
        raise ValueError(
            f"Invalid calibration_version format: {cal_str!r}. "
            f"Expected: cal-YYYY-MM-DD"
        )
    return cal_str


# Sentinels for stubs and missing calibration
STUB_MODEL_VERSION = "stub@stub-stub-v0.1.0"
STUB_CAL_VERSION = "stub-cal-v0.1"
UNCALIBRATED = "none"
