"""ml.data.license_gate — hard-fail data loading on license violations (B3-T09, invariant I7).

Training configs declare ``allow_noncommercial``. The ``commercial`` lineage sets
it to false and the loader refuses any row with ``commercial_use: false``.
Independently, eval-only corpora (In-the-Wild) may never enter a training set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ml.data.manifest import ManifestRow
from ml.data.registry import Registry, load_registry

Lineage = Literal["research", "commercial"]
Purpose = Literal["train", "eval"]


class LicenseViolation(RuntimeError):  # noqa: N818
    pass


@dataclass(frozen=True)
class DataPolicy:
    lineage: Lineage
    allow_noncommercial: bool

    def __post_init__(self) -> None:
        if self.lineage == "commercial" and self.allow_noncommercial:
            raise LicenseViolation("commercial lineage cannot set allow_noncommercial: true")

    @classmethod
    def from_config(cls, cfg: dict[str, object]) -> DataPolicy:
        if "allow_noncommercial" not in cfg or "lineage" not in cfg:
            raise LicenseViolation(
                "training config must declare `lineage` and `allow_noncommercial`"
            )
        allow = cfg["allow_noncommercial"]
        if not isinstance(allow, bool):
            raise LicenseViolation("allow_noncommercial must be a boolean")
        lineage = cfg["lineage"]
        if lineage not in ("research", "commercial"):
            raise LicenseViolation(f"unknown lineage {lineage!r}")
        return cls(lineage=lineage, allow_noncommercial=allow)  # type: ignore[arg-type]


def enforce(
    rows: Iterable[ManifestRow],
    policy: DataPolicy,
    purpose: Purpose = "train",
    registry: Registry | None = None,
) -> list[ManifestRow]:
    """Return rows unchanged, or raise LicenseViolation listing every offending row."""
    reg = registry or load_registry()
    registry_flags = {d.name: d for d in reg.datasets}
    rows = list(rows)
    problems: list[str] = []
    for r in rows:
        entry = registry_flags.get(r.source_corpus)
        if entry is None:
            problems.append(f"{r.utt_id}: source_corpus {r.source_corpus!r} not in registry")
            continue
        if purpose == "train" and entry.eval_only:
            problems.append(f"{r.utt_id}: {r.source_corpus} is eval-only and cannot be trained on")
        if not policy.allow_noncommercial:
            # A row is commercial only if both the row and its source corpus say so.
            if not (r.commercial_use and entry.commercial_use):
                problems.append(
                    f"{r.utt_id}: non-commercial data ({r.license}) in {policy.lineage} lineage"
                )
    if problems:
        head = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise LicenseViolation(f"license gate failed for {len(problems)} rows:\n  {head}{more}")
    return rows
