"""ml.data.registry — load and validate the dataset registry (B3-T01)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

Kind = Literal["spoof_corpus", "bona_fide", "augmentation"]
Role = Literal[
    "train",
    "train_research",
    "eval_only",
    "comparability",
    "augmentation",
    "bona_fide_source",
]
Access = Literal["public", "eula", "gated_hf", "request_form"]


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    sha256: str | None = None


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Kind
    role: Role
    license: str
    # StrictBool: "yes"/"no"/"unknown" must not silently coerce (invariant I7).
    commercial_use: StrictBool
    license_checked_by: str | None = None
    access: Access
    url: str
    size_gb: float = 0.0
    languages: list[str] = Field(default_factory=list)
    files: list[RegistryFile] = Field(default_factory=list)
    notes: str | None = None

    @property
    def eval_only(self) -> bool:
        return self.role == "eval_only"


class Registry(BaseModel):
    datasets: list[DatasetEntry]

    def get(self, name: str) -> DatasetEntry:
        for d in self.datasets:
            if d.name == name:
                return d
        raise KeyError(f"dataset {name!r} is not in the registry")

    def names(self) -> list[str]:
        return [d.name for d in self.datasets]


def load_registry(path: Path | str = REGISTRY_PATH) -> Registry:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for i, row in enumerate(raw.get("datasets", [])):
        if "commercial_use" not in row:
            raise ValueError(f"registry row {i} ({row.get('name')}) is missing commercial_use (I7)")
    reg = Registry.model_validate(raw)
    names = reg.names()
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate dataset names in registry: {sorted(dupes)}")
    return reg
