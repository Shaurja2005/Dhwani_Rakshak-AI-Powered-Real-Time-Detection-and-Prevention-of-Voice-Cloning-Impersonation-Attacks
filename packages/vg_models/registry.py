"""Model registry with blue/green promotion and instant rollback (B17-T07).

File-backed (``models/released.yaml``) so it works on a single on-prem node
without extra infrastructure; the same schema can move to a database later.

* ``register``  — add a trained artifact as ``staging`` (sha256 recorded).
* ``promote``   — make a staging version *active* for its kind (blue/green): the
  old active becomes ``previous``. Promotion requires an evaluation run id
  (every claim traces to docs/benchmarks/REPORT.md) and refuses a failed
  fairness gate; a commercial deployment refuses research-lineage models (I7).
* ``rollback``  — swap ``active`` and ``previous`` instantly.
* ``resolve``   — path of the active model for a kind, sha256-verified.

Every change is appended to ``history`` (never rewritten).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(os.getenv("VG_MODEL_REGISTRY", "models/released.yaml"))


class RegistryError(RuntimeError):
    pass


@dataclass
class ModelEntry:
    version: str
    kind: str  # head_a | head_b | head_c | head_d | fusion | ...
    path: str
    sha256: str
    lineage: str  # research | commercial
    status: str = "staging"  # staging | active | previous | retired
    eval_run_id: str | None = None
    fairness_gate: str | None = None
    created_at: str = field(default_factory=lambda: _now())
    notes: str = ""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        raw: dict[str, Any] = {}
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.entries = [ModelEntry(**e) for e in raw.get("models", [])]
        self.history: list[dict[str, Any]] = list(raw.get("history", []))

    # ------------------------------------------------------------------ io
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"models": [asdict(e) for e in self.entries], "history": self.history}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written registry

    def _log(self, action: str, **kw: Any) -> None:  # noqa: ANN401
        self.history.append({"at": _now(), "action": action, **kw})

    # ------------------------------------------------------------------ queries
    def get(self, version: str) -> ModelEntry:
        for e in self.entries:
            if e.version == version:
                return e
        raise RegistryError(f"model version {version!r} is not registered")

    def by_status(self, kind: str, status: str) -> ModelEntry | None:
        return next((e for e in self.entries if e.kind == kind and e.status == status), None)

    def active(self, kind: str) -> ModelEntry | None:
        return self.by_status(kind, "active")

    def resolve(self, kind: str, verify: bool = True) -> str | None:
        e = self.active(kind)
        if e is None:
            return None
        if verify and sha256_file(e.path) != e.sha256:
            raise RegistryError(
                f"{e.version}: checksum mismatch for {e.path} (tampered or replaced)"
            )
        return e.path

    # ------------------------------------------------------------------ changes
    def register(
        self,
        version: str,
        kind: str,
        path: Path | str,
        lineage: str,
        eval_run_id: str | None = None,
        fairness_gate: str | None = None,
        notes: str = "",
    ) -> ModelEntry:
        if any(e.version == version for e in self.entries):
            raise RegistryError(f"version {version!r} already registered (versions are immutable)")
        if lineage not in ("research", "commercial"):
            raise RegistryError(f"unknown lineage {lineage!r}")
        e = ModelEntry(
            version,
            kind,
            str(path),
            sha256_file(path),
            lineage,
            eval_run_id=eval_run_id,
            fairness_gate=fairness_gate,
            notes=notes,
        )
        self.entries.append(e)
        self._log("register", version=version, kind=kind, sha256=e.sha256)
        self.save()
        return e

    def attach_eval(self, version: str, run_id: str, fairness_gate: str) -> None:
        e = self.get(version)
        e.eval_run_id, e.fairness_gate = run_id, fairness_gate
        self._log("attach_eval", version=version, run_id=run_id, fairness_gate=fairness_gate)
        self.save()

    def promote(
        self, version: str, commercial_deployment: bool = False, actor: str = "unknown"
    ) -> None:
        e = self.get(version)
        if e.status not in ("staging", "previous"):
            raise RegistryError(f"{version} is {e.status}; only staging/previous can be promoted")
        if not e.eval_run_id:
            raise RegistryError(
                f"{version} has no evaluation run — run `make eval` first (AGENTS §5)"
            )
        if e.fairness_gate == "fail":
            raise RegistryError(f"{version} failed the fairness release gate")
        if commercial_deployment and e.lineage != "commercial":
            raise RegistryError(
                f"{version} is {e.lineage}-lineage; not deployable commercially (I7)"
            )
        if sha256_file(e.path) != e.sha256:
            raise RegistryError(f"{version}: checksum mismatch, refusing to promote")
        old = self.active(e.kind)
        prev = self.by_status(e.kind, "previous")
        if prev is not None and prev is not e:
            prev.status = "retired"
        if old is not None:
            old.status = "previous"
        e.status = "active"
        self._log(
            "promote",
            version=version,
            kind=e.kind,
            replaced=old.version if old else None,
            actor=actor,
        )
        self.save()

    def rollback(self, kind: str, actor: str = "unknown") -> ModelEntry:
        cur, prev = self.active(kind), self.by_status(kind, "previous")
        if prev is None:
            raise RegistryError(f"no previous {kind} model to roll back to")
        prev.status = "active"
        if cur is not None:
            cur.status = "previous"
        self._log(
            "rollback", kind=kind, to=prev.version, from_=cur.version if cur else None, actor=actor
        )
        self.save()
        return prev
