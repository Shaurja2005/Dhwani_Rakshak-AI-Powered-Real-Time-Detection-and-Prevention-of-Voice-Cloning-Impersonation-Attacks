"""Analyst feedback loop (B11-T08).

An analyst marks an alert (by evidence bundle) as a true positive, false
positive, or unsure. Labels flow to:
* B15 evaluation — ``export_eval_rows`` yields per-session rows with the label,
  model versions and the recorded scores (no audio);
* B4 continual learning — ``export_replay_candidates`` lists confirmed cases for
  the replay buffer / new-family procedure. Audio for those sessions exists
  only if a retention policy kept it (B16); rows reference, never embed, it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from services.policy.evidence import EvidenceStore

Label = Literal["true_positive", "false_positive", "unsure"]


class FeedbackStore:
    def __init__(self, evidence: EvidenceStore, path: str | Path = ":memory:") -> None:
        self.evidence = evidence
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS feedback (bundle_id TEXT, session_id TEXT, label TEXT, analyst TEXT, "
            "note TEXT, created_at TEXT)"
        )
        self._db.commit()

    def record(self, bundle_id: str, label: Label, analyst: str, note: str = "") -> dict[str, Any]:
        if label not in ("true_positive", "false_positive", "unsure"):
            raise ValueError(f"invalid label {label!r}")
        bundle = self.evidence.get(bundle_id)
        if bundle is None:
            raise KeyError(bundle_id)
        row = {
            "bundle_id": bundle_id,
            "session_id": bundle["session_id"],
            "label": label,
            "analyst": analyst,
            "note": note[:1000],
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
        with self._lock:
            self._db.execute(
                "INSERT INTO feedback VALUES (:bundle_id,:session_id,:label,:analyst,:note,:created_at)",
                row,
            )
            self._db.commit()
        return row

    def latest(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT bundle_id, session_id, label, analyst, note, created_at FROM feedback"
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for bid, sid, label, analyst, note, at in rows:  # later rows override earlier ones
            out[bid] = {
                "bundle_id": bid,
                "session_id": sid,
                "label": label,
                "analyst": analyst,
                "note": note,
                "created_at": at,
            }
        return out

    def export_eval_rows(self) -> list[dict[str, Any]]:
        rows = []
        for fb in self.latest().values():
            if fb["label"] == "unsure":
                continue
            b = self.evidence.get(fb["bundle_id"]) or {}
            risk = b.get("session_risk", {})
            rows.append(
                {
                    "session_id": fb["session_id"],
                    "bundle_id": fb["bundle_id"],
                    "is_spoof_or_fraud": fb["label"] == "true_positive",
                    "risk_score": risk.get("risk_score"),
                    "state": risk.get("state"),
                    "actions": b.get("policy_decision", {}).get("actions", []),
                    "model_versions": b.get("model_versions", {}),
                    "head_scores": b.get("head_scores", []),
                    "profile": b.get("policy_decision", {}).get("threshold_profile"),
                }
            )
        return rows

    def export_replay_candidates(self, path: str | Path) -> int:
        rows = [r for r in self.export_eval_rows() if r["is_spoof_or_fraud"]]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        return len(rows)
