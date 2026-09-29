"""Score-distribution drift monitoring (B17-T06).

A new generator family in the wild shows up first as a *shift in the scores of
calls we believe are genuine*. Labels arrive late (analyst feedback), so the
monitor tracks a bona fide **proxy**: window scores from calls whose session
never left LOW, plus calls analysts later confirmed genuine.

``DriftMonitor`` keeps a reference histogram (frozen from a known-good period
or the evaluation set) and a rolling current histogram per head, and reports

* PSI (population stability index) and
* the two-sample KS statistic,

exported as ``vg_score_drift_psi{head}`` for alerting
(``deploy/observability/alerts.yml``). Alert levels are operational knobs in
config (defaults follow the common PSI reading: < 0.1 stable, 0.1–0.25 watch,
> 0.25 shifted) and must be tuned in shadow mode — PROJECT_STATUS Q11.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EDGES = np.linspace(0.0, 1.0, 21)


def psi(ref: np.ndarray, cur: np.ndarray, eps: float = 1e-4) -> float:
    """PSI between two histograms (counts or probabilities over the same bins)."""
    r = np.asarray(ref, dtype=float)
    c = np.asarray(cur, dtype=float)
    r = r / max(r.sum(), 1e-12) + eps
    c = c / max(c.sum(), 1e-12) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def ks(ref_samples: np.ndarray, cur_samples: np.ndarray) -> float:
    a, b = np.sort(ref_samples), np.sort(cur_samples)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


@dataclass
class DriftStatus:
    head: str
    n_current: int
    psi: float | None
    ks: float | None
    level: str  # insufficient_data | stable | watch | shifted


class DriftMonitor:
    def __init__(
        self,
        window: int = 5000,
        min_samples: int = 500,
        watch_psi: float = 0.1,
        alert_psi: float = 0.25,
    ) -> None:
        self.window = window
        self.min_samples = min_samples
        self.watch_psi, self.alert_psi = watch_psi, alert_psi
        self._ref: dict[str, np.ndarray] = {}
        self._ref_samples: dict[str, np.ndarray] = {}
        self._cur: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ reference
    def set_reference(self, head: str, samples: np.ndarray) -> None:
        s = np.clip(np.asarray(samples, dtype=float), 0, 1)
        with self._lock:
            self._ref[head] = np.histogram(s, EDGES)[0].astype(float)
            self._ref_samples[head] = s[-self.window :]

    def freeze_current_as_reference(self, head: str) -> None:
        with self._lock:
            cur = np.array(self._cur.get(head, []), dtype=float)
        if len(cur) < self.min_samples:
            raise ValueError(
                f"need {self.min_samples} samples to freeze a reference, have {len(cur)}"
            )
        self.set_reference(head, cur)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps({h: s.tolist() for h, s in self._ref_samples.items()}), encoding="utf-8"
        )

    def load(self, path: Path | str) -> None:
        for h, s in json.loads(Path(path).read_text(encoding="utf-8")).items():
            self.set_reference(h, np.array(s))

    # ------------------------------------------------------------------ observation
    def observe_session(self, head_p_spoof: dict[str, list[float]], session_state: str) -> None:
        """Feed a finished session; only sessions that stayed LOW count as the bona fide proxy."""
        if session_state != "LOW":
            return
        self.observe_confirmed_genuine(head_p_spoof)

    def observe_confirmed_genuine(self, head_p_spoof: dict[str, list[float]]) -> None:
        with self._lock:
            for head, ps in head_p_spoof.items():
                q = self._cur.setdefault(head, deque(maxlen=self.window))
                q.extend(float(p) for p in ps if p is not None and np.isfinite(p))

    # ------------------------------------------------------------------ status
    def status(self, head: str) -> DriftStatus:
        with self._lock:
            cur = np.array(self._cur.get(head, []), dtype=float)
            ref = self._ref.get(head)
            ref_s = self._ref_samples.get(head)
        if ref is None or len(cur) < self.min_samples:
            return DriftStatus(head, len(cur), None, None, "insufficient_data")
        p = psi(ref, np.histogram(cur, EDGES)[0])
        level = "shifted" if p > self.alert_psi else "watch" if p > self.watch_psi else "stable"
        return DriftStatus(head, len(cur), p, ks(ref_s, cur) if ref_s is not None else None, level)

    def heads(self) -> list[str]:
        return sorted(set(self._ref) | set(self._cur))
