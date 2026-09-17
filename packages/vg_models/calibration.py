"""Per-head calibration (B9-T01).

Raw head scores live on different scales (cosine, GBDT log-odds, BiGRU logits),
so they cannot be averaged or fused directly. Each head gets a calibrator fitted
on a held-out, deployment-like set:

* ``platt``       — p = sigmoid(a · raw + b)
* ``temperature`` — p = sigmoid(raw / T)   (one parameter; for already-logit-like scores
  whose only problem is over/under-confidence — it cannot correct a bias or
  class-prior shift, use ``platt`` for that)

Parameters are stored per ``(head_id, model_version)`` (invariant I8): a new
model version never silently reuses an old calibration.

``expected_calibration_error`` is the B9 Definition-of-Done metric (target < 0.05).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from packages.vg_core.models import HeadScore

Method = Literal["platt", "temperature", "identity"]
EPS = 1e-6


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass
class Calibrator:
    head_id: str
    model_version: str
    method: Method = "identity"
    a: float = 1.0
    b: float = 0.0
    temperature: float = 1.0
    version: str = "uncalibrated"
    n_fit: int = 0

    def transform(self, raw: float | np.ndarray) -> float | np.ndarray:
        if self.method == "platt":
            return sigmoid(self.a * np.asarray(raw) + self.b)
        if self.method == "temperature":
            return sigmoid(np.asarray(raw) / self.temperature)
        return sigmoid(np.asarray(raw))

    @classmethod
    def fit(
        cls,
        head_id: str,
        model_version: str,
        raw: np.ndarray,
        is_spoof: np.ndarray,
        method: Method = "platt",
        steps: int = 3000,
        l2: float = 1e-4,
    ) -> Calibrator:
        raw = np.asarray(raw, dtype=np.float64)
        y = np.asarray(is_spoof, dtype=np.float64)
        mu, sd = raw.mean(), raw.std() + 1e-9
        z = (raw - mu) / sd
        a, b, t = 1.0, 0.0, 0.0  # t = log inverse temperature (temperature method)
        lr = 0.5
        for _ in range(steps):
            if method == "platt":
                p = sigmoid(a * z + b)
                g = p - y
                a -= lr * (float(np.mean(g * z)) + l2 * a)
                b -= lr * float(np.mean(g))
            else:
                s = math.exp(t)
                p = sigmoid(s * raw)
                g = p - y
                t -= 0.05 * float(np.mean(g * raw * s))
        stamp = f"cal-{dt.date.today().isoformat()}"
        if method == "platt":
            return cls(
                head_id,
                model_version,
                "platt",
                a=a / sd,
                b=b - a * mu / sd,
                version=stamp,
                n_fit=len(y),
            )
        return cls(
            head_id,
            model_version,
            "temperature",
            temperature=1.0 / math.exp(t),
            version=stamp,
            n_fit=len(y),
        )


class CalibrationStore:
    """JSON-backed registry of calibrators keyed by (head_id, model_version)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: dict[tuple[str, str], Calibrator] = {}
        if self._path and self._path.exists():
            for d in json.loads(self._path.read_text(encoding="utf-8")):
                c = Calibrator(**d)
                self._items[(c.head_id, c.model_version)] = c

    def put(self, c: Calibrator) -> None:
        self._items[(c.head_id, c.model_version)] = c
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps([asdict(x) for x in self._items.values()], indent=2), encoding="utf-8"
            )

    def get(self, head_id: str, model_version: str) -> Calibrator | None:
        return self._items.get((head_id, model_version))

    def calibrated_p(self, score: HeadScore) -> tuple[float | None, str]:
        """Calibrated p_spoof for a head score, and the calibration version used.

        If a calibrator exists for this exact head + model version it recalibrates
        ``raw_score``; otherwise the head's own ``p_spoof`` / ``calibration_version``
        is passed through unchanged.
        """
        if score.abstain or score.p_spoof is None:
            return None, score.calibration_version
        cal = self.get(score.head_id.value, score.model_version)
        if cal is None:
            return float(score.p_spoof), score.calibration_version
        return float(cal.transform(score.raw_score)), cal.version


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        m = idx == k
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
