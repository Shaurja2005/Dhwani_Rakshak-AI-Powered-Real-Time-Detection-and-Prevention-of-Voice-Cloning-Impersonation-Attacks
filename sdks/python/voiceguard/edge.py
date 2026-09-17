"""Edge / on-device scoring stub (B12-T07).

The privacy story made concrete: score audio on the device and send only the
score — never the audio — to the server. Loads a distilled Head A exported to
ONNX by B14 (``ml/export/to_onnx.py``) with ONNX Runtime (``onnxruntime`` on
desktop, ONNX Runtime Mobile / TFLite on phones).

Until B14 produces a model this abstains with ``untrained`` — it never guesses (I2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EdgeScore:
    p_spoof: float | None
    abstain_reason: str | None
    model_version: str


class EdgeScorer:
    def __init__(
        self, model_path: str | Path | None = None, model_version: str = "A@edge-distilled-unset"
    ) -> None:
        self.model_version = model_version
        self._session = None
        if model_path and Path(model_path).exists():
            import onnxruntime as ort  # lazy, optional dependency

            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

    def score(self, pcm16k: np.ndarray) -> EdgeScore:
        if self._session is None:
            return EdgeScore(None, "untrained", self.model_version)
        if len(pcm16k) < 16000 * 1.5:
            return EdgeScore(None, "insufficient_speech", self.model_version)
        name = self._session.get_inputs()[0].name
        logit = float(
            np.asarray(self._session.run(None, {name: pcm16k.astype(np.float32)[None]})[0]).reshape(
                -1
            )[0]
        )
        return EdgeScore(float(1 / (1 + np.exp(-logit))), None, self.model_version)
