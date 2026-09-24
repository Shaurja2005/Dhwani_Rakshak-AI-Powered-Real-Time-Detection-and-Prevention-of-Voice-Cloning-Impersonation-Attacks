"""Head A served through the shared batching inference server (B14-T04).

Same scoring, gating and abstain semantics as ``HeadA`` (B4); only the compute
moves to a process-wide ``BatchingInferenceServer`` so that windows from all
live calls share batches. Overload → the head abstains (``timeout``) at once.
"""

from __future__ import annotations

import threading

import numpy as np

from packages.vg_models.heads.head_a_ssl.head import HeadA
from services.inference.backends import backend_from_env
from services.inference.batching import BatchingInferenceServer


class BatchedHeadA(HeadA):
    def __init__(self, server: BatchingInferenceServer, **kwargs: object) -> None:
        calibration = getattr(server.backend, "calibration", None)
        if calibration and "cal_scale" not in kwargs:
            kwargs.update(cal_scale=calibration["scale"], cal_bias=calibration["bias"])
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._server = server

    def warmup(self) -> None:
        self._untrained = not self._server.backend.trained
        self._version = self._server.backend.model_version
        self._model = None  # compute lives in the server; HeadA re-calls this cheap warmup

    def _infer(self, pcm: np.ndarray) -> float:
        return self._server.infer(pcm, timeout_ms=self._budget_ms)


_shared: BatchingInferenceServer | None = None
_lock = threading.Lock()


def shared_server() -> BatchingInferenceServer:
    global _shared
    with _lock:
        if _shared is None:
            _shared = BatchingInferenceServer(backend_from_env())
        return _shared


def shared_head_a() -> BatchedHeadA:
    return BatchedHeadA(shared_server())
