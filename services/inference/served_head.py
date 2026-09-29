"""Head A served through the shared batching inference server (B14-T04).

Same scoring, gating and abstain semantics as ``HeadA`` (B4); only the compute
moves to a process-wide ``BatchingInferenceServer`` so that windows from all
live calls share batches. Overload → the head abstains (``timeout``) at once.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from packages.vg_core.logging import get_logger
from packages.vg_models.heads.head_a_ssl.head import HeadA
from services.inference.backends import TorchBackend, backend_from_env
from services.inference.batching import BatchingInferenceServer

log = get_logger(__name__)


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
_registry_state: dict[str, object] = {"mtime": None, "version": None}


def shared_server() -> BatchingInferenceServer:
    global _shared
    with _lock:
        if _shared is None:
            _shared = BatchingInferenceServer(backend_from_env())
        _maybe_swap(_shared)
        return _shared


def _maybe_swap(server: BatchingInferenceServer) -> None:
    """Blue/green (B17-T07): with ``VG_INFERENCE_BACKEND=registry`` follow the registry's
    active Head A. A promote or rollback takes effect on the next batch, no restart."""
    import os

    if os.getenv("VG_INFERENCE_BACKEND", "").lower() != "registry":
        return
    from packages.vg_models.registry import DEFAULT_PATH, ModelRegistry

    path = Path(os.getenv("VG_MODEL_REGISTRY", str(DEFAULT_PATH)))
    mtime = path.stat().st_mtime if path.exists() else None
    if mtime == _registry_state["mtime"]:
        return
    _registry_state["mtime"] = mtime
    entry = ModelRegistry(path).active("head_a") if mtime is not None else None
    if entry is None or entry.version == _registry_state["version"]:
        return
    from packages.vg_models.heads.head_a_ssl.model import load_checkpoint
    from packages.vg_models.registry import sha256_file

    if sha256_file(entry.path) != entry.sha256:
        log.error("registry_checksum_mismatch", version=entry.version)
        return  # keep serving the current model rather than an unverified one
    model, _ = load_checkpoint(entry.path)
    server.backend = TorchBackend(model, trained=True)
    server.backend.model_version = entry.version
    log.warning("head_a_model_swapped", to=entry.version, previous=_registry_state["version"])
    _registry_state["version"] = entry.version


def shared_head_a() -> BatchedHeadA:
    return BatchedHeadA(shared_server())
