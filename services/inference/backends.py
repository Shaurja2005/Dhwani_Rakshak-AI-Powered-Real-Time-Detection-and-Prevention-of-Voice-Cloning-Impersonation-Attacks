"""Inference back-ends for Head A (B14-T03/T04).

Every back-end maps a batch of equal-length 16 kHz windows ``float32[B, T]`` to
Head A ``raw_score[B]`` (cosine to the bona fide centre; higher = more bona fide).
Calibration stays in the head so that all back-ends are interchangeable.

* ``TorchBackend``  — PyTorch, optionally dynamic-INT8 (``ml.export.quantize``).
* ``OrtBackend``    — ONNX Runtime on an exported graph (``ml.export.to_onnx``).
* ``TritonBackend`` — NVIDIA Triton over HTTP (``deploy/triton/models/head_a``).

``trained`` is False for random-init models; the head then abstains with
reason ``untrained`` (ADR 0006) while still exercising the real compute path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from ml.export.to_onnx import INPUT_NAME, require
from packages.vg_models.heads.head_a_ssl.frontend import SSLFrontend
from packages.vg_models.heads.head_a_ssl.model import HeadAConfig, HeadAModel, load_checkpoint


class InferenceBackend(Protocol):
    name: str
    model_version: str
    trained: bool

    def infer(self, batch: np.ndarray) -> np.ndarray: ...


class TorchBackend:
    name = "torch"

    def __init__(self, model: HeadAModel, trained: bool, int8: bool = False) -> None:
        if int8:
            from ml.export.quantize import quantize_dynamic_int8

            model = quantize_dynamic_int8(model)
            self.name = "torch-int8"
        self.model = model.eval()
        self.trained = trained
        self.model_version = model.model_version + ("-int8" if int8 else "")

    def infer(self, batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            _, raw = self.model(torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)))
        return raw.numpy().astype(np.float32)


class OrtBackend:
    name = "onnxruntime"

    def __init__(
        self, onnx_path: Path | str, trained: bool = True, threads: int | None = None
    ) -> None:
        ort = require("onnxruntime")
        opts = ort.SessionOptions()  # type: ignore[attr-defined]
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(  # type: ignore[attr-defined]
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        meta_path = Path(onnx_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.model_version = str(meta.get("model_version", Path(onnx_path).stem)) + "-onnx"
        self.calibration = meta.get("calibration")
        self.trained = trained

    def infer(self, batch: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(batch, dtype=np.float32)
        _, raw = self.session.run(None, {INPUT_NAME: x})
        return np.asarray(raw, dtype=np.float32).reshape(-1)


class TritonBackend:
    name = "triton"

    def __init__(
        self, url: str, model: str = "head_a", model_version: str = "A@triton", trained: bool = True
    ) -> None:
        http = require("tritonclient.http")
        self._http = http
        self.client = http.InferenceServerClient(url=url)  # type: ignore[attr-defined]
        self.model = model
        self.model_version = model_version
        self.trained = trained

    def infer(self, batch: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(batch, dtype=np.float32)
        inp = self._http.InferInput(INPUT_NAME, list(x.shape), "FP32")  # type: ignore[attr-defined]
        inp.set_data_from_numpy(x)
        out = self._http.InferRequestedOutput("raw_score")  # type: ignore[attr-defined]
        res = self.client.infer(self.model, [inp], outputs=[out])
        return np.asarray(res.as_numpy("raw_score"), dtype=np.float32).reshape(-1)


class Wav2Vec2ShapedFrontend(SSLFrontend):
    """wav2vec2 / XLS-R geometry with random weights — a latency proxy, never a detector.

    Conv feature encoder 512 ch, kernels (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2)
    (320x hop); projection to ``hidden``; pre-LN transformer layers with 16 heads
    and FFN 4 x hidden (XLS-R 300M: hidden 1024, 24 layers).
    """

    def __init__(self, hidden: int = 1024, num_layers: int = 6, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden
        self.num_hidden_states = num_layers + 1
        self.name = f"xlsr-proxy-L{num_layers}"
        convs: list[nn.Module] = []
        ch = 1
        for k, s in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            convs += [nn.Conv1d(ch, 512, k, s, bias=False), nn.GELU()]
            ch = 512
        self.encoder = nn.Sequential(*convs)
        self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden))
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden, 16, 4 * hidden, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        )

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.proj(self.encoder(wav.unsqueeze(1)).transpose(1, 2))
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        return tuple(states)


def xlsr_shaped_proxy(num_layers: int = 6, seed: int = 0) -> HeadAModel:
    """Random-init model shaped like a truncated XLS-R student (1024-d, ``num_layers``).

    Used only to measure latency/throughput before a trained student exists: compute
    cost depends on tensor shapes, not on weight values. It is never ``trained``.
    """
    fe = Wav2Vec2ShapedFrontend(num_layers=num_layers, seed=seed).eval()
    layers = list(range(max(1, num_layers - 3), num_layers + 1))
    cfg = HeadAConfig(frontend="tiny", frontend_layers=layers, version="0.0.0")
    return HeadAModel(cfg, frontend=fe).eval()


def backend_from_env() -> InferenceBackend:
    """``VG_INFERENCE_BACKEND`` = torch | torch-int8 | onnx | triton | proxy | registry."""
    kind = os.getenv("VG_INFERENCE_BACKEND", "torch").lower()
    ckpt = os.getenv("VG_HEAD_A_CHECKPOINT")
    if kind == "onnx":
        trained = os.getenv("VG_HEAD_A_ONNX_TRAINED", "1") == "1"
        return OrtBackend(os.environ["VG_HEAD_A_ONNX"], trained=trained)
    if kind == "triton":
        return TritonBackend(os.getenv("VG_TRITON_URL", "localhost:8000"))
    if kind == "registry":  # B17-T07: served_head swaps in the registry's active model
        torch.manual_seed(0)
        return TorchBackend(HeadAModel(HeadAConfig(frontend="tiny", version="0.0.0")), False)
    if kind == "proxy":
        return TorchBackend(
            xlsr_shaped_proxy(int(os.getenv("VG_PROXY_LAYERS", "6"))), trained=False
        )
    int8 = kind == "torch-int8"
    if ckpt:
        model, _ = load_checkpoint(ckpt)
        return TorchBackend(model, trained=True, int8=int8)
    torch.manual_seed(0)
    return TorchBackend(HeadAModel(HeadAConfig(frontend="tiny", version="0.0.0")), False, int8=int8)
