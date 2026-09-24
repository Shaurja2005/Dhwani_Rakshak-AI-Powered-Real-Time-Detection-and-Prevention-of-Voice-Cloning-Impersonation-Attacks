"""ml.export.quantize — INT8 dynamic quantization with a published accuracy cost (B14-T02).

Dynamic INT8 (weights INT8, activations quantised at runtime) for Linear / GRU /
LSTM layers — the bulk of SSL transformer and back-end compute on CPU. Convolutions
stay FP32 (dynamic quantization does not cover them); static / QAT quantization of
the conv feature encoder is left for when a calibration set exists.

``compare`` measures, on the same inputs: score agreement, decision flips at a
threshold, EER for both models (if labels are given), model size and latency.
``write_report`` renders it to markdown for ``docs/benchmarks/`` — the cost is
published, not hidden.
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ml.training.metrics import compute_eer


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "x86" if "x86" in engines else "fbgemm"
    if any(isinstance(m, nn.TransformerEncoderLayer) for m in model.modules()):
        # nn.TransformerEncoderLayer's fused fast path reads ``linear.weight`` as a tensor,
        # which dynamic-quantized Linear does not expose; disable the fast path (process-wide).
        torch.backends.mha.set_fastpath_enabled(False)
    quantized: nn.Module = torch.ao.quantization.quantize_dynamic(
        model.eval(), {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8
    )
    return quantized


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1e6


@torch.no_grad()
def _scores(model: nn.Module, inputs: torch.Tensor, batch: int = 8) -> tuple[np.ndarray, float]:
    out, t0 = [], time.perf_counter()
    for i in range(0, len(inputs), batch):
        _, s = model(inputs[i : i + batch])
        out.append(s.numpy())
    ms_per_item = (time.perf_counter() - t0) * 1000 / len(inputs)
    return np.concatenate(out), ms_per_item


@dataclass
class QuantReport:
    n: int
    size_fp32_mb: float
    size_int8_mb: float
    ms_per_window_fp32: float
    ms_per_window_int8: float
    mean_abs_score_diff: float
    max_abs_score_diff: float
    decision_flip_rate: float
    eer_fp32: float | None
    eer_int8: float | None


def compare(
    fp32: nn.Module,
    int8: nn.Module,
    inputs: torch.Tensor,
    labels: np.ndarray | None = None,
    threshold: float = 0.0,
) -> QuantReport:
    _scores(fp32, inputs[:2])  # warm up
    _scores(int8, inputs[:2])
    a, ta = _scores(fp32.eval(), inputs)
    b, tb = _scores(int8.eval(), inputs)
    eer_a = eer_b = None
    if labels is not None and len(set(labels.tolist())) == 2:
        eer_a = compute_eer(a[labels == 1], a[labels == 0])[0]
        eer_b = compute_eer(b[labels == 1], b[labels == 0])[0]
    return QuantReport(
        n=len(inputs),
        size_fp32_mb=round(model_size_mb(fp32), 3),
        size_int8_mb=round(model_size_mb(int8), 3),
        ms_per_window_fp32=round(ta, 2),
        ms_per_window_int8=round(tb, 2),
        mean_abs_score_diff=float(np.mean(np.abs(a - b))),
        max_abs_score_diff=float(np.max(np.abs(a - b))),
        decision_flip_rate=float(np.mean((a >= threshold) != (b >= threshold))),
        eer_fp32=eer_a,
        eer_int8=eer_b,
    )


def write_report(rep: QuantReport, model_name: str, hardware: str, data_note: str) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in asdict(rep).items())
    return (
        f"# INT8 quantization report — {model_name}\n\n"
        f"- Hardware: {hardware}\n- Evaluation inputs: {data_note}\n"
        "- Method: PyTorch dynamic INT8 (Linear/GRU/LSTM); convolutions remain FP32.\n\n"
        f"| Metric | Value |\n|---|---|\n{rows}\n\n"
        "Accuracy cost must be re-measured on the held-out deployment set for every "
        "model version (B15).\n"
    )
