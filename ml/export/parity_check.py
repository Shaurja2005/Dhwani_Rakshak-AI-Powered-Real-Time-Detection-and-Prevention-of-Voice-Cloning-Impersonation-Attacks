"""ml.export.parity_check — ONNX Runtime vs PyTorch numerical parity (B14-T03).

Runs the same waveforms through the PyTorch model and the exported ONNX graph
and compares ``raw_score`` and ``p_spoof``. The export passes when the maximum
absolute difference is within tolerance **and** no decision flips at the
operating threshold. Variable-length inputs are included on purpose: dynamic
time axes are where exports usually break.

Usage:
    python -m ml.export.parity_check --checkpoint ckpt.pt --onnx model.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ml.export.to_onnx import INPUT_NAME, ExportWrapper, OnnxUnavailableError, require
from packages.vg_models.heads.head_a_ssl.model import HeadAModel, load_checkpoint

DEFAULT_ATOL = 1e-3


@dataclass
class ParityReport:
    n: int
    lengths: list[int]
    max_abs_diff_raw: float
    max_abs_diff_p: float
    decision_flips: int
    atol: float
    passed: bool


def probe_inputs(
    n: int = 8, lengths: tuple[int, ...] = (24000, 48000, 64000), seed: int = 0
) -> list[np.ndarray]:
    """Speech-like probe signals (harmonic + noise) at several lengths."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = np.arange(lengths[i % len(lengths)]) / 16000
        f0 = rng.uniform(90, 250)
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * 0.1
        out.append((x + 0.01 * rng.standard_normal(len(t))).astype(np.float32))
    return out


def ort_session(onnx_path: Path | str) -> object:
    ort = require("onnxruntime")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # type: ignore[attr-defined]


def check_parity(
    model: HeadAModel,
    onnx_path: Path | str,
    inputs: list[np.ndarray] | None = None,
    cal_scale: float = 8.0,
    cal_bias: float = 0.0,
    atol: float = DEFAULT_ATOL,
    threshold: float = 0.5,
) -> ParityReport:
    sess = ort_session(onnx_path)
    wrapper = ExportWrapper(model, cal_scale, cal_bias).eval()
    inputs = inputs if inputs is not None else probe_inputs()
    d_raw, d_p, flips = [], [], 0
    for x in inputs:
        with torch.no_grad():
            t_logit, t_raw = (v.numpy() for v in wrapper(torch.from_numpy(x)[None]))
        o_logit, o_raw = sess.run(None, {INPUT_NAME: x[None]})  # type: ignore[attr-defined]
        tp, op = 1 / (1 + np.exp(-t_logit)), 1 / (1 + np.exp(-np.asarray(o_logit)))
        d_raw.append(float(np.max(np.abs(t_raw - o_raw))))
        d_p.append(float(np.max(np.abs(tp - op))))
        flips += int(np.sum((tp >= threshold) != (op >= threshold)))
    rep = ParityReport(
        n=len(inputs),
        lengths=sorted({len(x) for x in inputs}),
        max_abs_diff_raw=max(d_raw),
        max_abs_diff_p=max(d_p),
        decision_flips=flips,
        atol=atol,
        passed=max(d_raw) <= atol and flips == 0,
    )
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = ap.parse_args(argv)
    model, meta = load_checkpoint(args.checkpoint)
    cal = meta.get("calibration", {"scale": 8.0, "bias": 0.0})
    try:
        rep = check_parity(
            model, args.onnx, cal_scale=cal["scale"], cal_bias=cal["bias"], atol=args.atol
        )
    except OnnxUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(asdict(rep), indent=2))
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
