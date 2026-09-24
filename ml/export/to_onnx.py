"""ml.export.to_onnx — export Head A (teacher or distilled student) to ONNX (B14-T03).

The exported graph takes ``wav: float32[B, T]`` (16 kHz, T dynamic) and returns

* ``spoof_logit: float32[B]`` — calibrated logit, ``p_spoof = sigmoid(spoof_logit)``.
  Calibration (scale, bias) is baked into the graph so edge clients need no
  side-car file: ``spoof_logit = -(scale · raw + bias)``, matching
  ``HeadA``'s ``p_spoof = 1 / (1 + exp(scale · raw + bias))``.
* ``raw_score: float32[B]`` — cosine to the bona fide centre (higher = more bona fide).

``onnx`` is an optional dependency (not in the base install); every entry point
fails with an actionable message when it is missing.

Usage:
    python -m ml.export.to_onnx --checkpoint runs/distilled/head_a_student.pt \
        --out deploy/triton/models/head_a/1/model.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from packages.vg_models.heads.head_a_ssl.model import HeadAModel, load_checkpoint

INPUT_NAME = "wav"
OUTPUT_NAMES = ["spoof_logit", "raw_score"]
DEFAULT_OPSET = 17


class OnnxUnavailableError(RuntimeError):
    pass


def require(module: str) -> object:
    """Import an optional export/runtime dependency or fail with the install command."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        pkg = {
            "onnx": "onnx",
            "onnxruntime": "onnxruntime",
            "tritonclient.http": "tritonclient[http]",
        }
        raise OnnxUnavailableError(
            f"{module} is not installed; run: pip install {pkg.get(module, module)} "
            "(see docs/SETUP_PENDING.md, B14)"
        ) from exc


class ExportWrapper(nn.Module):
    """HeadAModel → (spoof_logit, raw_score) with calibration folded in."""

    def __init__(self, model: HeadAModel, cal_scale: float = 8.0, cal_bias: float = 0.0) -> None:
        super().__init__()
        self.model = model.eval()
        self.register_buffer("scale", torch.tensor(float(cal_scale)))
        self.register_buffer("bias", torch.tensor(float(cal_bias)))

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, raw = self.model(wav)
        return -(self.scale * raw + self.bias), raw


def export_onnx(
    model: HeadAModel,
    out: Path | str,
    seconds: float = 3.0,
    opset: int = DEFAULT_OPSET,
    cal_scale: float = 8.0,
    cal_bias: float = 0.0,
) -> Path:
    require("onnx")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model, cal_scale, cal_bias).eval()
    dummy = torch.zeros(1, int(16000 * seconds))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=OUTPUT_NAMES,
            dynamic_axes={
                INPUT_NAME: {0: "batch", 1: "samples"},
                **{o: {0: "batch"} for o in OUTPUT_NAMES},
            },
            opset_version=opset,
            dynamo=False,
        )
    onnx = require("onnx")
    onnx.checker.check_model(onnx.load(str(out)))  # type: ignore[attr-defined]
    meta = {
        "model_version": model.model_version,
        "calibration": {"scale": cal_scale, "bias": cal_bias},
        "input": INPUT_NAME,
        "outputs": OUTPUT_NAMES,
        "opset": opset,
        "sample_rate": 16000,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = ap.parse_args(argv)
    model, meta = load_checkpoint(args.checkpoint)
    cal = meta.get("calibration", {"scale": 8.0, "bias": 0.0})
    try:
        path = export_onnx(
            model, args.out, opset=args.opset, cal_scale=cal["scale"], cal_bias=cal["bias"]
        )
    except OnnxUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"exported {model.model_version} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
