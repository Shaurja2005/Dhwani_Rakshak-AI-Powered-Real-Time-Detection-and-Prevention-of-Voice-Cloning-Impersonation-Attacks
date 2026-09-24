"""ml.export.edge — edge packaging of the distilled Head A (B14-T04, edge tier).

* ``quantize_onnx_int8`` — ONNX Runtime dynamic INT8 (MatMul/Gemm weights).
* ``to_ort_mobile`` — convert to the ORT format for onnxruntime-mobile /
  onnxruntime-android / -ios (reduced-operator builds).

TFLite is not produced here: the PyTorch → TFLite route (ai-edge-torch) needs a
separate toolchain; it is tracked in docs/SETUP_PENDING.md. The SDK edge scorer
(``sdks/python/voiceguard/edge.py``) consumes the ONNX / ORT file directly and
reads ``spoof_logit`` (output 0).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ml.export.to_onnx import require


def quantize_onnx_int8(src: Path | str, dst: Path | str) -> Path:
    require("onnxruntime")
    from onnxruntime.quantization import (  # type: ignore[import-not-found]
        QuantType,
        quantize_dynamic,
    )

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def to_ort_mobile(src: Path | str, out_dir: Path | str) -> Path:
    require("onnxruntime")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - fixed module invocation, no shell
        [
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(src),
            "--output_dir",
            str(out_dir),
            "--optimization_style",
            "Fixed",
        ],
        check=True,
    )
    return out_dir / (Path(src).stem + ".ort")
