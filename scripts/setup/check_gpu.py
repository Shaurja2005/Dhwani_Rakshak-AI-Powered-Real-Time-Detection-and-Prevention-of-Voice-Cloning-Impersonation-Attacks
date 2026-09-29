"""Check the training environment and measure real GPU throughput (base-model setup, step 1).

    python scripts/setup/check_gpu.py            # environment checks only
    python scripts/setup/check_gpu.py --bench    # + XLS-R speed test (downloads XLS-R once, ~1.3 GB)

Prints: Python / PyTorch / CUDA versions, the GPU PyTorch sees, free space on E:,
the Hugging Face cache location, and (with --bench) clips per second for the feature
extraction at several batch sizes, with the estimated time for the base model's cache.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = True
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    try:
        import torch
    except ImportError:
        print("torch       NOT INSTALLED -> see docs/TRAINING_GUIDE.md step 1")
        return 2
    print(f"torch       {torch.__version__}  cuda build: {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu         {p.name}, {p.total_memory / 2**30:.1f} GB, compute {p.major}.{p.minor}")
    else:
        ok = False
        print(
            "gpu         NOT VISIBLE to PyTorch -> you have the CPU build of torch, or no NVIDIA driver"
        )
    try:
        import transformers  # noqa: F401

        print(f"transformers {transformers.__version__} OK")
    except Exception as exc:  # noqa: BLE001 - report any import problem
        ok = False
        print(f"transformers BROKEN: {exc}")
    for mod in ("soundfile", "yaml", "scipy", "numpy", "jsonschema", "pandas"):
        try:
            __import__(mod)
        except ImportError:
            ok = False
            print(f"{mod:11s} MISSING")
    hf = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    print(
        f"HF cache    {hf}"
        + ("   <- on C:, set HF_HOME=E:\\hf_cache" if hf.upper().startswith("C:") else "")
    )
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"disk        {free:.0f} GB free on {ROOT.anchor}  (base model needs ~280 GB)")
    if "--bench" in sys.argv and torch.cuda.is_available():
        from ml.training.runtime import prepare_process, setup_cuda
        from packages.vg_models.heads.head_a_ssl.model import (
            HeadAConfig,
            HeadAModel,
            truncate_to_used_layers,
        )

        prepare_process()
        dev = setup_cuda()
        print("loading XLS-R 300M (first time downloads ~1.3 GB into the HF cache)...")
        fe = truncate_to_used_layers(HeadAModel(HeadAConfig())).frontend.eval().to(dev)
        for bs in (8, 16, 24, 32):
            try:
                x = torch.randn(bs, 64000, device=dev)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    fe(x)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(5):
                        fe(x)
                    torch.cuda.synchronize()
                rate = 5 * bs / (time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"bench       batch {bs:2d}: {rate:6.0f} clips/s (4 s each), peak {mem:.1f} GB "
                    f"-> 66k-clip cache ~{66000 / rate / 60:.0f} min GPU time"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"bench       batch {bs}: out of memory (use a smaller cache.batch)")
                torch.cuda.empty_cache()
                break
    print("RESULT      " + ("ready" if ok else "fix the items above first"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
