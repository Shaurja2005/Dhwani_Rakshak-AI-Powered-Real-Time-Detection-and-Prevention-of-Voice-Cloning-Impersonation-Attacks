"""Run the gateway for local UI development / demos.

    python scripts/dev_gateway.py            # stub heads (pseudo-random scores), dev key below
    VG_GATEWAY_HEADS=A,B,C,F python scripts/dev_gateway.py   # real heads (abstain until trained)

Console: http://localhost:8765/ui/   API key: vg_dev_local_console_key_0000   tenant: demo
Never use the dev key or stub heads outside a developer machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VG_GATEWAY_HEADS", "stub")
os.environ.setdefault("VG_BOOTSTRAP_KEY", "vg_dev_local_console_key_0000")
os.environ.setdefault("VG_GRPC_ADDRESS", "127.0.0.1:50051")
os.environ.setdefault("VG_GATEWAY_PORT", "8765")

from services.api_gateway.main import main  # noqa: E402

if __name__ == "__main__":
    main()
