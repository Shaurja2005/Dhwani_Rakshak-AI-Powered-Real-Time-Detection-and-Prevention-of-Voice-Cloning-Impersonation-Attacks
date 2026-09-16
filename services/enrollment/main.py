"""enrollment service entrypoint.

Env:
    VG_VAULT_PATH            SQLite vault file (default data/vault/voiceprints.db)
    VG_VAULT_KEY_<TENANT>    base64 32-byte AES key per tenant (dev only; use KMS in prod)
    VG_HEAD_D_EMBEDDER       ecapa | wespeaker | titanet | mfcc_stats (default ecapa)
    VG_ENROLL_PORT           default 8010
"""

from __future__ import annotations

import os
from pathlib import Path

from packages.vg_core.logging import get_logger
from packages.vg_models.heads.head_d_speaker.embedders import build_embedder
from packages.vg_models.heads.head_d_speaker.vault import VoiceprintVault
from services.enrollment.api import create_app
from services.enrollment.core import EnrollmentService

log = get_logger(__name__)


def main() -> None:
    import uvicorn

    path = Path(os.getenv("VG_VAULT_PATH", "data/vault/voiceprints.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    name = os.getenv("VG_HEAD_D_EMBEDDER", "ecapa")
    embedder = build_embedder(name)
    if embedder.is_baseline:
        log.warning("enrollment_baseline_embedder", note="classical baseline: dev/demo only")
    app = create_app(EnrollmentService(VoiceprintVault(path), embedder))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VG_ENROLL_PORT", "8010")))  # noqa: S104


if __name__ == "__main__":
    main()
