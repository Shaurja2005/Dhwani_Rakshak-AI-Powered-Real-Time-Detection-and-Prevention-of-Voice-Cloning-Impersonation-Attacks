"""api_gateway service entrypoint (B12).

Starts the REST/WebSocket gateway (uvicorn) and the gRPC server in one process.

Env:
    VG_GATEWAY_PORT       REST/WebSocket port (default 8080)
    VG_GRPC_ADDRESS       gRPC bind address (default [::]:50051)
    VG_BOOTSTRAP_TENANT   tenant for the bootstrap key (default "demo")
    VG_BOOTSTRAP_KEY      optional fixed dev key; otherwise one is generated and printed once
    VG_GATEWAY_HEADS      heads to run (default A,B,C,F)
    VG_DATA_DIR           data directory (default data/)
"""

from __future__ import annotations

import os
from pathlib import Path

from packages.vg_core.logging import get_logger
from services.api_gateway.app import create_app
from services.api_gateway.auth import SCOPES, KeyStore
from services.api_gateway.grpc_server import serve
from services.api_gateway.pipeline import Services
from services.fusion.persistence import SQLiteTimelineStore
from services.policy.evidence import EvidenceStore
from services.policy.profiles import ProfileStore

log = get_logger(__name__)


def bootstrap(keys: KeyStore) -> str:
    tenant = os.getenv("VG_BOOTSTRAP_TENANT", "demo")
    raw, _ = keys.create(tenant, set(SCOPES), env="dev", raw=os.getenv("VG_BOOTSTRAP_KEY") or None)
    return raw


def build_enrollment(data: Path) -> object:
    """Enrollment service for the admin console. Dev fallbacks are loud, never silent."""
    import base64
    import secrets

    from packages.vg_models.heads.head_d_speaker.embedders import build_embedder
    from packages.vg_models.heads.head_d_speaker.vault import VoiceprintVault
    from services.enrollment.core import EnrollmentService

    tenant = os.getenv("VG_BOOTSTRAP_TENANT", "demo")
    var = f"VG_VAULT_KEY_{tenant.upper().replace('-', '_')}"
    if not os.getenv(var):
        os.environ[var] = base64.b64encode(secrets.token_bytes(32)).decode()
        log.warning(
            "vault_ephemeral_key",
            tenant=tenant,
            note=f"{var} not set: voiceprints enrolled now cannot be decrypted after restart (dev only)",
        )
    name = os.getenv("VG_HEAD_D_EMBEDDER", "ecapa")
    try:
        embedder = build_embedder(name)
    except Exception as exc:  # noqa: BLE001 - optional model not installed
        log.warning("enrollment_baseline_embedder", wanted=name, error=str(exc))
        embedder = build_embedder("mfcc_stats")
    (data / "vault").mkdir(parents=True, exist_ok=True)
    return EnrollmentService(VoiceprintVault(data / "vault" / "voiceprints.db"), embedder)


def main() -> None:
    import uvicorn

    data = Path(os.getenv("VG_DATA_DIR", "data"))
    (data / "policy" / "profiles").mkdir(parents=True, exist_ok=True)
    (data / "timeline").mkdir(parents=True, exist_ok=True)
    services = Services(
        profiles=ProfileStore(data / "policy" / "profiles"),
        evidence=EvidenceStore(data / "policy" / "evidence.db"),
        timeline=SQLiteTimelineStore(data / "timeline" / "timeline.db"),
    )
    keys = KeyStore()
    raw = bootstrap(keys)
    enrollment = build_enrollment(data)
    from packages.vg_models.heads.head_e_liveness.challenge import ChallengeRegistry

    challenges = ChallengeRegistry()
    print(
        f"VoiceGuard gateway bootstrap API key for tenant '{os.getenv('VG_BOOTSTRAP_TENANT', 'demo')}': {raw}"
    )
    grpc_server = serve(services, keys, os.getenv("VG_GRPC_ADDRESS", "[::]:50051"))
    try:
        app = create_app(services, keys, enrollment=enrollment, challenges=challenges)
        port = int(os.getenv("VG_GATEWAY_PORT", "8080"))
        print(f"Agent / analyst UI: http://localhost:{port}/ui/")
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        grpc_server.stop(grace=5)


if __name__ == "__main__":
    main()
