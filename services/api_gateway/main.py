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
    print(
        f"VoiceGuard gateway bootstrap API key for tenant '{os.getenv('VG_BOOTSTRAP_TENANT', 'demo')}': {raw}"
    )
    grpc_server = serve(services, keys, os.getenv("VG_GRPC_ADDRESS", "[::]:50051"))
    try:
        uvicorn.run(
            create_app(services, keys),
            host="0.0.0.0",
            port=int(os.getenv("VG_GATEWAY_PORT", "8080")),
        )  # noqa: S104
    finally:
        grpc_server.stop(grace=5)


if __name__ == "__main__":
    main()
