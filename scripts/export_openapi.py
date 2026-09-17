"""Export the gateway's OpenAPI document (B12-T08).

python scripts/export_openapi.py --out docs/api/openapi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.api_gateway.app import create_app  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/api/openapi.json"))
    args = ap.parse_args(argv)
    spec = create_app().openapi()
    spec["components"] = spec.get("components", {})
    spec["components"]["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
    spec["security"] = [{"bearerAuth": []}]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote {len(spec['paths'])} paths -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
