"""10-minute quickstart: get a live risk score for your own WAV file (B12-T08).

    # terminal 1
    VG_BOOTSTRAP_KEY=vg_dev_quickstart_key_000000 python -m services.api_gateway.main
    # terminal 2
    python examples/quickstart.py path/to/your.wav

See docs/api/QUICKSTART.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdks.python.voiceguard import VoiceGuardClient  # noqa: E402


def run(wav: str, client: VoiceGuardClient, tenant: str = "demo") -> dict:
    result = client.analyze_file(wav, tenant_id=tenant)
    risk = result["session_risk"] or {}
    print(f"state      : {risk.get('state')}")
    print(f"risk score : {risk.get('risk_score')} / 100")
    print(f"windows    : {len(result['windows'])}")
    for d in risk.get("drivers", []):
        print(f"driver     : {d['factor']} ({d['weight']:.2f}) - {d['detail']}")
    if result.get("agent_prompt"):
        print(f"agent sees : {result['agent_prompt']}")
    decision = result.get("policy_decision") or {}
    print(
        f"evidence   : {decision.get('evidence_bundle_id')} (shadow mode: {decision.get('shadow_mode')})"
    )
    print(
        "Note: detection is advisory. Untrained heads abstain until models are trained (docs/SETUP_PENDING.md)."
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/quickstart.py your.wav")
    url = os.getenv("VG_URL", "http://localhost:8080")
    key = os.getenv("VG_API_KEY", "vg_dev_quickstart_key_000000")
    run(sys.argv[1], VoiceGuardClient(url, key))
