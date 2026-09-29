"""Run the B18 demo scenarios through the real pipeline and show every defence layer.

    python scripts/demo_scenarios.py                          # rehearse: layered table + checks
    python scripts/demo_scenarios.py --pack docs/demo/pack    # record the offline fallback pack
    VG_GATEWAY_HEADS=A,B,C,F VG_HEAD_A_CHECKPOINT=... python scripts/demo_scenarios.py

For each scenario in ``docs/demo/scenarios.yaml``: stream the audio in 1 s chunks,
send the transcript after 3 s, close, then report

    acoustic layer   fused session state + max p_spoof (heads A–F, B9)
    intent layer     intent labels + intent risk (B10)
    decision         band, actions and the agent prompt (B11, advisory)

and compare against the scenario's ``expect`` block. Mismatches are printed, never
hidden (a failed rehearsal is the point of rehearsing). The pack stores scores and
events only — never audio (I5).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000


@dataclass
class Result:
    id: str
    title: str
    placeholder_audio: bool
    acoustic_state: str
    p_spoof_max: float
    intent_labels: list[str]
    intent_risk: float
    band: str
    actions: list[str]
    agent_prompt: str
    model_versions: dict[str, str]
    checks: dict[str, bool] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


def load_audio(path: Path, seed: int) -> tuple[np.ndarray, bool]:
    if path.exists():
        import soundfile as sf
        from scipy.signal import resample_poly

        x, sr = sf.read(path, dtype="float32", always_2d=False)
        x = x.mean(axis=1) if x.ndim > 1 else x
        return (resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x), False
    from scripts.replay_calls import synth_call

    return synth_call(12.0, seed), True


def run_scenario(sc: dict[str, Any], services: Any, seed: int) -> Result:  # noqa: ANN401
    from packages.vg_core.models import CallMetadata
    from services.api_gateway.pipeline import SessionPipeline
    from services.context.asr import Segment

    pcm, placeholder = load_audio(ROOT / sc["audio"], seed)
    meta = CallMetadata(
        session_id=str(uuid.uuid4()),
        tenant_id="demo",
        direction="inbound",
        started_at=dt.datetime.now(dt.UTC),
        channel="pstn",
        codec_hint="pcm",
        source_sample_rate=SR,
        consent_basis="legitimate_use",
        shadow_mode=False,
        caller_number=f"+91-demo-{sc['id']}",
    )
    pipe = SessionPipeline(meta, services)
    events = []
    for k, off in enumerate(range(0, len(pcm), SR)):
        events += pipe.push_pcm16k(pcm[off : off + SR])
        if k == 2 and sc.get("transcript"):
            events += pipe.add_transcript([Segment(**s) for s in sc["transcript"]])
    events += pipe.close()
    ev = [e.to_json() for e in events]
    risk = next((e["data"] for e in reversed(ev) if e["type"] == "session_risk"), {})
    ctx = next((e["data"] for e in reversed(ev) if e["type"] == "context_signals"), {})
    dec = next((e for e in reversed(ev) if e["type"] == "policy_decision"), {})
    res = Result(
        id=sc["id"],
        title=sc["title"],
        placeholder_audio=placeholder,
        acoustic_state=str(risk.get("state", "ABSTAIN")),
        p_spoof_max=float(risk.get("p_spoof_session_max", 0.0)),
        intent_labels=sorted(ctx.get("intent_labels", [])),
        intent_risk=float(ctx.get("intent_risk", 0.0)),
        band=str(dec.get("band", "LOW")),
        actions=list(dec.get("data", {}).get("actions", [])),
        agent_prompt=str(dec.get("agent_prompt", "")),
        model_versions=dict(risk.get("model_versions", {})),
        events=ev,
    )
    exp = sc.get("expect", {})
    res.checks = {
        "acoustic": res.acoustic_state == exp.get("acoustic", res.acoustic_state),
        "intent": set(exp.get("intent_labels", [])) <= set(res.intent_labels)
        and (bool(exp.get("intent_labels")) or not res.intent_labels),
        "band": res.band == exp.get("band", res.band),
    }
    return res


def build_services() -> Any:  # noqa: ANN401
    from services.api_gateway.pipeline import Services

    os.environ.setdefault("VG_GATEWAY_HEADS", "A,B,C,F")
    return Services()


def print_table(results: list[Result]) -> None:
    from rich.console import Console
    from rich.table import Table

    t = Table(title="VoiceGuard demo: layered result per scenario (advisory, never authoritative)")
    for col in ("Scenario", "Acoustic", "Intent", "Decision", "Checks"):
        t.add_column(col)
    for r in results:
        ac = f"{r.acoustic_state} (max p={r.p_spoof_max:.2f})"
        it = f"{', '.join(r.intent_labels) or '-'} (risk {r.intent_risk:.2f})"
        chk = " ".join(f"{k}:{'ok' if v else 'MISMATCH'}" for k, v in r.checks.items())
        title = r.title + ("  [PLACEHOLDER AUDIO]" if r.placeholder_audio else "")
        t.add_row(title, ac, it, f"{r.band}: {', '.join(r.actions) or 'none'}", chk)
    Console().print(t)
    for r in results:
        if r.agent_prompt:
            Console().print(f"[bold]{r.id}[/bold] agent sees: {r.agent_prompt}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=ROOT / "docs/demo/scenarios.yaml")
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    ap.add_argument("--pack", type=Path, default=None, help="write the offline fallback pack here")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    wanted = set(args.only.split(",")) if args.only else None
    services = build_services()
    results = [
        run_scenario(sc, services, i)
        for i, sc in enumerate(cfg["scenarios"])
        if wanted is None or sc["id"] in wanted
    ]
    print_table(results)
    if args.pack:
        args.pack.mkdir(parents=True, exist_ok=True)
        for r in results:
            d = r.__dict__.copy()
            d["recorded_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            (args.pack / f"{r.id}.json").write_text(
                json.dumps(d, indent=1, default=str), encoding="utf-8"
            )
        print(f"offline pack written to {args.pack} (scores and events only, no audio)")
    return 0 if all(all(r.checks.values()) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
