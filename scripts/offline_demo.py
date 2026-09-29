"""Offline demo fallback (B18-T05): no network, no models, no audio needed on the day.

    python scripts/offline_demo.py                         # play every recorded scenario
    python scripts/offline_demo.py --scenario human_reads_fraud_script --speed 2

Plays a pack recorded by ``scripts/demo_scenarios.py --pack docs/demo/pack`` as a
live-updating terminal view: per-second risk gauge, window states, intent labels as
they appear, and the final advisory decision with the agent prompt. Conference Wi-Fi
fails; this does not.

Tiers of fallback on stage (docs/demo/DEMO_SCRIPT.md):
  1. live system (laptop gateway, local models)  →  2. recorded calls through the local
  gateway (`make demo-local`)  →  3. this replay of recorded scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BAND_STYLE = {"LOW": "green", "ELEVATED": "yellow", "HIGH": "bold red", "ABSTAIN": "cyan"}


def _utf8() -> bool:
    return (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8"


def gauge(p: float, width: int = 30, fancy: bool | None = None) -> str:
    full, empty = ("█", "░") if (_utf8() if fancy is None else fancy) else ("#", ".")
    n = int(round(max(0.0, min(1.0, p)) * width))
    return full * n + empty * (width - n)


def frames(pack: dict[str, Any]) -> list[dict[str, Any]]:
    """One frame per window: state + p, plus whatever intent/decision had arrived by then."""
    out, labels, decision = [], [], None
    for e in pack["events"]:
        if e["type"] == "context_signals":
            labels = e["data"]["intent_labels"]
        elif e["type"] == "policy_decision":
            decision = e
        elif e["type"] == "window_score":
            out.append(
                {
                    "t": len(out) + 3,
                    "p": e["data"]["p_spoof"],
                    "state": e["data"]["state"],
                    "labels": list(labels),
                    "decision": decision,
                }
            )
    if out:
        out[-1]["decision"] = decision
    return out


def play(
    pack: dict[str, Any], speed: float = 1.0, sleep: Callable[[float], object] = time.sleep
) -> list[str]:
    from rich.console import Console

    con = Console()
    lines: list[str] = []
    title = pack["title"] + (
        "  [placeholder audio at recording time]" if pack.get("placeholder_audio") else ""
    )
    con.rule(f"[bold]{title}")
    for f in frames(pack):
        style = BAND_STYLE.get(f["state"], "white")
        intent = ", ".join(f["labels"]) or "-"
        line = f"{f['t']:>3}s  {gauge(f['p'])} {f['p']:.2f}  {f['state']:<9} intent: {intent}"
        con.print(line, style=style)
        lines.append(line)
        sleep(1.0 / speed)
    d = pack
    verdict = f"Decision: {d['band']}: {', '.join(d['actions']) or 'no action'}"
    con.print(verdict, style=BAND_STYLE.get(d["band"], "white"))
    if d.get("agent_prompt"):
        con.print(f"Agent sees: {d['agent_prompt']}", style="bold")
    con.print(
        "Advisory only: the system adds friction and asks for verification; it never "
        "decides on its own.",
        style="dim",
    )
    return lines + [verdict]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pack", type=Path, default=ROOT / "docs/demo/pack")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args(argv)
    files = sorted(args.pack.glob("*.json"))
    if args.scenario:
        files = [f for f in files if f.stem == args.scenario]
    if not files:
        print(
            f"no recorded scenarios in {args.pack}; record with scripts/demo_scenarios.py --pack",
            file=sys.stderr,
        )
        return 2
    for f in files:
        play(json.loads(f.read_text(encoding="utf-8")), args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
