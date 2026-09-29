"""B18 — demo scenarios (layered defence), offline fallback, headline chart guards, write-up claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def stub_services():  # noqa: ANN201
    from packages.vg_core.stub_head import StubHead
    from services.api_gateway.pipeline import Services

    return Services(heads_factory=lambda: [StubHead(h, abstain_fraction=0.0) for h in "ABC"])


def scenarios() -> dict[str, dict]:
    cfg = yaml.safe_load((ROOT / "docs/demo/scenarios.yaml").read_text(encoding="utf-8"))
    return {s["id"]: s for s in cfg["scenarios"]}


# ---------------------------------------------------------------- T01/T02 scenarios
def test_scenarios_cover_both_demos_and_consent() -> None:
    sc = scenarios()
    assert {"genuine_customer", "cloned_executive_transfer", "human_reads_fraud_script"} <= set(sc)
    demo2 = sc["human_reads_fraud_script"]["expect"]
    assert (
        demo2["acoustic"] == "LOW" and demo2["band"] != "LOW"
    )  # layered defence: intent carries it
    assert "consent" in (ROOT / "docs/demo/scenarios.yaml").read_text(encoding="utf-8").lower()
    for s in sc.values():
        assert s["audio"].startswith("demo_assets/")  # outside git (I5, I9)


def test_intent_layer_carries_the_human_fraud_script(tmp_path: Path) -> None:
    import scripts.demo_scenarios as ds

    sc = scenarios()
    res = ds.run_scenario(sc["human_reads_fraud_script"], stub_services(), seed=0)
    assert res.placeholder_audio  # no demo_assets in the repo: says so, never pretends
    assert {"credential_request", "manufactured_urgency", "secrecy_demand"} <= set(
        res.intent_labels
    )
    assert res.intent_risk > 0.5 and res.band in ("ELEVATED", "HIGH")
    assert "verify" in res.agent_prompt.lower()
    assert res.checks["intent"] is True
    genuine = ds.run_scenario(sc["genuine_customer"], stub_services(), seed=1)
    assert genuine.intent_labels == [] and genuine.checks["intent"]
    assert ds.main(["--only", "genuine_customer", "--pack", str(tmp_path / "pack")]) in (0, 1)
    pack = json.loads((tmp_path / "pack" / "genuine_customer.json").read_text(encoding="utf-8"))
    assert pack["events"] and "pcm" not in json.dumps(pack)  # scores/events only, no audio


# ---------------------------------------------------------------- T05 offline fallback
def test_offline_player_needs_no_models_or_network(tmp_path: Path) -> None:
    import scripts.demo_scenarios as ds
    import scripts.offline_demo as od

    res = ds.run_scenario(scenarios()["human_reads_fraud_script"], stub_services(), seed=2)
    pack = res.__dict__
    slept: list[float] = []
    lines = od.play(json.loads(json.dumps(pack, default=str)), speed=4, sleep=slept.append)
    assert len(slept) == len(od.frames(pack)) >= 3 and all(s == 0.25 for s in slept)
    assert lines[-1].startswith(f"Decision: {res.band}")
    assert od.gauge(0.5, 10, fancy=False) == "#####....." and len(od.gauge(1.2, 10, True)) == 10
    assert od.main(["--pack", str(tmp_path / "empty")]) == 2


# ---------------------------------------------------------------- T03 headline chart
def test_headline_chart_guards(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    import scripts.make_headline_chart as hc
    from packages.vg_eval import protocols as P
    from packages.vg_eval.report import RunRecord

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        [
            {
                "utt_id": f"u{i}",
                "label": "spoof" if i % 2 else "bona_fide",
                "language": ["hi", "ta", "bn"][i % 3],
                "score": rng.normal(-1 if i % 2 else 1, 1.5),
            }
            for i in range(300)
        ]
    )
    recs = {}
    for name, data in (
        ("before", {"manifest_sha256": "m1"}),
        ("after", {"manifest_sha256": "m1"}),
        ("other", {"manifest_sha256": "m2"}),
        ("syn", {"manifest_sha256": "m1", "synthetic": True}),
    ):
        r = RunRecord(f"A@{name}", "head_a", {}, {"eval_sets": ["x"], **data})
        r.add_rows(P.per_language(df, bootstrap=0))
        recs[name] = r.save(tmp_path / name)
    out = tmp_path / "chart.png"
    assert (
        hc.main(["--before", str(recs["before"]), "--after", str(recs["after"]), "--out", str(out)])
        == 0
    )
    assert out.stat().st_size > 5000
    assert (
        hc.main(["--before", str(recs["before"]), "--after", str(recs["other"]), "--out", str(out)])
        == 2
    )
    assert (
        hc.main(["--before", str(recs["syn"]), "--after", str(recs["after"]), "--out", str(out)])
        == 2
    )


# ---------------------------------------------------------------- T04/T06 write-up
FRAMING = (
    "Detection is advisory, never authoritative",
    "strongest defences are partly non-acoustic",
)


@pytest.mark.parametrize("doc", ["README.md", "docs/demo/WRITEUP.md", "docs/demo/DEMO_SCRIPT.md"])
def test_framing_statements_present(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    for phrase in FRAMING:
        assert phrase.lower() in text.lower(), (doc, phrase)


@pytest.mark.parametrize(
    "doc",
    [
        "README.md",
        "docs/demo/WRITEUP.md",
        "docs/demo/DEMO_SCRIPT.md",
        "docs/demo/VIDEO_SCRIPT.md",
        "docs/ARCHITECTURE.md",
    ],
)
def test_no_untraceable_performance_numbers(doc: str) -> None:
    """AGENTS.md §5: any EER / accuracy percentage in pitch material must cite a REPORT.md row."""
    row_id = re.compile(r"\b(r[0-9a-f]{8}\.[\w.-]+|lt-[\w.-]+)")
    for line in (ROOT / doc).read_text(encoding="utf-8").splitlines():
        if re.search(r"\d+(\.\d+)?\s?%", line):
            assert row_id.search(line), f"{doc}: uncited number in: {line.strip()}"


def test_architecture_and_video_docs_exist() -> None:
    arch = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "```mermaid" in arch and "advisory" in arch.lower()
    video = (ROOT / "docs/demo/VIDEO_SCRIPT.md").read_text(encoding="utf-8")
    assert "3:00" in video and "consent" in video.lower()
