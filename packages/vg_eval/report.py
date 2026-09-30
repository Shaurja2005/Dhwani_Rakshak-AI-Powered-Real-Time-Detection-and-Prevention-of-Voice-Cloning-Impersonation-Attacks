"""vg_eval.report — run records and ``docs/benchmarks/REPORT.md`` (B15-T10).

Each evaluation run is stored as one JSON *run record*
(``docs/benchmarks/runs/<model_version>.json``). ``render`` rebuilds the whole
``REPORT.md`` from every record, so the report is always regenerable and never
hand-edited. Every table row carries a stable anchor id,

    <run_id>.<protocol>.<slice>        e.g. r3f9a1c02.logo.ALL_UNSEEN

which is what any number in a slide, README or commit message must cite
(AGENTS.md §5). Runs on synthetic data are refused for the published report.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from packages.vg_eval.fairness import FairnessResult, GateDecision
from packages.vg_eval.protocols import Row

PROTOCOL_TITLES = {
    "overall": "Overall",
    "logo": "Leave-one-generator-out (per attack family)",
    "cross_dataset": "Cross-dataset",
    "language": "Per language",
    "accent": "Per accent",
    "codec": "Per codec (as recorded in the manifest)",
    "snr": "Per SNR bin",
    "codec_sweep": "Codec sweep (same audio, re-encoded)",
    "snr_sweep": "SNR sweep (same audio, noise added)",
    "adversarial": "Adversarial / laundering robustness",
    "operational": "Operational",
}


@dataclass
class RunRecord:
    model_version: str
    scorer: str
    config: dict[str, Any]
    data: dict[str, Any]  # eval sets, lineage, manifest hashes, synthetic flag
    rows: list[dict[str, Any]] = field(default_factory=list)
    fairness: list[dict[str, Any]] = field(default_factory=list)
    gate: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    notes: list[str] = field(default_factory=list)
    # distinguishes evaluations of one model on different eval-set collections (e.g. the
    # per-language continual steps); not part of run_id, so existing row ids never change
    tag: str = ""

    @property
    def run_id(self) -> str:
        blob = json.dumps(
            {"m": self.model_version, "c": self.config, "d": self.data}, sort_keys=True, default=str
        )
        return "r" + hashlib.sha256(blob.encode()).hexdigest()[:8]

    @property
    def synthetic(self) -> bool:
        return bool(self.data.get("synthetic"))

    def add_rows(self, rows: list[Row]) -> None:
        for r in rows:
            d = asdict(r)
            d["low_n"] = r.low_n
            self.rows.append(d)

    def add_fairness(self, results: list[FairnessResult], gate: GateDecision) -> None:
        for res in results:
            self.fairness.append(
                {
                    "attribute": res.attribute,
                    "threshold": res.threshold,
                    "gap": res.gap,
                    "ratio": res.ratio,
                    "groups": [asdict(g) for g in res.groups],
                }
            )
        self.gate = {"status": gate.status, "detail": gate.detail}

    def save(self, directory: Path | str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        name = _slug(self.model_version) + (f"__{_slug(self.tag)}" if self.tag else "")
        path = d / f"{name}.json"
        payload = asdict(self) | {"run_id": self.run_id}
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> RunRecord:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("run_id", None)
        return cls(**d)


def _json_default(o: object) -> object:
    if isinstance(o, float | int):
        return o
    if hasattr(o, "item"):
        return o.item()  # numpy scalar
    if isinstance(o, tuple | set):
        return list(o)
    return str(o)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "x"


def row_id(run_id: str, protocol: str, slice_: str) -> str:
    return f"{run_id}.{protocol}.{_slug(slice_)}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.2f}%"


def _num(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _extra(e: dict[str, Any]) -> str:
    keep = {k: v for k, v in e.items() if v is not None}
    return ", ".join(f"{k}={v}" for k, v in keep.items())


def render_run(rec: RunRecord) -> str:
    rid = rec.run_id
    out = [
        f'## {rec.model_version}{f" · {rec.tag}" if rec.tag else ""}  <a id="{rid}"></a>',
        "",
        f"- Run id: `{rid}` · scorer: `{rec.scorer}` · generated {rec.created_at}",
        f"- Eval data: {', '.join(rec.data.get('eval_sets', [])) or '—'} "
        f"(lineage: {rec.data.get('lineage', '—')}, manifest sha256: "
        f"`{str(rec.data.get('manifest_sha256', '—'))[:12]}`)",
        f"- Seen attack families in training: {', '.join(rec.data.get('seen_families', [])) or 'unknown'}",
        f"- Training datasets (in-domain): {', '.join(rec.data.get('train_datasets', [])) or 'unknown'}",
    ]
    for n in rec.notes:
        out.append(f"- Note: {n}")
    by_proto: dict[str, list[dict[str, Any]]] = {}
    for r in rec.rows:
        by_proto.setdefault(r["protocol"], []).append(r)
    for proto in [p for p in PROTOCOL_TITLES if p in by_proto] + [
        p for p in by_proto if p not in PROTOCOL_TITLES
    ]:
        rows = by_proto[proto]
        out += ["", f"### {PROTOCOL_TITLES.get(proto, proto)}", ""]
        if proto == "operational":
            out += ["| Row id | Slice | Calls | Details |", "|---|---|---:|---|"]
            for r in rows:
                rid_ = row_id(rid, proto, r["slice"])
                out.append(
                    f"| <a id=\"{rid_}\"></a>`{rid_}` | {r['slice']} | "
                    f"{r['n_bona'] + r['n_spoof']} | {_extra(r['extra'])} |"
                )
            continue
        out += [
            "| Row id | Slice | Bona fide | Spoof | Coverage | EER | EER 95% CI | minDCF | AUC | pAUC@1% | ECE | Notes |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            rid_ = row_id(rid, proto, r["slice"])
            ci = r.get("eer_ci")
            ci_s = f"{100 * ci[0]:.2f}–{100 * ci[1]:.2f}%" if ci else "—"
            notes = _extra(r["extra"])
            if r.get("low_n"):
                notes = ("**low n** " + notes).strip()
            out.append(
                f"| <a id=\"{rid_}\"></a>`{rid_}` | {r['slice']} | {r['n_bona']} | {r['n_spoof']} | "
                f"{100 * r['coverage']:.1f}% | {_pct(r.get('eer'))} | {ci_s} | {_num(r.get('min_dcf'))} | "
                f"{_num(r.get('auc'))} | {_num(r.get('pauc_1pct'))} | {_num(r.get('ece'))} | {notes} |"
            )
    if rec.fairness:
        out += [
            "",
            "### Fairness — bona fide false-positive rate per group (B15-T05)",
            "",
            f"Release gate: **{rec.gate.get('status', '—')}** — {rec.gate.get('detail', '')}",
            "",
            "| Row id | Attribute | Group | Bona fide | FPR | 95% CI | Gap (max−min) |",
            "|---|---|---|---:|---:|---|---:|",
        ]
        for f in rec.fairness:
            for g in f["groups"]:
                rid_ = row_id(rid, f"fairness_{f['attribute']}", g["group"])
                lo, hi = g["ci"]
                out.append(
                    f"| <a id=\"{rid_}\"></a>`{rid_}` | {f['attribute']} | {g['group']} | {g['n_bona']} | "
                    f"{_pct(g['fpr'])} | {_pct(lo)}–{_pct(hi)} | {_pct(f['gap'])} |"
                )
    return "\n".join(out) + "\n"


HEADER = """# Benchmark report

Generated by `make eval` (`ml/eval/run_eval.py`) — **do not edit by hand**; it is
rebuilt from `docs/benchmarks/runs/*.json`. Every performance claim anywhere in
this project must cite a row id from this file (AGENTS.md §5).

Reading guide: EER is computed on scored trials only; *Coverage* is the share of
trials the system did not abstain on. **low n** marks slices with fewer than 30
trials per class. Cross-dataset and unseen-family numbers are expected to be much
worse than in-domain numbers — that gap is the honest measure of field performance.
"""


def render(records: list[RunRecord], allow_synthetic: bool = False, serving: str = "") -> str:
    recs = [r for r in records if allow_synthetic or not r.synthetic]
    body = HEADER
    if not recs:
        body += "\n_(no detection evaluation on real data has been run yet)_\n"
    else:
        recs.sort(key=lambda r: r.created_at, reverse=True)
        index = [
            "",
            "| Model version | Run id | Generated | Eval sets | Fairness gate |",
            "|---|---|---|---|---|",
        ]
        for r in recs:
            index.append(
                f"| {r.model_version}{f' ({r.tag})' if r.tag else ''} | [`{r.run_id}`](#{r.run_id}) | {r.created_at} | "
                f"{', '.join(r.data.get('eval_sets', []))} | {r.gate.get('status', '—')} |"
            )
        banner = []
        if any(r.synthetic for r in recs):
            banner = ["", "> **Contains SYNTHETIC smoke-test runs — not claims about the system.**"]
        body += "\n".join(banner + index) + "\n\n" + "\n".join(render_run(r) for r in recs)
    return body + (("\n" + serving) if serving else "")


# ---------------------------------------------------------------- serving (B14 load tests)
def render_serving(loadtest_dir: Path | str) -> str:
    """Latency / throughput rows from B14 load-test reports (``scripts/loadtest.py``).

    The load-test markdown files are the raw measurement; this section gives each row
    a citable id (``lt-<file>.<calls>``) so latency claims trace here like accuracy claims.
    """
    files = sorted(Path(loadtest_dir).glob("*.md")) if Path(loadtest_dir).exists() else []
    if not files:
        return ""
    out = [
        '## Serving latency and capacity (B14 load tests, B15-T07)  <a id="serving"></a>',
        "",
        "Measured by `scripts/loadtest.py`; hardware, model and caveats are in each source file and "
        "in `docs/benchmarks/LOADTEST.md`. Proxy-weight runs measure latency only, never accuracy.",
        "",
        "| Row id | Source | Model | Concurrent calls | p50 ms | p95 ms | p99 ms | Degraded windows | A timeouts | Keeps up |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for f in files:
        text = f.read_text(encoding="utf-8")
        model = text.splitlines()[0].removeprefix("# Load test — ").strip()
        stem = _slug(f.stem)
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 10 or not cells[0].isdigit():
                continue
            calls, p50, p95, p99 = cells[0], cells[2], cells[3], cells[4]
            degraded, a_to, keeps = cells[7], cells[8], cells[9]
            rid = f"lt-{stem}.{calls}"
            out.append(
                f'| <a id="{rid}"></a>`{rid}` | [{f.name}](loadtest/{f.name}) | {model} | {calls} | '
                f"{p50} | {p95} | {p99} | {degraded} | {a_to} | {keeps} |"
            )
    return "\n".join(out) + "\n"


def rebuild(
    runs_dir: Path | str,
    out: Path | str,
    allow_synthetic: bool = False,
    loadtest_dir: Path | str | None = "docs/benchmarks/loadtest",
) -> Path:
    recs = [RunRecord.load(p) for p in sorted(Path(runs_dir).glob("*.json"))]
    serving = render_serving(loadtest_dir) if loadtest_dir else ""
    Path(out).write_text(render(recs, allow_synthetic, serving), encoding="utf-8")
    return Path(out)


def plot_before_after(
    before: RunRecord, after: RunRecord, out_png: Path | str, protocol: str = "language"
) -> Path:
    """The B18 headline chart: per-language EER before vs after Indic + channel fine-tuning."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def eers(rec: RunRecord) -> dict[str, float]:
        return {
            r["slice"]: r["eer"]
            for r in rec.rows
            if r["protocol"] == protocol and r["eer"] is not None
        }

    b, a = eers(before), eers(after)
    langs = sorted(set(b) | set(a))
    x = range(len(langs))
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(langs) + 2), 4))
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        [100 * b.get(lg, float("nan")) for lg in langs],
        w,
        label=f"before ({before.model_version})",
        color="#9aa5b1",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [100 * a.get(lg, float("nan")) for lg in langs],
        w,
        label=f"after ({after.model_version})",
        color="#1f6feb",
    )
    ax.set_xticks(list(x), langs)
    ax.set_ylabel("EER (%) — lower is better")
    ax.set_title(f"Per-language EER · runs {before.run_id} → {after.run_id}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return Path(out_png)
