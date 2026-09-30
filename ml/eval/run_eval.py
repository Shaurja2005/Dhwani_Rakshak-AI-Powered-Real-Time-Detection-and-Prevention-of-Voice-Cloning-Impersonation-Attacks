"""Evaluation harness entrypoint (B15-T01): one command, every protocol, one report.

    make eval MODEL=head_a:runs/head_a_stage2/best.pt
    python ml/eval/run_eval.py --model pipeline:A,B,C --config ml/eval/configs/default.yaml
    python ml/eval/run_eval.py --model baseline:flatness --synthetic      # smoke test, no data

Steps: load eval manifests → license gate (purpose=eval) → score every utterance
→ overall, leave-one-generator-out, cross-dataset, per language/accent/codec/SNR,
fairness gate, codec + SNR sweeps, laundering + white-box attacks, operational
metrics → run record (``docs/benchmarks/runs/<model>.json``) → rebuild
``docs/benchmarks/REPORT.md``.

Synthetic runs (``--synthetic``) exercise the whole harness without data and are
written to ``runs/eval/smoke/`` — they are refused for the published report.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.eval import adversarial as adv  # noqa: E402
from ml.eval.scorers import HeadAScorer, ScoreOut, Scorer, build_scorer  # noqa: E402
from packages.vg_eval import fairness as F  # noqa: E402
from packages.vg_eval import metrics as M  # noqa: E402
from packages.vg_eval import protocols as P  # noqa: E402
from packages.vg_eval.report import RunRecord, rebuild  # noqa: E402

REPORT = Path("docs/benchmarks/REPORT.md")
RUNS = Path("docs/benchmarks/runs")
SMOKE = Path("runs/eval/smoke")
LoadFn = Callable[[str], np.ndarray]


# ---------------------------------------------------------------- data
def load_eval_data(
    cfg: dict[str, Any], limit: int | None
) -> tuple[pd.DataFrame, LoadFn, dict[str, Any]]:
    from ml.data.license_gate import DataPolicy, enforce
    from ml.data.manifest import read_manifest
    from ml.training.dataset import soundfile_loader

    policy = DataPolicy.from_config(cfg)
    rows, sets, sha = [], [], hashlib.sha256()
    for es in cfg["eval_sets"]:
        files = sorted(glob.glob(es["manifests"]))
        if not files:
            raise FileNotFoundError(f"eval set {es['name']}: no manifest matches {es['manifests']}")
        set_rows = [r for f in files for r in read_manifest(f)]
        if "split" in es:
            ids = set(
                json.loads(Path(es["split"]["file"]).read_text(encoding="utf-8"))["splits"][
                    es["split"]["name"]
                ]
            )
            set_rows = [r for r in set_rows if r.utt_id in ids]
        n_max = limit or es.get("sample")
        if n_max and len(set_rows) > n_max:
            set_rows = stratified(set_rows, int(n_max), int(cfg.get("sample_seed", 0)))
        for f in files:
            sha.update(Path(f).read_bytes())
        sha.update(f"{es['name']}:{len(set_rows)}".encode())  # sampled sets hash differently
        rows += set_rows
        sets.append(es["name"])
    enforce(rows, policy, purpose="eval")
    load_row = soundfile_loader(cfg.get("data_root", "data"))
    by_id = {r.utt_id: r for r in rows}
    df = pd.DataFrame(
        {
            "utt_id": r.utt_id,
            "label": r.label,
            "attack_family": r.generator_family,
            "generator": r.generator_name,
            "dataset": r.source_corpus,
            "language": r.language,
            "accent": r.accent,
            "gender": r.gender,
            "speaker_id": r.speaker_id,
            "codec": "+".join(c for c in r.codec_chain if c.startswith("codec:")) or "none",
            "snr_db": r.snr_db,
        }
        for r in rows
    )
    info = {
        "eval_sets": sets,
        "lineage": policy.lineage,
        "manifest_sha256": sha.hexdigest(),
        "synthetic": False,
    }
    return df, lambda u: load_row(by_id[u]), info


def stratified(rows: list[Any], n: int, seed: int) -> list[Any]:
    """Seeded sample keeping every (label, attack, codec) group, proportionally sized.

    Taking the first N rows of a protocol would silently drop whole attacks or codecs.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str, str], list[Any]] = {}
    for r in rows:
        codec = next((c for c in r.codec_chain if c.startswith("codec:")), "none")
        groups.setdefault((r.label, r.generator_family or "-", codec), []).append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        k = max(1, round(n * len(g) / len(rows)))
        out += [g[i] for i in sorted(rng.choice(len(g), size=min(k, len(g)), replace=False))]
    return out


def synthetic_eval(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, LoadFn, dict[str, Any]]:
    """Toy corpus with every metadata axis populated, to exercise all protocols."""
    rng = np.random.default_rng(seed)
    audio: dict[str, np.ndarray] = {}
    recs = []
    langs, genders = ["hi", "ta", "bn", "en"], ["female", "male"]
    for i in range(n):
        spoof = i % 2 == 1
        ds = "synthetic_in_domain" if i % 3 else "synthetic_cross"
        fam = f"family_{(i // 2) % 4}" if spoof else None
        t = np.arange(3 * 16000) / 16000
        f0 = rng.uniform(100, 250) * (1 + (0 if spoof else 0.02 * np.sin(2 * np.pi * 5 * t)))
        x = sum(np.sin(2 * np.pi * np.cumsum(f0 * k) / 16000) / k for k in range(1, 6))
        noise = (0.05 if spoof else 0.3) + (0.25 if ds == "synthetic_cross" else 0.0)
        x = x + noise * rng.standard_normal(len(t))
        utt = f"syn{i:05d}"
        audio[utt] = (0.3 * x / np.max(np.abs(x))).astype(np.float32)
        recs.append(
            {
                "utt_id": utt,
                "label": "spoof" if spoof else "bona_fide",
                "attack_family": fam,
                "generator": fam,
                "dataset": ds,
                "language": langs[i % 4],
                "accent": "indian_english" if langs[i % 4] == "en" else None,
                "gender": genders[(i // 4) % 2],
                "speaker_id": f"spk{i % 30}",
                "codec": ["none", "codec:g711u"][(i // 8) % 2],
                "snr_db": float(rng.uniform(0, 30)),
            }
        )
    info = {
        "eval_sets": ["synthetic"],
        "lineage": "synthetic",
        "manifest_sha256": f"synthetic-{seed}-{n}",
        "synthetic": True,
        "train_datasets": ["synthetic_in_domain"],
        "seen_families": ["family_0", "family_1"],
    }
    return pd.DataFrame(recs), audio.__getitem__, info


# ---------------------------------------------------------------- scoring
def score_all(
    scorer: Scorer,
    df: pd.DataFrame,
    load: LoadFn,
    transform: adv.Transform | None = None,
    only_spoof: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outs: list[ScoreOut] = []
    for utt, label in zip(df["utt_id"], df["label"], strict=True):
        x = load(utt)
        if transform is not None and (not only_spoof or label == "spoof"):
            x = transform(x, rng)
        outs.append(scorer.score(x))
    return df.assign(
        score=[o.score for o in outs],
        p_spoof=[o.p_spoof for o in outs],
        window_scores=[o.window_p for o in outs],
    )


def balanced_subset(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    parts = [g.sample(min(len(g), n // 2), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("utt_id")


def window_threshold(scored: pd.DataFrame) -> float | None:
    """Window-level alert threshold at the window EER (data-derived, no invented constant)."""
    b = [
        p
        for w, lab in zip(scored["window_scores"], scored["label"], strict=True)
        if lab == "bona_fide"
        for p in w
        if np.isfinite(p)
    ]
    s = [
        p
        for w, lab in zip(scored["window_scores"], scored["label"], strict=True)
        if lab == "spoof"
        for p in w
        if np.isfinite(p)
    ]
    if not b or not s:
        return None
    _, thr = M.eer(-np.array(b), -np.array(s))  # p_spoof -> bona fide score
    return -thr


# ---------------------------------------------------------------- main
def evaluate(
    scorer: Scorer,
    df: pd.DataFrame,
    load: LoadFn,
    info: dict[str, Any],
    cfg: dict[str, Any],
    log: Callable[[str], None] = print,
) -> RunRecord:
    boot = int(cfg.get("bootstrap", 200))
    rec = RunRecord(
        model_version=scorer.model_version,
        scorer=scorer.name,
        config={k: v for k, v in cfg.items() if k != "eval_sets"},
        data=info,
    )
    meta = getattr(scorer, "meta", {}) or {}
    seen = list(
        meta.get("train_families") or info.get("seen_families") or cfg.get("seen_families") or []
    )
    train_ds = list(
        meta.get("train_datasets") or info.get("train_datasets") or cfg.get("train_datasets") or []
    )
    rec.data["seen_families"], rec.data["train_datasets"] = seen, train_ds
    for a in meta.get("augmentation") or []:
        codecs = sorted((a.get("channel", {}).get("codec_weights") or {}).keys() - {"clean"})
        rec.notes.append(
            f"trained with channel augmentation ({a.get('split')}: {', '.join(codecs)}); "
            "the codec/SNR sweeps below use the same encoders, so they are not independent - "
            "the per-codec rows of the eval corpus are the independent check"
        )

    log(f"scoring {len(df)} utterances with {scorer.name} ({scorer.model_version})")
    scored = score_all(scorer, df, load)
    rec.add_rows([P.summarize(scored, "overall", "all", boot)])
    rec.add_rows(
        P.logo(scored, seen, boot)
        if seen
        else P.by_group(scored, "attack_family", "logo", True, boot)
    )
    rec.add_rows(P.cross_dataset(scored, train_ds, boot))
    rec.add_rows(P.per_language(scored, boot) + P.per_accent(scored, boot))
    rec.add_rows(P.per_codec(scored, boot) + P.per_snr(scored, boot))

    scored_ok = scored[np.isfinite(scored["score"].astype(float))]
    op = cfg.get("operating_point", {"mode": "eer"})
    if scored_ok["label"].nunique() == 2:
        thr = F.operating_threshold(scored_ok, op.get("mode", "eer"), op.get("target_fpr"))
        fcfg = cfg.get("fairness", {})
        res = [F.fpr_by(scored_ok, a, thr) for a in fcfg.get("attributes", ["gender", "language"])]
        rec.add_fairness(
            res, F.fairness_gate(res, fcfg.get("max_gap"), fcfg.get("min_group_n", 30))
        )
    else:
        rec.notes.append("fairness skipped: scorer abstained on every trial of at least one class")

    wthr = window_threshold(scored)
    if wthr is not None:
        rec.add_rows(P.operational(scored, wthr))

    sw = cfg.get("sweeps", {})
    if sw:
        sub = balanced_subset(df, int(sw.get("max_utts", 200)))
        conds = {"clean": None}
        conds |= {
            f"codec_{c}": adv.codec_roundtrip(c) for c in adv.available_codecs(sw.get("codecs", []))
        }
        missing = sorted(
            set(sw.get("codecs", [])) - set(adv.available_codecs(sw.get("codecs", [])))
        )
        if missing:
            rec.notes.append(f"codec sweep skipped unavailable encoders: {', '.join(missing)}")
        snrs = {f"snr_{s}dB": adv.add_noise(float(s)) for s in sw.get("snr_db", [])}
        for proto, group in (("codec_sweep", conds), ("snr_sweep", {"clean": None} | snrs)):
            parts = []
            for name, tf in group.items():
                log(f"{proto}: {name}")
                parts.append(score_all(scorer, sub, load, tf).assign(condition=name))
            rec.add_rows(P.per_condition(pd.concat(parts), proto, boot))

    ad = cfg.get("adversarial", {})
    if ad:
        sub = balanced_subset(df, int(ad.get("max_utts", 200)))
        parts = [score_all(scorer, sub, load).assign(condition="clean")]
        if ad.get("laundering", True):
            for name, tf in adv.laundering_suite(
                ad.get("codecs", ["g711u", "amr_nb", "opus"])
            ).items():
                log(f"laundering: {name}")
                parts.append(
                    score_all(scorer, sub, load, tf, only_spoof=True).assign(condition=name)
                )
        if ad.get("white_box", True) and isinstance(scorer, HeadAScorer):
            parts += white_box(scorer, sub, load, ad, rec)
        elif ad.get("white_box", True):
            rec.notes.append(
                "white-box attacks skipped: scorer is not differentiable (head_a only)"
            )
        rec.add_rows(P.per_condition(pd.concat(parts), "adversarial", boot))
    return rec


def white_box(
    scorer: HeadAScorer, sub: pd.DataFrame, load: LoadFn, ad: dict[str, Any], rec: RunRecord
) -> list[pd.DataFrame]:
    import torch

    def crop(x: np.ndarray) -> np.ndarray:
        return x[:48000] if len(x) >= 48000 else np.pad(x, (0, 48000 - len(x)))

    def score_fn(x: torch.Tensor) -> torch.Tensor:
        return scorer.model(x.to(scorer.device))[1].cpu()

    spoof_ids = sub.loc[sub["label"] == "spoof", "utt_id"].tolist()
    if not spoof_ids:
        return []
    wav = torch.from_numpy(np.stack([crop(load(u)) for u in spoof_ids]).astype(np.float32))
    eps = float(ad.get("pgd_eps", 0.002))
    x_pgd = torch.cat(
        [
            adv.pgd(score_fn, wav[i : i + 8], eps=eps, steps=int(ad.get("pgd_steps", 10)))
            for i in range(0, len(wav), 8)
        ]
    )
    train_n = max(1, len(wav) // 2)  # learn the filter on half, apply to all: measures transfer
    h = adv.learn_universal_filter(score_fn, wav[:train_n], steps=int(ad.get("filter_steps", 60)))
    x_flt = adv.apply_filter(wav, h)
    # white-box attacks run on 3 s crops, so they are compared with a 3 s-crop clean reference
    out = [score_all(scorer, sub, lambda u: crop(load(u))).assign(condition="clean_3s_crop")]
    snr = np.mean([adv.perturbation_snr(wav[i].numpy(), x_pgd[i].numpy()) for i in range(len(wav))])
    rec.notes.append(
        f"pgd_linf: eps={eps}, mean perturbation SNR {snr:.1f} dB on {len(wav)} spoofs"
    )
    mag_db = 20 * np.log10(np.abs(np.fft.rfft(h.numpy(), 512)) + 1e-9)
    rec.notes.append(
        f"universal_filter: {len(h)}-tap FIR learnt on {train_n} spoofs, applied to {len(wav)}; "
        f"gain range {mag_db.min():.1f}..{mag_db.max():.1f} dB (linear filtering: SNR not meaningful)"
    )
    for name, attacked in (("pgd_linf", x_pgd), ("universal_filter", x_flt)):
        cache = dict(zip(spoof_ids, attacked.numpy(), strict=True))

        def load_attacked(u: str, c: dict[str, np.ndarray] = cache) -> np.ndarray:
            return c[u] if u in c else crop(load(u))  # bona fide: same 3 s crop, untouched

        out.append(score_all(scorer, sub, load_attacked).assign(condition=name))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="scorer spec or registered model version")
    ap.add_argument("--config", type=Path, default=Path("ml/eval/configs/default.yaml"))
    ap.add_argument("--synthetic", action="store_true", help="toy data; smoke test only")
    ap.add_argument("--limit", type=int, default=None, help="max utterances per eval set")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--allow-synthetic-in-report", action="store_true")
    ap.add_argument("--tag", default="", help="names this eval-set collection (saved side by side)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.synthetic:
        df, load, info = synthetic_eval(int(cfg.get("synthetic_n", 240)))
        runs_dir, out = args.runs_dir or SMOKE / "runs", args.out or SMOKE / "REPORT.md"
        if out.resolve() == REPORT.resolve() and not args.allow_synthetic_in_report:
            print("refusing to write synthetic numbers into the published report", file=sys.stderr)
            return 2
    else:
        df, load, info = load_eval_data(cfg, args.limit)
        runs_dir, out = args.runs_dir or RUNS, args.out or REPORT
    scorer = build_scorer(args.model)
    rec = evaluate(scorer, df, load, info, cfg)
    rec.tag = args.tag
    path = rec.save(runs_dir)
    rebuild(runs_dir, out, allow_synthetic=args.synthetic)
    print(
        f"run {rec.run_id} -> {path}; report -> {out}; fairness gate: {rec.gate.get('status', 'n/a')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
