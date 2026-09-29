"""vg_eval.protocols — evaluation protocols over a scored table (B15-T02…T07).

Input is a *score table*: one row per evaluated utterance (``pandas.DataFrame``)
with at least

    utt_id, label ("bona_fide" | "spoof"), score (higher = more bona fide; NaN when
    the scorer abstained)

and optionally ``p_spoof``, ``attack_family``, ``generator``, ``dataset``,
``language``, ``accent``, ``gender``, ``speaker_id``, ``codec``, ``snr_db``,
``condition`` (e.g. an adversarial transform), ``window_scores`` (list, for
operational metrics).

Every protocol returns a list of ``Row`` objects (one report line each) — the
report renderer (``report.py``) gives every row a stable id so that claims can
cite it. Slices with too few trials are still reported, but flagged
``low_n``; nothing is silently dropped.

Abstentions are first-class: EER is computed on scored trials only, and
``coverage`` (fraction scored) is reported next to it. A detector that abstains
on hard cases must not look better than one that answers them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from packages.vg_eval import metrics as M

MIN_TRIALS = 30  # per class; below this a slice is reported but flagged low_n


@dataclass
class Row:
    protocol: str  # overall | logo | cross_dataset | language | accent | codec | snr | ...
    slice: str  # e.g. "tts_vits", "in_the_wild", "hi"
    n_bona: int
    n_spoof: int
    coverage: float
    eer: float | None = None
    eer_ci: tuple[float, float] | None = None
    min_dcf: float | None = None
    auc: float | None = None
    pauc_1pct: float | None = None
    ece: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def low_n(self) -> bool:
        return min(self.n_bona, self.n_spoof) < MIN_TRIALS


def _split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = df["score"].to_numpy(dtype=float)
    lab = df["label"].to_numpy()
    return s[lab == "bona_fide"], s[lab == "spoof"]


def summarize(
    df: pd.DataFrame,
    protocol: str,
    slice_: str,
    bootstrap: int = 200,
    **extra: Any,  # noqa: ANN401
) -> Row:
    bona, spoof = _split(df)
    scored = np.isfinite(df["score"].to_numpy(dtype=float))
    row = Row(
        protocol=protocol,
        slice=slice_,
        n_bona=int(np.isfinite(bona).sum()),
        n_spoof=int(np.isfinite(spoof).sum()),
        coverage=float(scored.mean()) if len(df) else 0.0,
        extra=dict(extra),
    )
    if row.n_bona == 0 or row.n_spoof == 0:
        row.extra.setdefault("note", "needs both classes")
        return row
    row.eer = M.eer(bona, spoof)[0]
    if bootstrap:
        row.eer_ci = M.bootstrap_ci(lambda b, s: M.eer(b, s)[0], bona, spoof, n=bootstrap)
    row.min_dcf = M.min_dcf(bona, spoof)
    row.auc = M.auc(bona, spoof)
    row.pauc_1pct = M.pauc(bona, spoof, 0.01)
    if "p_spoof" in df and df["p_spoof"].notna().any():
        row.ece = M.ece(df["p_spoof"].to_numpy(dtype=float), (df["label"] == "spoof").to_numpy())
    return row


def by_group(
    df: pd.DataFrame,
    col: str,
    protocol: str,
    shared_bona: bool = False,
    bootstrap: int = 200,
) -> list[Row]:
    """One row per value of ``col``.

    ``shared_bona=True`` is for attributes only spoofs carry (attack family,
    generator): every family is scored against the full bona fide pool.
    """
    if col not in df:
        return []
    out = []
    bona_all = df[df["label"] == "bona_fide"]
    for value, g in df.groupby(df[col].fillna("unknown"), sort=True):
        if shared_bona:
            if value == "unknown":
                continue  # the bona fide rows themselves
            g = pd.concat([bona_all, g[g["label"] == "spoof"]])
        out.append(summarize(g, protocol, str(value), bootstrap))
    return out


# ---------------------------------------------------------------- T02 leave-one-generator-out
def logo_folds(families: Iterable[str]) -> list[dict[str, Any]]:
    """Fold specs for true LOGO retraining: train on all families but one."""
    fams = sorted(set(families))
    return [{"held_out": f, "train_families": [x for x in fams if x != f]} for f in fams]


def logo(df: pd.DataFrame, seen_families: Sequence[str], bootstrap: int = 200) -> list[Row]:
    """Per attack family EER, marking families the model never saw in training.

    The pooled ``unseen`` row is the number that predicts field performance.
    """
    rows = by_group(df, "attack_family", "logo", shared_bona=True, bootstrap=bootstrap)
    seen = set(seen_families)
    for r in rows:
        r.extra["seen_in_training"] = r.slice in seen
    spoof = df[df["label"] == "spoof"]
    bona = df[df["label"] == "bona_fide"]
    unseen = spoof[~spoof["attack_family"].isin(seen)]
    seen_df = spoof[spoof["attack_family"].isin(seen)]
    if len(unseen):
        rows.append(summarize(pd.concat([bona, unseen]), "logo", "ALL_UNSEEN", bootstrap))
    if len(seen_df):
        rows.append(summarize(pd.concat([bona, seen_df]), "logo", "ALL_SEEN", bootstrap))
    return rows


# ---------------------------------------------------------------- T03 cross-dataset
def cross_dataset(
    df: pd.DataFrame, train_datasets: Sequence[str], bootstrap: int = 200
) -> list[Row]:
    rows = by_group(df, "dataset", "cross_dataset", bootstrap=bootstrap)
    ref = [r.eer for r in rows if r.slice in set(train_datasets) and r.eer is not None]
    in_domain = float(np.mean(ref)) if ref else None
    for r in rows:
        r.extra["in_domain"] = r.slice in set(train_datasets)
        if in_domain is not None and r.eer is not None:
            r.extra["eer_minus_in_domain"] = round(r.eer - in_domain, 4)
    return rows


# ---------------------------------------------------------------- T04 language / accent
def per_language(df: pd.DataFrame, bootstrap: int = 200) -> list[Row]:
    return by_group(df, "language", "language", bootstrap=bootstrap)


def per_accent(df: pd.DataFrame, bootstrap: int = 200) -> list[Row]:
    return by_group(df, "accent", "accent", bootstrap=bootstrap)


# ---------------------------------------------------------------- T06 codec / SNR
SNR_BINS = (-np.inf, 0, 5, 10, 15, 20, 30, np.inf)


def snr_bin(snr: float | None) -> str:
    if snr is None or not np.isfinite(snr):
        return "clean/unknown"
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:], strict=True):
        if lo <= snr < hi:
            return f"[{lo:g},{hi:g}) dB"
    return "clean/unknown"


def per_codec(df: pd.DataFrame, bootstrap: int = 200) -> list[Row]:
    return by_group(df, "codec", "codec", bootstrap=bootstrap)


def per_snr(df: pd.DataFrame, bootstrap: int = 200) -> list[Row]:
    if "snr_db" not in df:
        return []
    d = df.assign(snr_bin=df["snr_db"].map(snr_bin))
    return by_group(d, "snr_bin", "snr", bootstrap=bootstrap)


WHITE_BOX = {"pgd_linf", "universal_filter"}


def per_condition(df: pd.DataFrame, protocol: str, bootstrap: int = 200) -> list[Row]:
    """Rows per ``condition`` (codec sweep, SNR sweep, adversarial transform)."""
    rows = by_group(df, "condition", protocol, bootstrap=bootstrap)
    base = {r.slice: r.eer for r in rows if r.slice.startswith("clean")}
    for r in rows:
        # white-box conditions (3 s crops) compare with the cropped clean reference
        ref = base.get("clean_3s_crop") if r.slice in WHITE_BOX else base.get("clean")
        if ref is not None and r.eer is not None and not r.slice.startswith("clean"):
            r.extra["eer_minus_clean"] = round(r.eer - ref, 4)
    return rows


# ---------------------------------------------------------------- T07 operational
def time_to_first_alert(
    window_p_spoof: Sequence[float],
    threshold: float,
    hop_s: float = 1.0,
    first_window_s: float = 3.0,
) -> float | None:
    """Seconds of audio until the first window with p_spoof >= threshold (None = never)."""
    for i, p in enumerate(window_p_spoof):
        if p is not None and np.isfinite(p) and p >= threshold:
            return first_window_s + i * hop_s
    return None


def score_stability(window_p_spoof: Sequence[float], threshold: float) -> dict[str, float]:
    """Within-call jitter: std of window scores and decision flips per minute."""
    p = np.array([x for x in window_p_spoof if x is not None and np.isfinite(x)], dtype=float)
    if len(p) < 2:
        return {"std": float("nan"), "flips_per_min": float("nan")}
    flips = int(np.sum(np.diff((p >= threshold).astype(int)) != 0))
    return {"std": float(p.std()), "flips_per_min": flips / (len(p) / 60.0)}


def _nanmean(v: Sequence[float]) -> float | None:
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def operational(df: pd.DataFrame, threshold: float) -> list[Row]:
    """Time-to-first-alert on spoofed calls, false-alert rate and stability on bona fide calls."""
    if "window_scores" not in df:
        return []
    out = []
    for label in ("spoof", "bona_fide"):
        g = df[df["label"] == label]
        if g.empty:
            continue
        ttfa = [time_to_first_alert(w, threshold) for w in g["window_scores"]]
        hit = [t for t in ttfa if t is not None]
        stab = [score_stability(w, threshold) for w in g["window_scores"]]
        extra = {
            "threshold": threshold,
            "alerted_fraction": round(len(hit) / len(g), 4),
            "median_seconds_to_first_alert": float(np.median(hit)) if hit else None,
            "p90_seconds_to_first_alert": float(np.quantile(hit, 0.9)) if hit else None,
            "mean_window_std": _nanmean([s["std"] for s in stab]),
            "mean_flips_per_min": _nanmean([s["flips_per_min"] for s in stab]),
        }
        out.append(
            Row(
                protocol="operational",
                slice=label,
                n_bona=len(g) if label == "bona_fide" else 0,
                n_spoof=len(g) if label == "spoof" else 0,
                coverage=float(np.isfinite(g["score"].to_numpy(dtype=float)).mean()),
                extra=extra,
            )
        )
    return out
