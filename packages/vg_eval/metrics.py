"""vg_eval.metrics — detection metrics for the benchmark report (B15-T09).

Score convention (ASVspoof): **higher score = more bona fide**. A detector that
emits ``p_spoof`` is converted with ``score = -p_spoof`` (or ``1 - p_spoof``)
before calling anything here. Every function takes the two score sets
separately so that label bugs cannot silently invert a metric.

Implemented:
* ``eer`` — equal error rate + threshold (ROC convex-hull-free, exact on the
  empirical curve; midpoint of FRR/FAR at the crossing).
* ``min_dcf`` — ASVspoof 5 countermeasure normalised minimum DCF
  (C_miss=1, C_fa=10, π_spoof=0.05 by default).
* ``min_tdcf`` — ASVspoof 2019 tandem DCF, given the ASV system's error rates.
* ``min_adcf`` — ASVspoof 5 / SASV architecture-agnostic DCF over target,
  non-target and spoof trials of a single SASV score.
* ``auc``, ``pauc`` (area under ROC restricted to FPR ≤ max_fpr, McClish-standardised).
* ``ece`` — expected calibration error of p_spoof.
* ``bootstrap_ci`` — percentile bootstrap CI for any two-sample metric.

"FPR" in this module means the false-alarm rate on **bona fide** audio (a real
customer flagged as synthetic) — the error that hurts customers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _clean(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def det_curve(bona: np.ndarray, spoof: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frr, far, thresholds): at threshold t, accept as bona fide iff score > t.

    frr = bona fide rejected (score <= t); far = spoof accepted (score > t).
    """
    bona, spoof = _clean(bona), _clean(spoof)
    if len(bona) == 0 or len(spoof) == 0:
        raise ValueError("need at least one bona fide and one spoof score")
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    frr = np.concatenate([[0.0], np.cumsum(labels) / len(bona)])
    far = np.concatenate([[1.0], 1.0 - np.cumsum(1 - labels) / len(spoof)])
    thr = np.concatenate([[scores[0] - 1e-9], scores])
    return frr, far, thr


def eer(bona: np.ndarray, spoof: np.ndarray) -> tuple[float, float]:
    frr, far, thr = det_curve(bona, spoof)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(thr[i])


def rates_at(bona: np.ndarray, spoof: np.ndarray, threshold: float) -> tuple[float, float]:
    """(bona fide false-alarm rate, spoof miss rate) when flagging score <= threshold as spoof."""
    b, s = _clean(bona), _clean(spoof)
    fpr = float(np.mean(b <= threshold)) if len(b) else float("nan")
    fnr = float(np.mean(s > threshold)) if len(s) else float("nan")
    return fpr, fnr


def threshold_at_fpr(bona: np.ndarray, target_fpr: float) -> float:
    """Largest threshold whose bona fide false-alarm rate is <= target_fpr."""
    b = np.sort(_clean(bona))
    k = int(np.floor(target_fpr * len(b)))
    return float(b[k - 1]) if k > 0 else float(b[0] - 1e-9)


@dataclass(frozen=True)
class DCFParams:
    """ASVspoof 5 CM defaults (evaluation plan, phase 1)."""

    c_miss: float = 1.0
    c_fa: float = 10.0
    p_spoof: float = 0.05


DCF_DEFAULT = DCFParams()


def min_dcf(bona: np.ndarray, spoof: np.ndarray, p: DCFParams = DCF_DEFAULT) -> float:
    """Normalised minimum DCF; 1.0 == a system that always accepts or always rejects."""
    frr, far, _ = det_curve(bona, spoof)
    dcf = p.c_miss * (1 - p.p_spoof) * frr + p.c_fa * p.p_spoof * far
    default = min(p.c_miss * (1 - p.p_spoof), p.c_fa * p.p_spoof)
    return float(np.min(dcf) / default)


@dataclass(frozen=True)
class TDCFParams:
    """ASVspoof 2019 t-DCF cost model (legacy formulation, Kinnunen et al. 2018)."""

    pi_tar: float = 0.9405
    pi_non: float = 0.0095
    pi_spoof: float = 0.05
    c_miss_asv: float = 1.0
    c_fa_asv: float = 10.0
    c_miss_cm: float = 1.0
    c_fa_cm: float = 10.0


TDCF_DEFAULT = TDCFParams()


def min_tdcf(
    bona: np.ndarray,
    spoof: np.ndarray,
    asv_p_miss: float,
    asv_p_fa: float,
    asv_p_miss_spoof: float,
    p: TDCFParams = TDCF_DEFAULT,
) -> float:
    """Normalised min t-DCF of a CM in tandem with an ASV system of the given error rates.

    ``asv_p_miss_spoof`` is the ASV miss rate on spoof trials (1 - spoof false-accept rate).
    """
    frr, far, _ = det_curve(bona, spoof)
    c1 = p.pi_tar * (p.c_miss_cm - p.c_miss_asv * asv_p_miss) - p.pi_non * p.c_fa_asv * asv_p_fa
    c2 = p.c_fa_cm * p.pi_spoof * (1 - asv_p_miss_spoof)
    if c1 <= 0 or c2 <= 0:
        raise ValueError("t-DCF undefined for these ASV error rates (C1 or C2 <= 0)")
    tdcf = c1 * frr + c2 * far
    return float(np.min(tdcf) / min(c1, c2))


@dataclass(frozen=True)
class ADCFParams:
    """a-DCF cost model (Shim et al., 2024; ASVspoof 5 SASV track). Verify the
    constants against the evaluation plan in use before publishing numbers."""

    c_miss: float = 1.0
    c_fa_non: float = 10.0
    c_fa_spoof: float = 10.0
    pi_tar: float = 0.9
    pi_non: float = 0.05
    pi_spoof: float = 0.05


ADCF_DEFAULT = ADCFParams()


def min_adcf(
    target: np.ndarray, nontarget: np.ndarray, spoof: np.ndarray, p: ADCFParams = ADCF_DEFAULT
) -> float:
    """Normalised min a-DCF of one SASV score (accept iff score > t)."""
    tar, non, spf = _clean(target), _clean(nontarget), _clean(spoof)
    thr = np.unique(np.concatenate([tar, non, spf, [-np.inf]]))
    p_miss = np.array([np.mean(tar <= t) for t in thr])
    p_fa_non = np.array([np.mean(non > t) for t in thr])
    p_fa_spf = np.array([np.mean(spf > t) for t in thr])
    adcf = (
        p.c_miss * p.pi_tar * p_miss
        + p.c_fa_non * p.pi_non * p_fa_non
        + p.c_fa_spoof * p.pi_spoof * p_fa_spf
    )
    default = min(p.c_miss * p.pi_tar, p.c_fa_non * p.pi_non + p.c_fa_spoof * p.pi_spoof)
    return float(np.min(adcf) / default)


def roc(bona: np.ndarray, spoof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) with the spoof class as positive: fpr = bona flagged, tpr = spoof flagged."""
    frr, far, _ = det_curve(bona, spoof)
    return frr, 1.0 - far  # flagging score <= t: fpr = frr(bona), tpr = spoof rejected


def auc(bona: np.ndarray, spoof: np.ndarray) -> float:
    """P(score_bona > score_spoof) + 0.5 P(tie) (Mann-Whitney)."""
    b, s = _clean(bona), _clean(spoof)
    allv = np.concatenate([b, s])
    ranks = _rankdata(allv)
    r_b = ranks[: len(b)].sum()
    return float((r_b - len(b) * (len(b) + 1) / 2) / (len(b) * len(s)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pauc(bona: np.ndarray, spoof: np.ndarray, max_fpr: float = 0.01) -> float:
    """McClish-standardised partial AUC for FPR in [0, max_fpr] (0.5 = chance, 1 = perfect)."""
    fpr, tpr = roc(bona, spoof)
    order = np.lexsort((tpr, fpr))
    fpr, tpr = fpr[order], tpr[order]
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x = np.concatenate([fpr[:stop], [max_fpr]])
    y = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    area = float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))
    min_area, max_area = max_fpr**2 / 2, max_fpr
    return float(0.5 * (1 + (area - min_area) / (max_area - min_area)))


def ece(p_spoof: np.ndarray, is_spoof: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of p_spoof (equal-width bins)."""
    p = np.asarray(p_spoof, dtype=np.float64)
    y = np.asarray(is_spoof, dtype=np.float64)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    bona: np.ndarray,
    spoof: np.ndarray,
    n: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    b, s = _clean(bona), _clean(spoof)
    vals = []
    for _ in range(n):
        vals.append(metric(rng.choice(b, len(b)), rng.choice(s, len(s))))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
