"""Indic prosody care: accent-aware normalisation + per-language evaluation (B6-T06).

Two safeguards against the catastrophic false-positive rates English-trained
prosody models produce on Indian speakers:

1. **Language-aware measurement.** Geminates and retroflex stops create silent
   closures of 100–200 ms that an English-tuned pause detector counts as
   pauses, inflating "choppiness". Indic languages use a longer minimum pause.
2. **Per-language z-scoring.** Each prosodic feature is normalised with bona fide
   statistics *of the same language* (fallback: global) before the model sees
   it, so "normal for Tamil" is not scored against "normal for English".

``per_language_report`` produces the per-language FPR / EER table the training
script stores and B15 re-checks. A large FPR gap is a release blocker.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

INDIC = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur", "sa"}
MIN_PAUSE_MS_DEFAULT = 150.0
MIN_PAUSE_MS_INDIC = 220.0
MIN_LANG_COUNT = 30


def base_language(language: str | None) -> str | None:
    return language.split("-")[0].lower() if language else None


def min_pause_ms(language: str | None) -> float:
    return MIN_PAUSE_MS_INDIC if base_language(language) in INDIC else MIN_PAUSE_MS_DEFAULT


@dataclass
class LanguageNorm:
    global_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_language: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: list[dict[str, float]], languages: list[str | None]) -> LanguageNorm:
        names = list(rows[0]) if rows else []

        def stats(subset: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
            out = {}
            for n in names:
                vals = np.array([r[n] for r in subset], dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 2:
                    out[n] = (float(vals.mean()), float(max(vals.std(), 1e-6)))
            return out

        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        for r, lang in zip(rows, languages, strict=True):
            if base_language(lang):
                groups[base_language(lang)].append(r)  # type: ignore[index]
        return cls(
            global_stats=stats(rows),
            per_language={k: stats(v) for k, v in groups.items() if len(v) >= MIN_LANG_COUNT},
        )

    def apply(self, feats: dict[str, float], language: str | None) -> tuple[dict[str, float], str]:
        lang = base_language(language)
        table = self.per_language.get(lang or "", None)
        source = lang if table is not None else "global"
        table = table or self.global_stats
        out = {}
        for n, v in feats.items():
            mu_sd = table.get(n) or self.global_stats.get(n)
            out[n] = (v - mu_sd[0]) / mu_sd[1] if mu_sd and not math.isnan(v) else v
        return out, source

    def to_json(self) -> dict[str, object]:
        return {"global": self.global_stats, "per_language": self.per_language}

    @classmethod
    def from_json(cls, d: dict[str, object]) -> LanguageNorm:
        g = {k: tuple(v) for k, v in d.get("global", {}).items()}  # type: ignore[union-attr]
        pl = {
            lang: {k: tuple(v) for k, v in tbl.items()}
            for lang, tbl in d.get("per_language", {}).items()  # type: ignore[union-attr]
        }
        return cls(g, pl)  # type: ignore[arg-type]


def per_language_report(
    p_spoof: np.ndarray,
    is_spoof: np.ndarray,
    languages: list[str | None],
    threshold: float = 0.5,
    max_fpr_gap: float = 0.05,
) -> dict[str, object]:
    """FPR (bona fide flagged) and miss rate per language at ``threshold``, plus a gap verdict."""
    table: dict[str, dict[str, float]] = {}
    langs = np.array([base_language(lg) or "unknown" for lg in languages])
    for lang in sorted(set(langs)):
        m = langs == lang
        bona, spoof = m & (is_spoof == 0), m & (is_spoof == 1)
        table[lang] = {
            "n_bona_fide": int(bona.sum()),
            "n_spoof": int(spoof.sum()),
            "fpr": float((p_spoof[bona] >= threshold).mean()) if bona.any() else float("nan"),
            "miss_rate": (
                float((p_spoof[spoof] < threshold).mean()) if spoof.any() else float("nan")
            ),
        }
    fprs = [v["fpr"] for v in table.values() if not math.isnan(v["fpr"]) and v["n_bona_fide"] >= 20]
    gap = (max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
    return {
        "threshold": threshold,
        "languages": table,
        "max_fpr_gap": gap,
        "fairness_ok": gap <= max_fpr_gap,
    }
