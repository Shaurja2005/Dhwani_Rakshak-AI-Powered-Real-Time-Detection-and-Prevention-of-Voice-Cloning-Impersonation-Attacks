"""Breath groups (B6-T02) and pause / speaking-rate regularity (B6-T03).

Breaths: unvoiced, broadband-noise, low-level events of 120–800 ms that end
shortly before a phrase starts (inhalation precedes speech). Neural TTS often
omits them or inserts identical ones.

Rhythm: pauses are non-speech runs longer than a *language-dependent* minimum
(so geminate / retroflex closures in Indic languages are not counted as pauses,
see normalize.py); phrases are speech runs between pauses. Over-regularity
(low coefficient of variation of pauses, phrase lengths and syllable rate) is
the classic TTS tell.

Also detects acoustic *filled pauses* ("uhh", "mmm"): sustained voiced runs with
flat pitch and flat energy — a disfluency cue that needs no ASR.
"""

from __future__ import annotations

import numpy as np

from packages.vg_models.heads.head_c_prosody.pitch import FRAME, HOP, SR, PitchTrack, _runs, frames

FRAME_S = HOP / SR
_FREQS = np.fft.rfftfreq(FRAME, 1 / SR)
_BREATH_BAND = (_FREQS >= 100) & (_FREQS <= 4000)


def _cv(x: list[float] | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.mean() <= 0:
        return float("nan")
    return float(x.std() / x.mean())


def speech_mask(p: PitchTrack) -> np.ndarray:
    floor = np.percentile(p.frame_db, 10)
    loud = p.frame_db > floor + 10.0
    # Bridge 1-2 frame gaps so plosive bursts don't split phrases.
    m = loud | p.voiced
    for k in (1, 2):
        m[k:-k] |= m[: -2 * k] & m[2 * k :]
    return m


def breath_features(x: np.ndarray, p: PitchTrack, speech: np.ndarray) -> dict[str, float]:
    fr = frames(np.asarray(x, dtype=np.float64))[: len(p.frame_db)]
    spec = np.abs(np.fft.rfft(fr * np.hanning(FRAME), axis=1)) ** 2 + 1e-12
    band = spec[:, _BREATH_BAND]
    flat = np.exp(np.log(band).mean(axis=1)) / band.mean(axis=1)

    floor = np.percentile(p.frame_db, 10)
    speech_level = np.median(p.frame_db[p.voiced]) if p.voiced.sum() > 5 else p.frame_db.max()
    cand = (
        (~p.voiced)
        & (p.frame_db > floor + 4.0)
        & (p.frame_db < speech_level - 10.0)
        & (flat > 0.15)
    )

    phrase_starts = [a for a, _ in _runs(p.voiced) if a > 0]
    breaths: list[tuple[int, int]] = []
    for a, b in _runs(cand):
        dur = (b - a) * FRAME_S
        if not 0.12 <= dur <= 0.8:
            continue
        if any(0 <= s - b <= int(0.3 / FRAME_S) for s in phrase_starts):
            breaths.append((a, b))
    n_phrases = max(1, len(_runs(p.voiced)))
    minutes = len(p.frame_db) * FRAME_S / 60
    rel = [float(np.mean(p.frame_db[a:b]) - speech_level) for a, b in breaths]
    durs = [(b - a) * FRAME_S for a, b in breaths]
    return {
        "breath_count": float(len(breaths)),
        "breath_per_min": len(breaths) / max(minutes, 1e-6),
        "breath_rel_db": float(np.mean(rel)) if rel else float("nan"),
        "breath_to_phrase_ratio": len(breaths) / n_phrases,
        "breath_duration_cv": _cv(durs),
    }


def rhythm_features(p: PitchTrack, speech: np.ndarray, min_pause_ms: float) -> dict[str, float]:
    min_pause = int(min_pause_ms / 1000 / FRAME_S)
    pauses = [(a, b) for a, b in _runs(~speech) if b - a >= min_pause and a > 0 and b < len(speech)]
    # Phrases: speech between qualifying pauses (short closures merged in).
    cuts = [0] + [x for a, b in pauses for x in (a, b)] + [len(speech)]
    phrases = [
        (cuts[i], cuts[i + 1]) for i in range(0, len(cuts) - 1, 2) if cuts[i + 1] - cuts[i] >= 10
    ]

    env = np.convolve(p.frame_db, np.ones(3) / 3, mode="same")
    rates = []
    for a, b in phrases:
        seg = env[a:b]
        peaks = (
            np.where(
                (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > seg.max() - 15)
            )[0]
            + 1
        )
        keep: list[int] = []
        for pk in peaks:  # syllable nuclei >= 80 ms apart
            if not keep or pk - keep[-1] >= 8:
                keep.append(int(pk))
        dur = (b - a) * FRAME_S
        if dur >= 0.3:
            rates.append(len(keep) / dur)

    pause_s = [(b - a) * FRAME_S for a, b in pauses]
    phrase_s = [(b - a) * FRAME_S for a, b in phrases]
    cvs = [c for c in (_cv(pause_s), _cv(phrase_s), _cv(rates)) if not np.isnan(c)]
    return {
        "pause_count": float(len(pauses)),
        "pause_mean_s": float(np.mean(pause_s)) if pause_s else float("nan"),
        "pause_cv": _cv(pause_s),
        "phrase_cv": _cv(phrase_s),
        "syllable_rate": float(np.mean(rates)) if rates else float("nan"),
        "syllable_rate_cv": _cv(rates),
        # 1 = metronomic (TTS-like), 0 = natural variability. NaN if nothing to measure.
        "over_regularity": float(np.clip(1 - np.mean(cvs) / 0.5, 0, 1)) if cvs else float("nan"),
    }


def filled_pause_features(p: PitchTrack) -> dict[str, float]:
    st = p.semitones
    count = 0
    for a, b in _runs(p.voiced):
        dur = (b - a) * FRAME_S
        if 0.25 <= dur <= 1.2 and np.std(st[a:b]) < 0.6 and np.std(p.frame_db[a:b]) < 2.5:
            count += 1
    minutes = len(p.frame_db) * FRAME_S / 60
    return {"filled_pause_count": float(count), "filled_pause_per_min": count / max(minutes, 1e-6)}
