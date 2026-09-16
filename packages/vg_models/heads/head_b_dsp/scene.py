"""Room impulse response / scene consistency (B5-T04).

A cloned voice is usually generated dry (anechoic) and then mixed over a
recorded, reverberant background. A real speaker and the sounds around them
share one room, so their reverberation tails should agree.

Method, on 20 ms frames with a 10 ms hop:
1. Detect onsets (>= 9 dB rise within 20 ms). Each onset starts an event that
   lasts until the next onset. Classify by the onset: **voice** (harmonic, low
   spectral flatness, sustained >= 100 ms) or **background transient**
   (noise-like, <= 60 ms).
2. For each event, fit the late dB decay (10 dB below the source's last loud
   frame, down to 5 dB above the noise floor, never past the next onset)
   -> RT60 = 60 / slope.
3. Median RT60 per event type. If the voice is much drier than the background,
   the scene is inconsistent.

Survives codecs much better than high-frequency artifacts, because it only
uses frame energy envelopes. Needs at least 2 events of each type; otherwise
the result is ``unknown`` and contributes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.vg_models.heads.head_b_dsp.features import _FREQS, EPS, HOP, SR, Spectra

MIN_EVENTS = 2
_TEL_BAND = (_FREQS >= 100) & (_FREQS <= 3800)
FRAME_S = HOP / SR


@dataclass
class SceneConsistency:
    rt60_voice_s: float | None
    rt60_background_s: float | None
    n_voice_events: int
    n_background_events: int
    inconsistency: float  # 0 = consistent / unknown, 1 = dry voice over reverberant background

    @property
    def known(self) -> bool:
        return self.rt60_voice_s is not None and self.rt60_background_s is not None

    def as_features(self) -> dict[str, float]:
        return {
            "scene_rt60_voice": self.rt60_voice_s if self.rt60_voice_s is not None else -1.0,
            "scene_rt60_background": (
                self.rt60_background_s if self.rt60_background_s is not None else -1.0
            ),
            "scene_known": float(self.known),
            "scene_inconsistency": self.inconsistency,
        }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


def _decay_rt60(db: np.ndarray, peak: int, limit: int, floor: float) -> float | None:
    """Late-decay RT60 (T20-style): fit from 10 dB below the peak down to floor + 5 dB.

    Skipping the first 10 dB ignores how the source itself tapers off (speech
    offsets fade gradually), leaving the room's reverberant tail. ``limit`` is
    the next event onset: the fit never crosses into another event.
    """
    stop = floor + 5.0
    if db[peak] - stop < 20.0:  # need 10 dB skip + >=10 dB of fit range
        return None
    start = peak
    while start + 1 < limit and db[start] > db[peak] - 10.0:
        start += 1
    if db[start] > db[peak] - 10.0:
        return None  # next event arrived before the tail decayed 10 dB
    end = start
    while end + 1 < limit and db[end + 1] > stop:
        end += 1
    if end - start < 1:
        return 0.02  # dropped through the fit range within one hop: anechoic
    if db[end] > stop + 5.0 and end + 1 >= limit:
        return None  # truncated by the next event: slope would be unreliable
    t = np.arange(end - start + 1) * FRAME_S
    slope = np.polyfit(t, db[start : end + 1], 1)[0]  # dB/s, negative
    if slope >= -1.0:
        return None
    return float(np.clip(60.0 / -slope, 0.02, 3.0))


def _onsets(db: np.ndarray, floor: float) -> np.ndarray:
    """Frames where energy jumps >= 9 dB within 20 ms up to a loud level."""
    rise = db[2:] - db[:-2]
    cand = np.where((rise >= 9.0) & (db[2:] > floor + 15.0))[0] + 2
    keep: list[int] = []
    for c in cand:
        if not keep or c - keep[-1] > int(0.05 / FRAME_S):
            keep.append(int(c))
    return np.array(keep, dtype=int)


def analyse(s: Spectra) -> SceneConsistency:
    db = s.frame_db
    floor = float(np.percentile(db, 10))
    # Flatness over the telephone band only, so narrowband codecs (empty 4-8 kHz)
    # don't make noise transients look tonal.
    band = s.power[:, _TEL_BAND]
    flatness = np.exp(np.log(band).mean(axis=1)) / (band.mean(axis=1) + EPS)
    onsets = _onsets(db, floor)

    voice_rt, bg_rt = [], []
    for i, on in enumerate(onsets):
        nxt = int(onsets[i + 1]) if i + 1 < len(onsets) else len(db)
        seg = db[on:nxt]
        loud = np.where(seg >= seg.max() - 6.0)[0]
        # Source is "on" while within 6 dB of its max; decay begins after the last such frame.
        peak = on + int(loud[-1])
        active_s = (loud[-1] - loud[0] + 1) * FRAME_S
        onset_flat = float(np.median(flatness[on : min(nxt, on + int(0.04 / FRAME_S))]))
        rt = _decay_rt60(db, peak, nxt, floor)
        if rt is None:
            continue
        if onset_flat < 0.2 and active_s >= 0.1:
            voice_rt.append(rt)
        elif onset_flat >= 0.2 and active_s <= 0.06:
            bg_rt.append(rt)

    rt_v = float(np.median(voice_rt)) if len(voice_rt) >= MIN_EVENTS else None
    rt_b = float(np.median(bg_rt)) if len(bg_rt) >= MIN_EVENTS else None
    inconsistency = 0.0
    if rt_v is not None and rt_b is not None and rt_b > rt_v:
        # ratio 2x -> 0, 8x -> 1
        inconsistency = float(np.clip((np.log(rt_b / rt_v) - np.log(2)) / np.log(4), 0, 1))
    return SceneConsistency(rt_v, rt_b, len(voice_rt), len(bg_rt), inconsistency)
