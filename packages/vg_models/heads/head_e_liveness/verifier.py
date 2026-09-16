"""Challenge response verification (B8-T02).

Inputs: the challenge and the ASR tokens of the caller's answer (from B10, or
entered by the agent). Three signals:

1. **Content match** — ordered alignment of expected tokens (any accepted form)
   against the transcript: fraction of expected tokens found in order.
2. **Response latency** — time from the end of the prompt to the first answer
   token. Humans repeating a short prompt typically start within ~0.3–2.0 s;
   a real-time clone pipeline (ASR → text → TTS/VC) adds processing delay, and
   an instant (< 150 ms) start suggests audio that was not a reaction at all.
   Latency is often the strongest single tell.
3. **Delivery plausibility** — speaking rate and the regularity of gaps between
   tokens (TTS reads digit strings with metronomic spacing).

Thresholds are initial heuristics (``HEURISTIC_VERSION``), to be tuned in
shadow mode (B11-T07) on real calls.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from packages.vg_models.heads.head_e_liveness.challenge import Challenge

HEURISTIC_VERSION = "heuristic-uncalibrated"
_TOKEN = re.compile(r"[\wऀ-෿]+", re.UNICODE)


def normalise(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return _TOKEN.findall(text)


@dataclass
class Verification:
    content_match: float  # 0..1
    latency_ms: int | None
    rate_tokens_per_s: float | None
    gap_cv: float | None
    p_spoof: float
    reasons: list[str]


def _match(challenge: Challenge, words: list[str]) -> float:
    """Longest ordered subsequence of expected tokens found in the transcript, as a fraction."""
    exp = [
        set(normalise(" ".join(t.forms))) | {f.lower() for f in t.forms} for t in challenge.expected
    ]
    n, m = len(exp), len(words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = max(
                dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + (words[j - 1] in exp[i - 1])
            )
    return float(dp[n, m] / max(n, 1))


def verify(challenge: Challenge, tokens: list[tuple[str, int, int]]) -> Verification:
    reasons: list[str] = []
    words: list[str] = []
    timed: list[tuple[int, int]] = []
    for text, start, end in tokens:
        for w in normalise(text):
            words.append(w)
            timed.append((start, end))
    match = _match(challenge, words) if words else 0.0

    latency = None
    if timed and challenge.prompt_end_ms is not None:
        latency = int(timed[0][0] - challenge.prompt_end_ms)

    rate = gap_cv = None
    if len(timed) >= 3:
        dur_s = (timed[-1][1] - timed[0][0]) / 1000
        rate = len(timed) / dur_s if dur_s > 0 else None
        gaps = np.diff([s for s, _ in timed])
        gap_cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else None

    # Evidence as log-odds contributions (spoof positive). Starts neutral.
    z = 0.0
    if match < 0.5:
        z += 4.0  # failing an unpredictable challenge outright is strong evidence
        reasons.append(f"Caller did not repeat the challenge correctly ({match:.0%} matched).")
    elif match < 1.0:
        z += 1.0
        reasons.append(f"Challenge only partly repeated ({match:.0%} matched).")
    else:
        z -= 1.0
    if latency is not None:
        if latency < 150:
            z += 3.0  # nobody can start repeating unpredictable content before hearing it
            reasons.append(
                f"Answer started {latency} ms after the prompt — too fast to be a reaction."
            )
        elif latency > 3500:
            # Weighted to outweigh a correct answer: latency is the strongest single tell.
            z += 3.5
            reasons.append(
                reasons.append(
                    f"Answer started {latency / 1000:.1f} s after the prompt — "
                    "consistent with a voice-cloning pipeline delay."
                )
            )
        elif latency > 2000:
            z += 1.0
            reasons.append(f"Slow start to the answer ({latency / 1000:.1f} s).")
        else:
            z -= 1.0
    if rate is not None and not 0.8 <= rate <= 6.0:
        z += 1.0
        reasons.append(f"Implausible speaking rate for the answer ({rate:.1f} words/s).")
    if gap_cv is not None and gap_cv < 0.08:
        z += 1.0
        reasons.append("Words in the answer were evenly spaced like synthetic speech.")
    p = 1.0 / (1.0 + math.exp(-(z - 1.0)))
    if not reasons:
        reasons.append("Challenge repeated correctly with natural timing.")
    return Verification(match, latency, rate, gap_cv, p, reasons)
