"""Lexical disfluency detection from the ASR stream (B6-T04).

Consumes timestamped tokens produced by B10's streaming ASR. Until B10 exists
the head passes no tokens and these features are NaN (the acoustic filled-pause
detector in breath_rhythm.py still works).

Counts: fillers (per-language lexicon), immediate word repetitions, and
restarts (truncated words marked by the ASR with a trailing hyphen, or a
repeated 1–2 word prefix followed by a correction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FILLERS: dict[str, set[str]] = {
    "en": {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "like", "you know", "i mean"},
    "hi": {
        "अं",
        "अम्म",
        "हम्म",
        "मतलब",
        "वो",
        "यानी",
        "तो",
        "हाँ",
        "matlab",
        "yaani",
        "woh",
        "haan",
        "hmm",
        "um",
    },
    "ta": {"அது", "வந்து", "அப்புறம்", "ம்ம்", "vandhu", "adhu", "appuram", "hmm"},
    "te": {"అంటే", "అది", "ఏంటంటే", "ante", "adi", "hmm"},
    "bn": {"মানে", "ইয়ে", "mane", "iye", "hmm"},
    "mr": {"म्हणजे", "ते", "mhanje", "hmm"},
}
_GENERIC = {"um", "umm", "uh", "uhh", "hmm", "mm", "er", "ah"}
_WORD = re.compile(r"[\wऀ-෿'-]+", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start_ms: int
    end_ms: int


def disfluency_features(tokens: list[Token] | None, language: str | None) -> dict[str, float]:
    nan = float("nan")
    if not tokens:
        return {
            "filler_per_min": nan,
            "repetition_per_min": nan,
            "restart_per_min": nan,
            "disfluency_per_min": nan,
        }
    lexicon = FILLERS.get((language or "en").split("-")[0], set()) | _GENERIC | FILLERS["en"]
    words = [w.lower() for t in tokens for w in _WORD.findall(t.text)]
    joined = " ".join(words)
    fillers = sum(1 for w in words if w in lexicon)
    fillers += sum(joined.count(p) for p in lexicon if " " in p)
    repetitions = sum(
        1 for a, b in zip(words, words[1:], strict=False) if a == b and a not in lexicon
    )
    restarts = sum(1 for w in words if w.endswith("-") and len(w) > 1)
    # "I went- I go": a word recurs two positions later with a different continuation.
    restarts += sum(
        1
        for i in range(len(words) - 3)
        if words[i] == words[i + 2] and words[i + 1] != words[i + 3] and words[i] not in lexicon
    )
    minutes = max((tokens[-1].end_ms - tokens[0].start_ms) / 60000, 1e-6)
    return {
        "filler_per_min": fillers / minutes,
        "repetition_per_min": repetitions / minutes,
        "restart_per_min": restarts / minutes,
        "disfluency_per_min": (fillers + repetitions + restarts) / minutes,
    }
