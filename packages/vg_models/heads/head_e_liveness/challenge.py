"""Liveness challenges (B8-T01) and the per-session challenge registry.

Challenges are generated with ``secrets`` (unpredictable to an attacker) and
target what real-time cloning pipelines are bad at: content they could not
pre-generate, sudden code-switching, and an unusual word order — all of which
add pipeline latency or break the voice converter / TTS front-end.

Kinds:
* ``digits_codeswitch`` — e.g. "4 सात 9 two" (digits alternating between languages)
* ``nonce_phrase``      — two or three unrelated common words
* ``reverse_order``     — "say these words in reverse order: blue, mango, seven"

Each expected token carries every acceptable surface form (digit, English word,
native-script word, romanised word), so ASR output in any of them matches.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["digits_codeswitch", "nonce_phrase", "reverse_order"]

DIGITS: dict[str, dict[int, tuple[str, ...]]] = {
    "en": {
        0: ("zero",),
        1: ("one",),
        2: ("two",),
        3: ("three",),
        4: ("four",),
        5: ("five",),
        6: ("six",),
        7: ("seven",),
        8: ("eight",),
        9: ("nine",),
    },
    "hi": {
        0: ("शून्य", "shunya"),
        1: ("एक", "ek"),
        2: ("दो", "do"),
        3: ("तीन", "teen"),
        4: ("चार", "char"),
        5: ("पांच", "पाँच", "paanch"),
        6: ("छह", "छः", "chhah", "che"),
        7: ("सात", "saat"),
        8: ("आठ", "aath"),
        9: ("नौ", "nau"),
    },
    "ta": {
        0: ("பூஜ்யம்", "poojyam"),
        1: ("ஒன்று", "ondru"),
        2: ("இரண்டு", "irandu"),
        3: ("மூன்று", "moondru"),
        4: ("நான்கு", "naangu"),
        5: ("ஐந்து", "ainthu"),
        6: ("ஆறு", "aaru"),
        7: ("ஏழு", "ezhu"),
        8: ("எட்டு", "ettu"),
        9: ("ஒன்பது", "onbathu"),
    },
}

# Short, common, easily pronounced words; each entry lists acceptable forms.
WORDS: dict[str, list[tuple[str, ...]]] = {
    "en": [
        ("blue",),
        ("mango",),
        ("river",),
        ("pencil",),
        ("tiger",),
        ("window",),
        ("orange",),
        ("garden",),
        ("rocket",),
        ("candle",),
        ("silver",),
        ("monkey",),
    ],
    "hi": [
        ("नीला", "neela"),
        ("आम", "aam"),
        ("नदी", "nadi"),
        ("किताब", "kitaab"),
        ("शेर", "sher"),
        ("खिड़की", "khidki"),
        ("बादल", "baadal"),
        ("चाबी", "chaabi"),
        ("पत्ता", "patta"),
        ("दीया", "diya"),
    ],
}


@dataclass(frozen=True)
class ExpectedToken:
    forms: tuple[str, ...]
    display: str


@dataclass
class Challenge:
    challenge_id: str
    session_id: str
    kind: Kind
    prompt_text: str  # what the agent says / the IVR plays
    expected: list[ExpectedToken]
    languages: tuple[str, ...]
    issued_at: float = field(default_factory=time.time)
    ttl_s: float = 30.0
    prompt_end_ms: int | None = None  # call-relative time the prompt finished playing / being read

    @property
    def expired(self) -> bool:
        return time.time() > self.issued_at + self.ttl_s


def _pick(seq: list[tuple[str, ...]] | tuple[str, ...], k: int) -> list:  # type: ignore[type-arg]
    pool = list(seq)
    return [pool.pop(secrets.randbelow(len(pool))) for _ in range(k)]


def generate(
    session_id: str,
    kind: Kind | None = None,
    languages: tuple[str, ...] = ("en", "hi"),
    length: int = 4,
) -> Challenge:
    langs = tuple(lg for lg in languages if lg in DIGITS) or ("en", "hi")
    kind = kind or ("digits_codeswitch", "nonce_phrase", "reverse_order")[secrets.randbelow(3)]  # type: ignore[assignment]
    cid = str(uuid.uuid4())
    if kind == "digits_codeswitch":
        expected = []
        for i in range(length):
            d = secrets.randbelow(10)
            lang = langs[i % len(langs)] if len(langs) > 1 else langs[0]
            forms = (str(d), *DIGITS["en"][d], *DIGITS.get(lang, DIGITS["en"])[d])
            display = DIGITS[lang][d][0] if lang != "en" and secrets.randbelow(2) else str(d)
            expected.append(ExpectedToken(tuple(dict.fromkeys(forms)), display))
        prompt = "Please repeat after me: " + " ".join(t.display for t in expected)
        return Challenge(cid, session_id, "digits_codeswitch", prompt, expected, langs)

    word_langs = [lg for lg in langs if lg in WORDS] or ["en"]
    chosen = [
        ExpectedToken(tuple(forms), forms[0])
        for forms in _pick([w for lg in word_langs for w in WORDS[lg]], 3)
    ]
    if kind == "nonce_phrase":
        prompt = "Please say: " + " ".join(t.display for t in chosen)
        return Challenge(cid, session_id, "nonce_phrase", prompt, chosen, langs)
    prompt = "Please say these words in reverse order: " + ", ".join(t.display for t in chosen)
    return Challenge(cid, session_id, "reverse_order", prompt, list(reversed(chosen)), langs)


@dataclass
class ResponseRecord:
    tokens: list[tuple[str, int, int]]  # (text, start_ms, end_ms), call-relative
    received_at: float = field(default_factory=time.time)


class ChallengeRegistry:
    """Per-session active challenge + its response. One active challenge per session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._challenges: dict[str, Challenge] = {}
        self._responses: dict[str, ResponseRecord] = {}

    def issue(self, challenge: Challenge) -> Challenge:
        with self._lock:
            old = self._challenges.get(challenge.session_id)
            if old is not None:
                self._responses.pop(old.challenge_id, None)
            self._challenges[challenge.session_id] = challenge
        return challenge

    def mark_prompt_end(self, session_id: str, prompt_end_ms: int) -> None:
        with self._lock:
            ch = self._challenges.get(session_id)
            if ch is not None:
                ch.prompt_end_ms = prompt_end_ms

    def active(self, session_id: str) -> Challenge | None:
        with self._lock:
            return self._challenges.get(session_id)

    def submit_response(
        self, session_id: str, challenge_id: str, tokens: list[tuple[str, int, int]]
    ) -> bool:
        with self._lock:
            ch = self._challenges.get(session_id)
            if ch is None or ch.challenge_id != challenge_id:
                return False
            self._responses[challenge_id] = ResponseRecord(tokens)
            return True

    def response(self, challenge_id: str) -> ResponseRecord | None:
        with self._lock:
            return self._responses.get(challenge_id)

    def end_session(self, session_id: str) -> None:
        with self._lock:
            ch = self._challenges.pop(session_id, None)
            if ch is not None:
                self._responses.pop(ch.challenge_id, None)
