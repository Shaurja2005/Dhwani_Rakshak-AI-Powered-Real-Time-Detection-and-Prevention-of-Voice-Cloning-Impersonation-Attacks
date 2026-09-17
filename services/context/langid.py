"""Language identification and code-switch detection (B10-T02).

Token-level, lightweight, deterministic:
* native scripts map to languages by Unicode block (Devanagari → hi, unless
  Marathi markers; Tamil, Telugu, Bengali/Assamese, Gujarati, Kannada,
  Malayalam, Gurmukhi, Odia, Arabic-script → ur);
* Latin-script tokens are split into English vs romanised Hindi (Hinglish) with
  a lexicon of high-frequency Hindi function/content words.

A segment is *code-switched* when two or more languages each cover at least
15 % of its word tokens (Indian call audio switches mid-sentence constantly).
ASR-reported language (e.g. Whisper's) can be passed in as a prior.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0600, 0x06FF, "ur"),
]
MARATHI_MARKERS = {"आहे", "नाही", "आणि", "मला", "तुम्ही", "काय", "आहेत", "झाले"}
ASSAMESE_CHARS = {"ৰ", "ৱ"}

HINGLISH = {
    "hai",
    "hain",
    "nahi",
    "nahin",
    "kya",
    "kyun",
    "aap",
    "aapka",
    "aapko",
    "mera",
    "meri",
    "mujhe",
    "hum",
    "haan",
    "ji",
    "jaldi",
    "abhi",
    "paisa",
    "paise",
    "bhejo",
    "bhej",
    "karo",
    "kar",
    "kijiye",
    "dijiye",
    "bataiye",
    "batao",
    "matlab",
    "yaar",
    "accha",
    "acha",
    "theek",
    "thik",
    "bolo",
    "boliye",
    "sir",
    "madam",
    "kisi",
    "ko",
    "mat",
    "bataana",
    "batana",
    "turant",
    "bank",
    "khata",
    "rupaye",
    "lakh",
    "crore",
    "wala",
    "wali",
    "se",
    "ka",
    "ki",
    "ke",
    "aur",
    "lekin",
    "toh",
    "bhi",
    "hoga",
    "hogi",
    "raha",
    "rahi",
    "rahe",
    "ho",
    "jayega",
    "jayegi",
    "jaayega",
    "hoon",
    "hun",
    "main",
    "mein",
    "tum",
    "tumhara",
    "apna",
    "apni",
    "karna",
    "karni",
    "kariye",
    "liye",
    "kuch",
    "sab",
    "yeh",
    "woh",
    "vo",
    "ye",
    "par",
    "pe",
    "tak",
    "gaya",
    "gayi",
    "diya",
    "liya",
    "dena",
    "lena",
    "chahiye",
    "sakte",
    "sakta",
    "sakti",
    "kaise",
    "kab",
    "kahan",
}
# Letters plus Indic / Arabic-script blocks: Python's \w does not match combining vowel
# signs (matras), which would otherwise split every Indic word into fragments.
_WORD = re.compile(r"(?:[^\W\d_]|[ऀ-෿؀-ۿ])+", re.UNICODE)


def token_language(token: str) -> str | None:
    t = unicodedata.normalize("NFC", token)
    for ch in t:
        cp = ord(ch)
        for lo, hi, lang in SCRIPT_RANGES:
            if lo <= cp <= hi:
                if lang == "hi" and t in MARATHI_MARKERS:
                    return "mr"
                if lang == "bn" and any(c in ASSAMESE_CHARS for c in t):
                    return "as"
                return lang
    if t.isascii() and t.isalpha():
        return "hi-Latn" if t.lower() in HINGLISH else "en"
    return None


@dataclass
class LanguageResult:
    language: str  # dominant language code, e.g. "hi", "en"
    shares: dict[str, float]
    code_switched: bool
    switches: int


def detect(text: str, asr_language: str | None = None, min_share: float = 0.15) -> LanguageResult:
    tokens = [token_language(w) for w in _WORD.findall(text)]
    tokens = [t for t in tokens if t]
    # Hindi and Marathi share Devanagari: if the segment carries Marathi markers, its
    # unmarked Devanagari words are Marathi too (resolved per segment, not per word).
    if "mr" in tokens:
        tokens = ["mr" if t == "hi" else t for t in tokens]
    if not tokens:
        lang = (asr_language or "unknown").split("-")[0]
        return LanguageResult(lang, {}, False, 0)
    # Romanised Hindi counts as Hindi for the dominant language, but is kept distinct for switching.
    counts = Counter(tokens)
    shares = {k: v / len(tokens) for k, v in counts.items()}
    base = Counter()
    for k, v in counts.items():
        base[k.split("-")[0]] += v
    dominant = base.most_common(1)[0][0]
    # Hinglish: Hindi is the matrix language (grammar) with English content words inserted,
    # so a substantial share of romanised Hindi function words makes Hindi dominant.
    if dominant == "en" and counts.get("hi-Latn", 0) / len(tokens) >= 0.3:
        dominant = "hi"
    if asr_language and base.get(asr_language.split("-")[0], 0) == base[dominant]:
        dominant = asr_language.split("-")[0]
    switches = sum(
        1 for a, b in zip(tokens, tokens[1:], strict=False) if a.split("-")[0] != b.split("-")[0]
    )
    langs_over = [k for k, v in base.items() if v / len(tokens) >= min_share]
    return LanguageResult(dominant, shares, len(langs_over) >= 2, switches)
