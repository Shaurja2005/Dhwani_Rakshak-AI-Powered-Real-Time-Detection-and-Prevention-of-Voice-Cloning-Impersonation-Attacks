"""PII redaction on transcripts (B10-T07).

Applied *before* a transcript is stored, logged, shown, or sent to the intent
LLM. Covers what shows up in Indian banking calls:

card numbers (Luhn-checked) · bank account numbers · OTP / PIN / CVV (digits
near a trigger word, incl. Hindi) · IFSC · PAN · Aadhaar · phone numbers ·
email · UPI IDs · digit strings *spoken as words* (English and Hindi number
words), which ASR often produces instead of numerals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER_WORDS = {
    "zero",
    "oh",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "double",
    "triple",
    "shunya",
    "ek",
    "do",
    "teen",
    "char",
    "chaar",
    "paanch",
    "panch",
    "chhe",
    "che",
    "saat",
    "aath",
    "nau",
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पांच",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
}
OTP_TRIGGERS = r"(?:otp|o\.t\.p|pin|cvv|cvc|passcode|password|code|ओटीपी|पिन|कोड)"


@dataclass
class Redaction:
    kind: str
    start: int
    end: int


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    (
        "upi_id",
        re.compile(r"\b[\w.-]{2,}@(?:ok\w+|ybl|paytm|upi|apl|ibl|axl|icici|sbi|hdfcbank)\b", re.I),
    ),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("otp", re.compile(OTP_TRIGGERS + r"[^\d\n]{0,20}(\d[\d\s-]{2,9}\d)", re.I)),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("aadhaar", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("phone", re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")),
    ("account_number", re.compile(r"\b\d{9,18}\b")),
]


def redact(text: str) -> tuple[str, list[Redaction]]:
    spans: list[Redaction] = []

    def taken(a: int, b: int) -> bool:
        return any(not (b <= s.start or a >= s.end) for s in spans)

    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            a, b = (m.start(1), m.end(1)) if kind == "otp" else (m.start(), m.end())
            if taken(a, b):
                continue
            if kind == "card_number" and not _luhn(re.sub(r"\D", "", m.group())):
                continue
            spans.append(Redaction(kind, a, b))

    # Digit strings spoken as words: 4+ consecutive number words.
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[^\s,.;:!?]+", text)]
    run: list[tuple[str, int, int]] = []
    for w in [*words, ("", len(text), len(text))]:
        if w[0].lower() in NUMBER_WORDS or (w[0].isdigit() and len(w[0]) == 1):
            run.append(w)
            continue
        if len(run) >= 4 and not taken(run[0][1], run[-1][2]):
            spans.append(Redaction("spoken_digits", run[0][1], run[-1][2]))
        run = []

    out = text
    for s in sorted(spans, key=lambda s: -s.start):
        out = out[: s.start] + f"[{s.kind.upper()}]" + out[s.end :]
    return out, sorted(spans, key=lambda s: s.start)
