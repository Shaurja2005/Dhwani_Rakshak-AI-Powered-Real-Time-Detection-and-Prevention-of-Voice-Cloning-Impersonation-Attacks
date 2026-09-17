"""Labeled seed set of scripted call scenarios for the intent classifier (B10-T03).

Deterministically generates fraud and benign call transcripts in English,
Hinglish and Hindi from templates and slot values, with gold labels.

Uses: few-shot / regression set for the local LLM prompt, and a smoke check for
the rule classifier. It is written by the same people who wrote the rules, so a
high rule score on it is *not* evidence of real-world accuracy — B15 must
evaluate on transcripts of real (consented) calls.

    python -m services.context.seed --out services/context/data/intent_seed.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

ROLES = ["CFO", "CEO", "managing director", "director", "COO", "chairman"]
AMOUNTS = ["18 lakh", "4.5 crore", "9,80,000 rupees", "25 lakh", "7 lakh", "1.2 crore"]
DEADLINES = ["right now", "immediately", "within 10 minutes", "before the market closes"]
AUTHORITIES = ["RBI", "income tax department", "cyber cell police", "CBI"]

FRAUD_TEMPLATES: list[tuple[str, str, list[str]]] = [
    (
        "en",
        "This is the {role}. I need you to wire {amount} to the new vendor account {deadline}, and don't tell the team.",
        [
            "authority_pressure",
            "payment_request",
            "unusual_beneficiary",
            "manufactured_urgency",
            "secrecy_demand",
        ],
    ),
    (
        "en",
        "Hi, it's the {role}. I'm in a board meeting so I can't take a call back. Transfer {amount} {deadline}.",
        ["authority_pressure", "callback_resistance", "payment_request", "manufactured_urgency"],
    ),
    (
        "en",
        "Calling from the {authority}. Your account will be frozen {deadline} unless you read me the OTP we just sent.",
        ["authority_pressure", "manufactured_urgency", "credential_request"],
    ),
    (
        "en",
        "The vendor has updated bank details, please pay {amount} to this account and keep it confidential.",
        ["unusual_beneficiary", "payment_request", "secrecy_demand"],
    ),
    (
        "en",
        "Message me on my personal WhatsApp, don't call me back on the office line, and send {amount} {deadline}.",
        ["unusual_channel", "callback_resistance", "payment_request", "manufactured_urgency"],
    ),
    (
        "en",
        "Install AnyDesk so I can fix your account, then tell me the OTP.",
        ["unusual_channel", "credential_request"],
    ),
    (
        "hi-Latn",
        "Sir main {role} bol raha hoon, {amount} naya account mein transfer karo, jaldi, kisi ko mat batana.",
        [
            "authority_pressure",
            "payment_request",
            "unusual_beneficiary",
            "manufactured_urgency",
            "secrecy_demand",
        ],
    ),
    (
        "hi-Latn",
        "{authority} se call hai, aapka account block ho jayega, turant OTP batao.",
        ["authority_pressure", "manufactured_urgency", "credential_request"],
    ),
    (
        "hi-Latn",
        "Call back mat karo, main meeting mein hoon, {amount} abhi ke abhi bhej do.",
        ["callback_resistance", "manufactured_urgency", "payment_request"],
    ),
    (
        "hi",
        "मैं {authority} से बोल रहा हूँ, तुरंत ओटीपी बताइए, किसी को मत बताना।",
        ["authority_pressure", "manufactured_urgency", "credential_request", "secrecy_demand"],
    ),
    (
        "hi",
        "नया खाता है, पैसे भेजो, जल्दी करो।",
        ["unusual_beneficiary", "payment_request", "manufactured_urgency"],
    ),
]

BENIGN_TEMPLATES: list[tuple[str, str]] = [
    ("en", "Hi, I'd like to check my account balance and whether my cheque has cleared."),
    ("en", "Can you tell me the branch timings on Saturday?"),
    ("en", "I want to update my address; I moved last month."),
    ("en", "My debit card is not working at the ATM, can you help me?"),
    ("en", "Please call me back on my registered number when the loan statement is ready."),
    ("en", "I'd like to know the interest rate on a fixed deposit for one year."),
    ("hi-Latn", "Namaste, mujhe apna balance check karna hai."),
    ("hi-Latn", "Mera credit card ka bill kitna hai, bata dijiye."),
    ("hi-Latn", "Branch kab khulti hai Sunday ko?"),
    ("hi", "मुझे अपना पता बदलना है।"),
    ("hi", "मेरा एटीएम कार्ड काम नहीं कर रहा है।"),
    ("en", "I got an SMS about a transaction I don't recognise, can you block my card?"),
    (
        "en",
        "Our CFO asked me to confirm the payment run through the usual approval workflow tomorrow.",
    ),
    ("hi-Latn", "Mujhe apne loan ki EMI date change karni hai."),
    ("hi-Latn", "Kal branch aa kar KYC update kar dunga, time bata dijiye."),
]


def generate(seed: int = 7) -> list[dict[str, object]]:
    rng = random.Random(seed)  # noqa: S311 - reproducible test data
    rows: list[dict[str, object]] = []
    for lang, tpl, labels in FRAUD_TEMPLATES:
        combos = list(itertools.product(ROLES, AMOUNTS, DEADLINES, AUTHORITIES))
        rng.shuffle(combos)
        for role, amount, deadline, authority in combos[:22]:
            text = tpl.format(role=role, amount=amount, deadline=deadline, authority=authority)
            if text not in {r["text"] for r in rows}:
                rows.append(
                    {"text": text, "language": lang, "fraud": True, "labels": sorted(labels)}
                )
    for lang, text in BENIGN_TEMPLATES:
        suffixes = [
            "",
            " Thank you.",
            " Thanks a lot.",
            " Dhanyavaad.",
            " Please help.",
            " Sorry to bother you.",
        ]
        for suffix in suffixes[: 6 if lang != "hi" else 3]:
            rows.append({"text": text + suffix, "language": lang, "fraud": False, "labels": []})
    rng.shuffle(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("services/context/data/intent_seed.jsonl"))
    args = ap.parse_args(argv)
    rows = generate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} scenarios -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
