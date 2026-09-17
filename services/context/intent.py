"""Social-engineering intent classification over the rolling transcript (B10-T03).

Catches attacks where the voice is a *real human* reading a fraud script — a
class the acoustic heads cannot see.

Labels (the coercion script):
    manufactured_urgency · secrecy_demand · authority_pressure · unusual_channel ·
    callback_resistance · unusual_beneficiary · credential_request · payment_request

Two classifiers, combined:
* ``RuleIntentClassifier`` — multilingual (English, Hindi, Hinglish) patterns;
  microseconds, always on, deterministic, explainable.
* ``LLMIntentClassifier`` — a *local* LLM (Ollama / vLLM OpenAI-compatible API)
  with few-shot prompting and strict JSON output. The base URL must resolve to
  the deployment boundary (loopback / private network / cluster DNS) unless the
  tenant explicitly opted in to an external endpoint (invariant I6). It runs
  asynchronously with a timeout; on timeout or error the rule result stands (I10).

Only *redacted* transcripts are ever passed in (B10-T07).
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

LABELS = (
    "manufactured_urgency",
    "secrecy_demand",
    "authority_pressure",
    "unusual_channel",
    "callback_resistance",
    "unusual_beneficiary",
    "credential_request",
    "payment_request",
)
# Contribution of each label to intent_risk (noisy-OR). Credential requests and secrecy are the strongest tells.
LABEL_WEIGHT = {
    "manufactured_urgency": 0.35,
    "secrecy_demand": 0.55,
    "authority_pressure": 0.35,
    "unusual_channel": 0.3,
    "callback_resistance": 0.5,
    "unusual_beneficiary": 0.45,
    "credential_request": 0.6,
    "payment_request": 0.3,
}


@dataclass
class IntentHit:
    label: str
    confidence: float
    evidence: str  # the (redacted) phrase that triggered it


@dataclass
class IntentResult:
    hits: list[IntentHit] = field(default_factory=list)
    source: str = "rules"

    @property
    def labels(self) -> list[str]:
        return sorted({h.label for h in self.hits})

    @property
    def risk(self) -> float:
        best: dict[str, float] = {}
        for h in self.hits:
            best[h.label] = max(best.get(h.label, 0.0), h.confidence)
        p_clean = 1.0
        for label, conf in best.items():
            p_clean *= 1 - LABEL_WEIGHT.get(label, 0.3) * conf
        # Co-occurring labels *are* the coercion script, which is far stronger evidence than
        # independent cues: three or more distinct labels halve the remaining doubt.
        if len(best) >= 3:
            p_clean *= 0.5
        return round(1 - p_clean, 4)


class IntentClassifier(Protocol):
    def classify(self, transcript: str, language: str | None = None) -> IntentResult: ...


# ---------------------------------------------------------------- rules
_R = dict[str, list[str]]
RULES: _R = {
    "manufactured_urgency": [
        r"\b(right now|immediately|urgent(ly)?|as soon as possible|asap|within (the next )?\d+ (minutes|mins|hours)|before (the )?(end of (the )?day|market closes?|\d))\b",
        r"\b(jaldi|turant|abhi ke abhi|foran|fauran)\b",
        r"(तुरंत|जल्दी|अभी के अभी|फौरन)",
        r"\b(account will be (blocked|frozen|suspended)|last chance|deal will fall through)\b",
    ],
    "secrecy_demand": [
        r"\b(don'?t|do not|never) (tell|inform|mention|discuss|share)( this)? (with |to )?(anyone|anybody|the team|your (manager|colleagues|family|bank))\b",
        r"\b(keep (this|it) (confidential|secret|between us|quiet))\b",
        r"\b(kisi ko (bhi )?(mat|na) (batana|bataiye|bolna|batao))\b",
        r"(किसी को (भी )?(मत|ना) (बताना|बताइए|बोलना))",
    ],
    "authority_pressure": [
        r"\b(this is|it'?s|i'?m|i am) (the |your )?(ceo|cfo|coo|md|managing director|director|chairman)\b",
        r"\bmain (ceo|cfo|coo|md|managing director|director|chairman) (bol raha|bol rahi|baat kar raha)",
        r"\b(rbi|reserve bank|income tax (department|officer)|cbi|police|cyber cell|customs|enforcement directorate)\b",
        r"\b(on (my|the (board'?s|ceo'?s)) (authority|instructions|orders)|i'?m ordering you)\b",
        r"(आरबीआई|पुलिस|सीबीआई|इनकम टैक्स)",
    ],
    "unusual_channel": [
        r"\b(whatsapp|telegram|signal|personal (number|phone|email)|gmail|my other number)\b",
        r"\b(anydesk|teamviewer|quick ?support|screen ?shar(e|ing))\b",
    ],
    "callback_resistance": [
        r"\b(don'?t|do not|no need to) (call|ring) (me )?back\b",
        r"\b(i can'?t|cannot|won'?t be able to) (take|receive) (a |your )?call(s)?\b",
        r"\b(no time (to|for) (verify|verification|call ?back))\b",
        r"\b(call ?back mat (karo|kijiye|karna))\b",
    ],
    "unusual_beneficiary": [
        r"\b(new (vendor|beneficiary|supplier|account)|different account|updated (bank|account) details)\b",
        r"\b(transfer (it )?to (this|a new|another) account)\b",
        r"\b(naya (account|khata))\b",
        r"(नया (खाता|अकाउंट))",
    ],
    "credential_request": [
        r"\b(tell|give|share|read( out)?|send) (me )?(the |your )?(\[OTP\]|otp|pin|cvv|password|passcode|verification code)\b",
        r"\b(otp|pin|cvv) (batao|bataiye|bhejo|dijiye|share karo)\b",
        r"(ओटीपी|पिन) (बताइए|बताओ|भेजो)",
        r"\[OTP\]",
    ],
    "payment_request": [
        r"\b(wire|transfer|send|remit|pay) (the )?(money|funds|amount|payment|\d[\d,]*|rs\.?|inr|₹|lakh|crore)",
        r"\b(paise|paisa|rupaye|amount) (bhejo|bhej do|transfer karo|daal do)\b",
        r"\b\d[\d.,]*\s*(lakh|crore|rupees|rs|inr)\b.{0,40}\b(transfer|bhej|send|pay|wire)",
        r"(पैसे|रुपये) (भेजो|भेज दो|ट्रांसफर)",
        r"\b(gift cards?|crypto|bitcoin|usdt)\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in RULES.items()}


class RuleIntentClassifier:
    def classify(self, transcript: str, language: str | None = None) -> IntentResult:
        hits = []
        for label, pats in _COMPILED.items():
            for pat in pats:
                m = pat.search(transcript)
                if m:
                    hits.append(IntentHit(label, 0.8, m.group(0)))
                    break
        return IntentResult(hits, "rules")


# ---------------------------------------------------------------- local LLM
class NonLocalEndpointError(RuntimeError):  # noqa: N818
    pass


def is_local_endpoint(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(
        (".local", ".svc", ".svc.cluster.local", ".internal")
    ):
        return True
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except OSError:
        return False
    return bool(addrs) and all(
        ipaddress.ip_address(a).is_private or ipaddress.ip_address(a).is_loopback for a in addrs
    )


SYSTEM_PROMPT = """You detect social-engineering fraud in bank / enterprise phone-call transcripts.
Transcripts may mix English, Hindi (Devanagari) and romanised Hindi. Personal data is already redacted.
Return ONLY JSON: {"labels": [{"label": <one of LABELS>, "confidence": <0..1>, "evidence": "<short quote>"}]}
Use an empty list when the call is ordinary. Do not invent quotes.""".replace("LABELS", ", ".join(LABELS))

FEW_SHOT = [
    (
        "This is the CFO. I need you to wire 18 lakh to the new vendor account right now, and don't tell the team.",
        {
            "labels": [
                {"label": "authority_pressure", "confidence": 0.9, "evidence": "This is the CFO"},
                {"label": "payment_request", "confidence": 0.9, "evidence": "wire 18 lakh"},
                {
                    "label": "unusual_beneficiary",
                    "confidence": 0.8,
                    "evidence": "new vendor account",
                },
                {"label": "manufactured_urgency", "confidence": 0.8, "evidence": "right now"},
                {"label": "secrecy_demand", "confidence": 0.9, "evidence": "don't tell the team"},
            ]
        },
    ),
    ("Hi, I wanted to check my account balance and whether my cheque has cleared.", {"labels": []}),
    (
        "Sir aapka account block ho jayega, jaldi se OTP batao, kisi ko mat batana.",
        {
            "labels": [
                {"label": "manufactured_urgency", "confidence": 0.9, "evidence": "jaldi se"},
                {"label": "credential_request", "confidence": 0.95, "evidence": "OTP batao"},
                {"label": "secrecy_demand", "confidence": 0.9, "evidence": "kisi ko mat batana"},
            ]
        },
    ),
]


class LLMIntentClassifier:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 5.0,
        tenant_allows_external: bool = False,
    ) -> None:
        if not tenant_allows_external and not is_local_endpoint(base_url):
            raise NonLocalEndpointError(
                f"intent LLM endpoint {base_url!r} is outside the deployment boundary (I6); "
                "use a local Ollama/vLLM server or record explicit tenant opt-in"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def messages(self, transcript: str) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for text, answer in FEW_SHOT:
            msgs += [
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(answer)},
            ]
        msgs.append({"role": "user", "content": transcript[-4000:]})
        return msgs

    @staticmethod
    def parse(raw: str) -> IntentResult:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return IntentResult([], "llm")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return IntentResult([], "llm")
        hits = []
        for item in data.get("labels", []):
            label = str(item.get("label", ""))
            if label in LABELS:
                conf = float(min(max(float(item.get("confidence", 0.5)), 0.0), 1.0))
                hits.append(IntentHit(label, conf, str(item.get("evidence", ""))[:200]))
        return IntentResult(hits, "llm")

    def classify(self, transcript: str, language: str | None = None) -> IntentResult:
        import httpx

        body = {
            "model": self.model,
            "messages": self.messages(transcript),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        return self.parse(r.json()["choices"][0]["message"]["content"])


def merge(rule: IntentResult, llm: IntentResult | None) -> IntentResult:
    """Union of hits; the LLM can add labels and raise confidence, never silently delete a rule hit."""
    if llm is None:
        return rule
    return IntentResult(rule.hits + llm.hits, "rules+llm")
