"""B10 — context & intent: ASR buffer, language ID, redaction, intent, metadata, transaction, assembly."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from packages.vg_core.models import CallMetadata, Channel, ConsentBasis, RiskState, SessionRisk
from services.context import metadata_risk, transaction
from services.context.asr import RollingTranscript, ScriptedASR, Segment
from services.context.engine import ContextEngine, final_risk
from services.context.intent import (
    IntentResult,
    LLMIntentClassifier,
    NonLocalEndpointError,
    RuleIntentClassifier,
    is_local_endpoint,
)
from services.context.langid import detect
from services.context.main import create_app
from services.context.redact import redact
from services.context.seed import generate

SID = "01900b1a-0000-7000-8000-0000000c10c1"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def meta(**kw: object) -> CallMetadata:
    base: dict[str, object] = dict(
        session_id=SID,
        tenant_id="bank-a",
        direction="inbound",
        started_at=dt.datetime(2026, 9, 17, 11, 0, tzinfo=IST),
        caller_number="+919876543210",
        channel=Channel.PSTN,
        codec_hint="g711u",
        source_sample_rate=8000,
        consent_basis=ConsentBasis.LEGITIMATE_USE,
        claimed_identity_id="cust-1",
    )
    base.update(kw)
    return CallMetadata.model_validate(base)


# ---------------------------------------------------------------- ASR (T01)
def test_scripted_asr_and_rolling_transcript() -> None:
    asr = ScriptedASR([Segment(0, 1500, "hello this is"), Segment(1500, 3200, "the bank calling")])
    rt = RollingTranscript(keep_s=60)
    rt.add(asr.feed(np.zeros(24000, dtype=np.float32), 0))  # clock reaches 1.5 s
    assert rt.text() == "hello this is"
    rt.add(asr.feed(np.zeros(32000, dtype=np.float32), 1500))  # clock reaches 3.5 s
    assert rt.text() == "hello this is the bank calling"
    toks = rt.tokens()
    assert toks[0][0] == "hello" and toks[-1][2] == 3200 and len(toks) == 6
    rt.add([Segment(3200, 4000, "partial words", final=False)])
    assert rt.text().endswith("partial words") and not rt.text(include_partial=False).endswith(
        "words"
    )
    rt.add([Segment(200_000, 201_000, "much later")])
    assert rt.segments(include_partial=False)[0].text == "much later"  # old segments aged out


# ---------------------------------------------------------------- language (T02)
@pytest.mark.parametrize(
    ("text", "lang", "switched"),
    [
        ("मुझे अपना बैलेंस जानना है", "hi", False),
        ("Please check my account balance today", "en", False),
        ("Sir aapka account block ho jayega please OTP share kijiye", "hi", True),
        ("எனது கணக்கு இருப்பு என்ன", "ta", False),
        ("मला माहिती नाही आहे", "mr", False),
        ("నా ఖాతా బ్యాలెన్స్ ఎంత", "te", False),
    ],
)
def test_language_and_code_switch(text: str, lang: str, switched: bool) -> None:
    r = detect(text)
    assert r.language == lang and r.code_switched is switched


# ---------------------------------------------------------------- redaction (T07)
def test_redaction_covers_indian_banking_pii() -> None:
    text = (
        "my card is 4111 1111 1111 1111, otp is 482913, aadhaar 2345 6789 0123, call +91 98765 43210, "
        "IFSC HDFC0001234 PAN ABCDE1234F account 123456789012 upi ravi.k@okaxis mail ravi@example.com "
        "and the code is four five six seven"
    )
    out, spans = redact(text)
    kinds = {s.kind for s in spans}
    assert {
        "card_number",
        "otp",
        "aadhaar",
        "phone",
        "ifsc",
        "pan",
        "upi_id",
        "email",
        "spoken_digits",
    } <= kinds
    for secret in (
        "4111",
        "482913",
        "2345 6789",
        "98765",
        "HDFC0001234",
        "ABCDE1234F",
        "okaxis",
        "example.com",
        "four five",
    ):
        assert secret not in out
    assert "[OTP]" in out


def test_redaction_leaves_amounts_and_non_luhn_numbers_readable() -> None:
    out, spans = redact("please transfer 18 lakh, reference 1234 5678 9012 3456")
    assert "18 lakh" in out
    assert not any(s.kind == "card_number" for s in spans)  # fails Luhn
    hindi, _ = redact("ओटीपी 7788 बताइए")
    assert "7788" not in hindi


# ---------------------------------------------------------------- intent (T03)
def test_rule_intent_multilingual_script_detection() -> None:
    c = RuleIntentClassifier()
    en = c.classify(
        "This is the CFO. Wire 18 lakh to the new vendor account right now and don't tell the team."
    )
    assert {
        "authority_pressure",
        "payment_request",
        "unusual_beneficiary",
        "manufactured_urgency",
        "secrecy_demand",
    } <= set(en.labels)
    assert en.risk > 0.9
    hinglish = c.classify("aapka account block ho jayega, turant OTP batao, kisi ko mat batana")
    assert {"credential_request", "secrecy_demand", "manufactured_urgency"} <= set(hinglish.labels)
    hindi = c.classify("तुरंत ओटीपी बताइए, किसी को मत बताना")
    assert {"credential_request", "secrecy_demand"} <= set(hindi.labels)
    benign = c.classify(
        "Hi, I'd like to check my balance and the branch timings. Please call me back later."
    )
    assert benign.labels == [] and benign.risk == 0.0


def test_seed_set_size_and_rule_regression() -> None:
    rows = generate()
    assert (
        len(rows) >= 190
        and sum(r["fraud"] for r in rows) > 80
        and sum(not r["fraud"] for r in rows) > 60
    )
    c = RuleIntentClassifier()
    agree = sum(set(c.classify(str(r["text"])).labels) == set(r["labels"]) for r in rows)  # type: ignore[arg-type]
    assert agree / len(rows) >= 0.97
    committed = Path("services/context/data/intent_seed.jsonl")
    assert len(committed.read_text(encoding="utf-8").splitlines()) == len(rows)


def test_llm_endpoint_must_be_local_unless_opted_in() -> None:
    assert is_local_endpoint("http://localhost:11434") and is_local_endpoint(
        "http://127.0.0.1:8000"
    )
    assert is_local_endpoint("http://10.2.3.4:8000") and is_local_endpoint(
        "http://ollama.svc.cluster.local:11434"
    )
    assert not is_local_endpoint("http://8.8.8.8:443")
    with pytest.raises(NonLocalEndpointError, match="I6"):
        LLMIntentClassifier("http://8.8.8.8:443", "some-model")
    LLMIntentClassifier("http://8.8.8.8:443", "some-model", tenant_allows_external=True)


def test_llm_output_parsing_is_strict_and_robust() -> None:
    raw = 'Sure! {"labels": [{"label": "secrecy_demand", "confidence": 1.7, "evidence": "don\'t tell"}, {"label": "made_up", "confidence": 1}]}'
    r = LLMIntentClassifier.parse(raw)
    assert [h.label for h in r.hits] == ["secrecy_demand"] and r.hits[0].confidence == 1.0
    assert LLMIntentClassifier.parse("no json here").hits == []
    assert LLMIntentClassifier.parse("{not json}").hits == []
    msgs = LLMIntentClassifier("http://localhost:1", "m").messages("[OTP] please")
    assert msgs[0]["role"] == "system" and msgs[-1]["content"] == "[OTP] please"


# ---------------------------------------------------------------- metadata + transaction (T04, T05)
def test_metadata_risk_signals() -> None:
    h = metadata_risk.CallHistory(
        registered_numbers={"cust-1": {"+919000000001"}},
        trunk_reputation={"trunk-x": 0.8},
        international_trunks={"intl-gw"},
    )
    start = dt.datetime(2026, 9, 17, 23, 30, tzinfo=IST)
    for i in range(3):
        h.record("9876543210", start - dt.timedelta(minutes=10 * (i + 1)))
    r = metadata_risk.score(meta(started_at=start, trunk_id="intl-gw", source_asn="trunk-x"), h)
    assert {
        "cli_mismatch",
        "international_spoofed_cli",
        "bad_trunk_reputation",
        "off_hours",
        "high_velocity",
    } <= set(r.signals)
    assert r.risk > 0.9
    clean = metadata_risk.score(
        meta(caller_number="+919000000001"),
        metadata_risk.CallHistory(
            registered_numbers={"cust-1": {"+919000000001"}},
            calls={"9000000001": [dt.datetime(2026, 1, 1, tzinfo=IST)]},
        ),
    )
    assert clean.signals == {} and clean.risk == 0.0


def test_transaction_risk_and_tier() -> None:
    now = dt.datetime(2026, 9, 17, tzinfo=dt.UTC)
    prof = transaction.CustomerProfile(
        "cust-1",
        past_amounts=[20_000, 25_000, 18_000, 30_000, 22_000, 27_000],
        beneficiaries={
            "ben-old": now - dt.timedelta(days=200),
            "ben-new": now - dt.timedelta(hours=2),
        },
    )
    wire = transaction.score(
        transaction.PendingTransaction("transfer", 1_800_000, beneficiary_id="ben-x"), prof, now
    )
    assert {"unusual_amount", "new_beneficiary", "first_high_value"} <= set(
        wire.signals
    ) and wire.tier == "high"
    assert wire.risk > 0.7
    routine = transaction.score(
        transaction.PendingTransaction("transfer", 24_000, beneficiary_id="ben-old"), prof, now
    )
    assert routine.signals == {} and routine.tier == "medium" and routine.risk == 0.0
    recent = transaction.score(
        transaction.PendingTransaction("transfer", 24_000, beneficiary_id="ben-new"), prof, now
    )
    assert "recent_beneficiary" in recent.signals
    assert (
        transaction.score(transaction.PendingTransaction("limit_increase"), prof, now).tier
        == "high"
    )
    assert transaction.score(None, prof, now).tier == "low"


# ---------------------------------------------------------------- assembly + DoD (T06)
def _voice_low() -> SessionRisk:
    return SessionRisk(
        session_id=SID,
        updated_at=dt.datetime.now(dt.UTC),
        risk_score=8,
        state=RiskState.LOW,
        p_spoof_session_max=0.1,
        p_spoof_session_mean=0.05,
        abstain_ratio=0.1,
    )


def test_dod_genuine_voice_reading_fraud_script() -> None:
    bank = transaction.MockCoreBanking()
    bank.profiles["cust-1"] = transaction.CustomerProfile("cust-1", past_amounts=[20_000] * 10)
    bank.pending_by_session[SID] = transaction.PendingTransaction(
        "transfer", 1_800_000, beneficiary_id="ben-x"
    )
    eng = ContextEngine(meta(), banking=bank)
    eng.add_segments(
        [
            Segment(
                0, 4000, "Hello, this is the CFO, I am in a board meeting so don't call me back."
            ),
            Segment(4000, 9000, "I need you to wire 18 lakh to the new vendor account right now."),
            Segment(9000, 12000, "Keep this confidential, don't tell the team. My OTP is 482913."),
        ]
    )
    rep = eng.report()
    s = rep.signals
    assert s.intent_risk > 0.9 and len(s.intent_labels) >= 5
    assert (
        s.language_detected == "en" and s.transaction_risk > 0.5 and rep.transaction_tier == "high"
    )
    assert all("482913" not in sn.text for sn in s.transcript_snippets)  # redacted before storage
    assert all("482913" not in seg.text for seg in eng.transcript.segments())
    risk = final_risk(_voice_low(), rep)
    # The DoD moment: acoustic evidence is LOW, intent is HIGH, and both are visible in the breakdown.
    assert risk["voice_authenticity"] < 0.1 and risk["intent"] > 0.9 and risk["final"] > 0.9


def test_final_risk_properties() -> None:
    quiet = final_risk(_voice_low(), None)
    assert quiet["final"] == pytest.approx(0.08 * 0.8, abs=1e-3)  # low tier dampens weak evidence
    abstain = _voice_low().model_copy(update={"state": RiskState.ABSTAIN, "risk_score": 90})
    assert final_risk(abstain, None)["voice_authenticity"] == 0.0


def test_llm_runs_async_never_blocks_and_failures_are_ignored() -> None:
    class SlowLLM:
        def classify(self, transcript: str, language: str | None = None) -> IntentResult:
            time.sleep(1.0)
            return LLMIntentClassifier.parse(
                '{"labels": [{"label": "unusual_channel", "confidence": 0.9, "evidence": "x"}]}'
            )

    class BrokenLLM:
        def classify(self, transcript: str, language: str | None = None) -> IntentResult:
            raise ConnectionError("ollama down")

    eng = ContextEngine(meta(), llm=SlowLLM(), llm_every_s=0)
    t0 = time.monotonic()
    eng.add_segments([Segment(0, 2000, "hello, can you check my balance")])
    rep = eng.report()
    assert time.monotonic() - t0 < 0.5 and rep.signals.intent_labels == []  # not blocked by the LLM
    assert "unusual_channel" in eng.report(wait_llm_s=3.0).signals.intent_labels

    broken = ContextEngine(meta(), llm=BrokenLLM(), llm_every_s=0)
    broken.add_segments([Segment(0, 2000, "turant OTP batao")])
    time.sleep(0.1)
    broken.add_segments([Segment(2000, 3000, "please")])
    assert "credential_request" in broken.report().signals.intent_labels  # rules still stand


def test_context_service_api() -> None:
    c = TestClient(create_app())
    assert c.post("/v1/sessions", json=json.loads(meta().model_dump_json())).status_code == 201
    r = c.post(
        f"/v1/sessions/{SID}/transcript",
        json=[{"start_ms": 0, "end_ms": 3000, "text": "kisi ko mat batana, jaldi OTP 1234 batao"}],
    )
    assert r.status_code == 204
    body = c.get(f"/v1/sessions/{SID}/context").json()
    assert (
        body["signals"]["intent_risk"] > 0.5
        and "secrecy_demand" in body["signals"]["intent_labels"]
    )
    assert "1234" not in json.dumps(body)
    assert c.get("/v1/sessions/nope/context").status_code == 404
