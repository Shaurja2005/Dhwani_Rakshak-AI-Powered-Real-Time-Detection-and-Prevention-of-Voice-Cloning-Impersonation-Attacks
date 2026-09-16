"""B8 — Head E (active liveness) and Head F (watermark probe) tests."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml.data.channel.base import ChannelRecord, resample
from ml.data.channel.codecs import G711, FFmpegCodec, codec_available
from packages.vg_core.models import AbstainReason, AnalysisWindow, HeadID, HeadScore, SessionContext
from packages.vg_core.sample_store import put_samples
from packages.vg_core.stub_head import StubHead
from packages.vg_models.heads.head_e_liveness.api import create_router
from packages.vg_models.heads.head_e_liveness.challenge import ChallengeRegistry, generate
from packages.vg_models.heads.head_e_liveness.head import HeadE
from packages.vg_models.heads.head_e_liveness.verifier import verify
from packages.vg_models.heads.head_f_watermark.asymmetry import fusion_inputs, is_exonerating_misuse
from packages.vg_models.heads.head_f_watermark.detectors import (
    PartnerAPIDetector,
    SpreadSpectrumDetector,
    embed_spread_spectrum,
)
from packages.vg_models.heads.head_f_watermark.head import HeadF

SR = 16000
SID = "01900b1a-0000-7000-8000-00000000e8e1"
CTX = SessionContext(session_id=SID, tenant_id="t")


def _window(wid: int, pcm: np.ndarray | None = None, session_id: str = SID) -> AnalysisWindow:
    ref = f"shm://{session_id}/e{wid}"
    if pcm is not None:
        put_samples(ref, pcm)
    return AnalysisWindow(
        session_id=session_id,
        window_id=wid,
        start_ms=0,
        end_ms=3000,
        samples_ref=ref,
        voiced_ms=2500,
        snr_db=20.0,
        clipping_ratio=0.0,
        quality_ok=True,
        original_sample_rate=16000,
    )


def _answer(
    challenge_tokens: list[str], start_ms: int, gaps_ms: list[int] | None = None
) -> list[tuple[str, int, int]]:
    out, t = [], start_ms
    for i, w in enumerate(challenge_tokens):
        out.append((w, t, t + 300))
        t += 300 + (gaps_ms[i % len(gaps_ms)] if gaps_ms else 150)
    return out


# ---------------------------------------------------------------- challenge generator (T01)
def test_challenges_are_unpredictable_and_code_switched() -> None:
    prompts = {generate(SID, "digits_codeswitch").prompt_text for _ in range(30)}
    assert len(prompts) > 25
    ch = generate(SID, "digits_codeswitch", ("en", "hi"), length=4)
    assert len(ch.expected) == 4 and ch.prompt_text.startswith("Please repeat")
    hindi_forms = {f for t in ch.expected[1::2] for f in t.forms}
    assert hindi_forms & {"एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "शून्य"}
    rev = generate(SID, "reverse_order", ("en",))
    shown = [w.strip() for w in rev.prompt_text.split(":", 1)[1].split(",")]
    assert [t.display for t in rev.expected] == list(reversed(shown))


# ---------------------------------------------------------------- verifier (T02)
def test_correct_prompt_answer_with_human_latency_is_low_risk() -> None:
    ch = generate(SID, "digits_codeswitch")
    ch.prompt_end_ms = 10_000
    v = verify(ch, _answer([t.forms[-1] for t in ch.expected], 10_800, gaps_ms=[120, 260, 90]))
    assert v.content_match == 1.0 and v.latency_ms == 800 and v.p_spoof < 0.2


def test_romanised_and_digit_forms_both_match() -> None:
    ch = generate(SID, "digits_codeswitch")
    ch.prompt_end_ms = 0
    assert (
        verify(ch, _answer([t.forms[0] for t in ch.expected], 700, [100, 250])).content_match == 1.0
    )


def test_pipeline_latency_wrong_content_and_metronomic_delivery_raise_risk() -> None:
    ch = generate(SID, "nonce_phrase", ("en",))
    ch.prompt_end_ms = 0
    slow = verify(ch, _answer([t.forms[0] for t in ch.expected], 4200, [180, 60]))
    wrong = verify(ch, _answer(["hello", "there", "friend"], 900, [100, 300]))
    instant = verify(ch, _answer([t.forms[0] for t in ch.expected], 50, [180, 60]))
    metronome = verify(ch, _answer([t.forms[0] for t in ch.expected] * 2, 900, [150]))
    assert slow.p_spoof > 0.8 and any("pipeline" in r for r in slow.reasons)
    assert wrong.p_spoof > 0.85 and wrong.content_match == 0.0
    assert instant.p_spoof > 0.6
    assert any("evenly spaced" in r for r in metronome.reasons)


def test_reverse_order_requires_reversed_answer() -> None:
    ch = generate(SID, "reverse_order", ("en",))
    ch.prompt_end_ms = 0
    in_order = [t.forms[0] for t in reversed(ch.expected)]
    assert verify(ch, _answer(in_order, 800, [100, 200])).content_match < 1.0
    assert verify(ch, _answer(list(reversed(in_order)), 800, [100, 200])).content_match == 1.0


# ---------------------------------------------------------------- head E + operator hook (T03)
def test_head_e_abstains_until_answered_then_scores() -> None:
    reg = ChallengeRegistry()
    head = HeadE(reg)
    assert head.score(_window(0), CTX).abstain_reason == AbstainReason.NO_CHALLENGE
    ch = reg.issue(generate(SID, "digits_codeswitch"))
    reg.mark_prompt_end(SID, 5000)
    pending = head.score(_window(1), CTX)
    assert (
        pending.abstain_reason == AbstainReason.NO_CHALLENGE
        and pending.evidence["note"] == "awaiting answer"
    )
    assert reg.submit_response(
        SID, ch.challenge_id, _answer([t.forms[0] for t in ch.expected], 9000, [100])
    )
    s = head.score(_window(2), CTX)
    assert not s.abstain and s.p_spoof is not None and s.p_spoof > 0.8
    assert s.evidence["latency_ms"] == 4000 and s.calibration_version == "heuristic-uncalibrated"
    assert not reg.submit_response(SID, "not-this-one", [])
    ch.issued_at = time.time() - 60
    reg.issue(ch)  # re-issue drops the old answer
    assert head.score(_window(3), CTX).evidence["note"] == "challenge expired without an answer"


def test_operator_challenge_api() -> None:
    reg = ChallengeRegistry()
    app = FastAPI()
    app.include_router(create_router(reg))
    c = TestClient(app)
    base = f"/v1/sessions/{SID}/challenges"
    r = c.post(base, json={"kind": "nonce_phrase", "languages": ["en"]})
    assert r.status_code == 200 and r.json()["prompt_text"].startswith("Please say")
    cid = r.json()["challenge_id"]
    assert c.post(f"{base}/{cid}/prompt-end", json={"prompt_end_ms": 1000}).status_code == 204
    words = [t.forms[0] for t in reg.active(SID).expected]  # type: ignore[union-attr]
    tokens = [
        {"text": w, "start_ms": 1900 + 450 * i, "end_ms": 2200 + 450 * i + 40 * i}
        for i, w in enumerate(words)
    ]
    r = c.post(f"{base}/{cid}/response", json={"tokens": tokens})
    assert (
        r.status_code == 200 and r.json()["content_match"] == 1.0 and r.json()["latency_ms"] == 900
    )
    assert c.get(f"{base}/active").json()["answered"] is True
    assert c.post(f"{base}/nope/response", json={"tokens": []}).status_code == 404


# ---------------------------------------------------------------- watermark detectors (T04/T05)
def _speechlike(seed: int, seconds: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = (np.sin(2 * np.pi * 2.5 * t) > 0).astype(float)
    x = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 20)) * env
    return (0.2 * x / np.abs(x).max() + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def test_spread_spectrum_detects_from_any_offset_and_rejects_wrong_key_and_clean() -> None:
    det = SpreadSpectrumDetector({"demo": "secret-1"})
    other = SpreadSpectrumDetector({"x": "secret-2"})
    x = _speechlike(0)
    wm = embed_spread_spectrum(x, "secret-1")
    for off in (0, 1234, 9001):
        r = det.detect(wm[off : off + 48000])
        assert r.present and r.detail["key_id"] == "demo" and r.score > 20
    assert not det.detect(x[:48000]).present
    assert not other.detect(wm[:48000]).present
    assert det.detect(wm[:4000]).detail["note"] == "window too short"


def test_spread_spectrum_survives_telephony_codecs() -> None:
    det = SpreadSpectrumDetector({"demo": "secret-1"})
    wm = embed_spread_spectrum(_speechlike(1), "secret-1")
    y, sr = G711().apply(wm, SR, np.random.default_rng(0), ChannelRecord())
    assert det.detect(resample(y, sr, SR)[3000:51000]).present
    if codec_available("opus"):
        o, _ = FFmpegCodec("opus", 16).apply(wm, SR, np.random.default_rng(0), ChannelRecord())
        assert det.detect(o[3000:51000]).present


def test_partner_api_detector_requires_tenant_opt_in() -> None:
    with pytest.raises(PermissionError, match="opt-in"):
        PartnerAPIDetector("some-vendor", tenant_opted_in=False)


# ---------------------------------------------------------------- head F + asymmetry (T04/T06, DoD)
def test_watermarked_window_flagged_within_one_window() -> None:
    head = HeadF([SpreadSpectrumDetector({"demo": "secret-1"})])
    s = head.score(_window(10, embed_spread_spectrum(_speechlike(2), "secret-1")[:48000]), CTX)
    assert not s.abstain and (s.p_spoof or 0) >= 0.95 and s.evidence["watermark"] == "present"


def test_absent_watermark_is_neutral_and_no_detector_is_not_checked() -> None:
    s = HeadF([SpreadSpectrumDetector({"demo": "secret-1"})]).score(
        _window(11, _speechlike(3)[:48000]), CTX
    )
    assert s.abstain and s.abstain_reason is None and s.p_spoof is None
    assert s.evidence["watermark"] == "absent" and s.evidence["neutral"] is True
    none = HeadF([]).score(_window(12, _speechlike(3)[:48000]), CTX)
    assert (
        none.abstain_reason == AbstainReason.UNTRAINED
        and none.evidence["watermark"] == "not_checked"
    )


def _mean_fusion(scores: list[HeadScore]) -> float:
    active = [s.p_spoof for s in fusion_inputs(scores) if not s.abstain and s.p_spoof is not None]
    return float(np.mean(active)) if active else 0.0


def test_dod_unwatermarked_clip_changes_fused_score_by_exactly_zero() -> None:
    session = "01900b1a-0000-7000-8000-00000000e8e2"
    ctx = SessionContext(session_id=session, tenant_id="t")
    others = [
        StubHead(h, abstain_fraction=0.0).score(_window(0, session_id=session), ctx) for h in "ABC"
    ]
    head = HeadF([SpreadSpectrumDetector({"demo": "secret-1"})])
    clean = head.score(_window(20, _speechlike(4)[:48000], session), ctx)
    marked = head.score(
        _window(21, embed_spread_spectrum(_speechlike(4), "secret-1")[:48000], session), ctx
    )
    assert _mean_fusion(others + [clean]) == _mean_fusion(others)  # exactly zero change
    assert _mean_fusion(others + [marked]) > _mean_fusion(others)

    # A misbehaving Head F that tries to exonerate is filtered out and detectable.
    bad = clean.model_copy(update={"abstain": False, "p_spoof": 0.0, "abstain_reason": None})
    assert is_exonerating_misuse(bad)
    assert _mean_fusion(others + [bad]) == _mean_fusion(others)
    assert all(s.head_id != HeadID.F for s in fusion_inputs(others + [bad]))


def test_head_f_never_raises() -> None:
    class Broken:
        name = "broken"

        def detect(self, pcm: np.ndarray, sample_rate: int = SR) -> object:
            raise RuntimeError("detector crashed")

    s = HeadF([Broken()]).score(_window(30, _speechlike(5)[:48000]), CTX)  # type: ignore[list-item]
    assert s.abstain and s.evidence["neutral"] is True
