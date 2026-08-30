"""Unit tests for vg_core.models — Pydantic model validation.

These tests verify that the models match the contracts in SOURCE_OF_TRUTH §4.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from packages.vg_core.models import (
    AbstainReason,
    AnalysisWindow,
    AudioChunk,
    AudioEncoding,
    CallMetadata,
    Channel,
    ConsentBasis,
    ContextSignals,
    FusedWindowScore,
    HeadID,
    HeadScore,
    PolicyDecision,
    RiskState,
    SessionContext,
    SessionRisk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_ID = "01900b1a-0000-7000-8000-000000000001"
NOW = datetime.now(tz=timezone.utc)


def minimal_window() -> AnalysisWindow:
    return AnalysisWindow(
        session_id=SESSION_ID,
        window_id=0,
        start_ms=0,
        end_ms=3000,
        samples_ref="shm://test/0",
        voiced_ms=2500,
        snr_db=18.5,
        clipping_ratio=0.001,
        quality_ok=True,
        original_sample_rate=8000,
    )


def minimal_head_score(abstain: bool = False) -> HeadScore:
    return HeadScore(
        session_id=SESSION_ID,
        window_id=0,
        head_id=HeadID.A,
        raw_score=-1.87,
        p_spoof=None if abstain else 0.83,
        abstain=abstain,
        abstain_reason=AbstainReason.TIMEOUT if abstain else None,
        confidence=0.71,
        latency_ms=42,
        model_version="A@xlsr300m-nes2net-v0.3.1",
        calibration_version="cal-2026-02-11",
        evidence={"test": True},
    )


# ---------------------------------------------------------------------------
# CallMetadata
# ---------------------------------------------------------------------------


class TestCallMetadata:
    def test_valid_inbound(self) -> None:
        m = CallMetadata(
            session_id=SESSION_ID,
            tenant_id="bfsi-demo",
            direction="inbound",
            started_at=NOW,
            channel=Channel.PSTN,
            codec_hint="g711u",
            source_sample_rate=8000,
            consent_basis=ConsentBasis.LEGITIMATE_USE,
        )
        assert m.shadow_mode is True  # default

    def test_invalid_direction(self) -> None:
        with pytest.raises(Exception):
            CallMetadata(
                session_id=SESSION_ID,
                tenant_id="demo",
                direction="sideways",  # invalid
                started_at=NOW,
                channel=Channel.FILE,
                codec_hint="pcm",
                source_sample_rate=16000,
                consent_basis=ConsentBasis.NONE,
            )


# ---------------------------------------------------------------------------
# AnalysisWindow
# ---------------------------------------------------------------------------


class TestAnalysisWindow:
    def test_valid_window(self) -> None:
        w = minimal_window()
        assert w.sample_rate == 16000  # always

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(Exception):
            AnalysisWindow(
                session_id=SESSION_ID,
                window_id=0,
                start_ms=3000,
                end_ms=0,  # end before start
                samples_ref="shm://test/0",
                voiced_ms=0,
                snr_db=0.0,
                clipping_ratio=0.0,
                quality_ok=False,
                original_sample_rate=8000,
            )


# ---------------------------------------------------------------------------
# HeadScore
# ---------------------------------------------------------------------------


class TestHeadScore:
    def test_p_spoof_required_when_not_abstain(self) -> None:
        with pytest.raises(Exception):
            HeadScore(
                session_id=SESSION_ID,
                window_id=0,
                head_id=HeadID.A,
                raw_score=0.0,
                p_spoof=None,    # must be set when abstain=False
                abstain=False,
                confidence=0.5,
                latency_ms=10,
                model_version="A@xlsr300m-nes2net-v0.1",
                calibration_version="cal-2026-01-01",
                evidence={},
            )

    def test_abstain_score_valid(self) -> None:
        s = minimal_head_score(abstain=True)
        assert s.p_spoof is None
        assert s.abstain_reason == AbstainReason.TIMEOUT

    def test_scored_result_valid(self) -> None:
        s = minimal_head_score(abstain=False)
        assert 0.0 <= s.p_spoof <= 1.0  # type: ignore[operator]


# ---------------------------------------------------------------------------
# FusedWindowScore
# ---------------------------------------------------------------------------


class TestFusedWindowScore:
    def test_valid(self) -> None:
        f = FusedWindowScore(
            session_id=SESSION_ID,
            window_id=0,
            p_spoof=0.79,
            state=RiskState.ELEVATED,
            contributions={"A": 0.51, "B": 0.12},
            heads_abstained=["E", "F"],
            fusion_version="fuse-lr-v0.2",
        )
        assert f.state == RiskState.ELEVATED


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    def test_valid_actions(self) -> None:
        p = PolicyDecision(
            session_id=SESSION_ID,
            decided_at=NOW,
            threshold_profile="bfsi-default-v1",
            actions=["agent_banner", "hold_transaction"],
            rationale="risk_score 78 >= 70",
            evidence_bundle_id="sha256:abc123",
            shadow_mode=False,
            suppressed=False,
        )
        assert "hold_transaction" in p.actions

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(Exception):
            PolicyDecision(
                session_id=SESSION_ID,
                decided_at=NOW,
                threshold_profile="test",
                actions=["do_something_illegal"],
                rationale="...",
                evidence_bundle_id="sha256:abc",
                shadow_mode=True,
                suppressed=True,
            )
