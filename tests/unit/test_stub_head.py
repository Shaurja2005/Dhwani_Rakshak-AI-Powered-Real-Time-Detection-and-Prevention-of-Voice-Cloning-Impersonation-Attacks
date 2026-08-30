"""Unit tests for vg_core.stub_head and vg_core.head_api."""
from __future__ import annotations

import pytest

from packages.vg_core.head_api import BaseDetectionHead, HeadRegistry
from packages.vg_core.models import (
    AbstainReason,
    AnalysisWindow,
    HeadID,
    SessionContext,
)
from packages.vg_core.stub_head import StubHead

SESSION_ID = "01900b1a-0000-7000-8000-000000000001"


def make_window(quality_ok: bool = True) -> AnalysisWindow:
    return AnalysisWindow(
        session_id=SESSION_ID,
        window_id=0,
        start_ms=0,
        end_ms=3000,
        samples_ref="shm://test/0",
        voiced_ms=2500 if quality_ok else 100,
        snr_db=18.0,
        clipping_ratio=0.0,
        quality_ok=quality_ok,
        original_sample_rate=8000,
    )


def make_ctx() -> SessionContext:
    return SessionContext(
        session_id=SESSION_ID,
        tenant_id="test-tenant",
    )


class TestStubHead:
    def test_deterministic_score(self) -> None:
        """Same inputs must always produce the same score."""
        head = StubHead(head_id="A", seed=42)
        w = make_window()
        ctx = make_ctx()
        s1 = head.score(w, ctx)
        s2 = head.score(w, ctx)
        assert s1.p_spoof == s2.p_spoof

    def test_different_windows_give_different_scores(self) -> None:
        head = StubHead(head_id="A", seed=42)
        ctx = make_ctx()
        scores = set()
        for wid in range(10):
            w = AnalysisWindow(
                session_id=SESSION_ID,
                window_id=wid,
                start_ms=wid * 1000,
                end_ms=(wid + 3) * 1000,
                samples_ref=f"shm://test/{wid}",
                voiced_ms=2500,
                snr_db=18.0,
                clipping_ratio=0.0,
                quality_ok=True,
                original_sample_rate=8000,
            )
            s = head.score(w, ctx)
            if not s.abstain:
                scores.add(s.p_spoof)
        assert len(scores) > 1, "Expected diverse scores for different windows"

    def test_abstains_on_bad_quality(self) -> None:
        head = StubHead(head_id="A", seed=42, abstain_fraction=0.0)
        w = make_window(quality_ok=False)
        ctx = make_ctx()
        score = head.score(w, ctx)
        assert score.abstain is True
        assert score.p_spoof is None
        assert score.abstain_reason == AbstainReason.QUALITY_GATE

    def test_p_spoof_in_range(self) -> None:
        head = StubHead(head_id="A", seed=99, abstain_fraction=0.0)
        w = make_window()
        ctx = make_ctx()
        score = head.score(w, ctx)
        if not score.abstain:
            assert 0.0 <= score.p_spoof <= 1.0

    def test_head_id_matches(self) -> None:
        for hid in "ABCDEF":
            head = StubHead(head_id=hid)
            assert head.head_id == hid


class TestHeadRegistry:
    def test_register_and_get(self) -> None:
        reg = HeadRegistry()  # fresh instance (not singleton) for test isolation
        head = StubHead(head_id="A")
        reg.register(head)
        retrieved = reg.get("A")
        assert retrieved is head

    def test_get_missing_raises(self) -> None:
        from packages.vg_core.errors import HeadNotFoundError
        reg = HeadRegistry()
        with pytest.raises(HeadNotFoundError):
            reg.get("Z")

    def test_warmup_all(self) -> None:
        reg = HeadRegistry()
        for hid in "ABCDEF":
            reg.register(StubHead(head_id=hid))
        reg.warmup_all()  # should not raise

    def test_all_heads_returns_list(self) -> None:
        reg = HeadRegistry()
        for hid in "ABC":
            reg.register(StubHead(head_id=hid))
        heads = reg.all_heads()
        assert len(heads) == 3


class TestBaseDetectionHeadBudget:
    """Test the score_with_budget safety wrapper."""

    class RaisingHead(BaseDetectionHead):
        head_id = "A"
        model_version = "test-v0.1"
        budget_ms = 100

        def score(self, window: AnalysisWindow, ctx: SessionContext):  # type: ignore
            raise RuntimeError("deliberate error")

    def test_exception_becomes_abstain(self) -> None:
        head = self.RaisingHead()
        w = make_window()
        ctx = make_ctx()
        score = head.score_with_budget(w, ctx)
        assert score.abstain is True
        assert "error" in score.evidence
