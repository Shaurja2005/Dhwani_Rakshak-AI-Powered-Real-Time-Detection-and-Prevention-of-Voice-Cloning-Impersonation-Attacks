"""Session risk engine: calibration → fusion → temporal → states → SessionRisk.

One ``RiskEngine`` per session. Call ``update(window_id, head_scores)`` once per
analysis window; it returns the ``FusedWindowScore`` and the updated
``SessionRisk`` and (optionally) persists the timeline for forensics (B9-T08).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from packages.vg_core.models import (
    FusedWindowScore,
    HeadScore,
    RiskDriver,
    RiskState,
    SessionRisk,
    WindowSummary,
)
from packages.vg_models.calibration import CalibrationStore
from services.fusion.fuser import FusionModel, fuse
from services.fusion.operating_point import OperatingProfile, StateMachine
from services.fusion.persistence import TimelineRow, TimelineStore
from services.fusion.temporal import TemporalConfig, TemporalState

FACTORS = {
    "A": "synthetic_speech_ssl",
    "B": "dsp_artifacts_and_scene",
    "C": "prosody",
    "D": "speaker_mismatch",
    "E": "liveness_challenge",
    "F": "watermark",
}


class RiskEngine:
    def __init__(
        self,
        session_id: str,
        model: FusionModel | None = None,
        profile: OperatingProfile | None = None,
        calibration: CalibrationStore | None = None,
        temporal: TemporalConfig | None = None,
        store: TimelineStore | None = None,
        timeline_limit: int = 600,
    ) -> None:
        self.session_id = session_id
        self.model = model or FusionModel.prior()
        self.profile = profile or OperatingProfile()
        self.calibration = calibration or CalibrationStore()
        self.temporal = TemporalState(temporal or TemporalConfig())
        self.states = StateMachine(self.profile)
        self.store = store
        self.timeline: list[WindowSummary] = []
        self.timeline_limit = timeline_limit
        self.model_versions: dict[str, str] = {}
        self._contrib_sum: dict[str, float] = defaultdict(float)
        self._signed_sum: dict[str, float] = defaultdict(float)

    def update(
        self, window_id: int, scores: list[HeadScore]
    ) -> tuple[FusedWindowScore, SessionRisk]:
        wf = fuse(scores, self.model, self.calibration)
        self.model_versions.update(wf.model_versions)
        self.temporal.update(window_id, wf.p_spoof)
        for h, v in wf.signed.items():
            self._signed_sum[h] += v
            self._contrib_sum[h] += abs(v)

        state = self.states.update(
            self.temporal.p_session, len(self.temporal.scored), self.temporal.abstain_ratio
        )
        window_state = RiskState.ABSTAIN if wf.p_spoof is None else self._window_state(wf.p_spoof)
        fused = FusedWindowScore(
            session_id=self.session_id,
            window_id=window_id,
            p_spoof=0.0 if wf.p_spoof is None else wf.p_spoof,
            state=window_state,
            contributions=wf.contributions,
            heads_abstained=wf.abstained,
            fusion_version=self.model.version,
        )
        self.timeline.append(
            WindowSummary(window_id=window_id, p_spoof=fused.p_spoof, state=window_state)
        )
        if len(self.timeline) > self.timeline_limit:
            self.timeline = self.timeline[-self.timeline_limit :]

        risk = self.session_risk(state)
        if self.store is not None:
            self.store.append(
                TimelineRow(
                    session_id=self.session_id,
                    window_id=window_id,
                    ts=datetime.now(tz=UTC).isoformat(),
                    p_window=wf.p_spoof,
                    window_state=window_state.value,
                    p_session=self.temporal.p_session,
                    p_hmm=self.temporal.p_hmm,
                    p_segment_max=self.temporal.segment_max,
                    state=state.value,
                    risk_score=risk.risk_score,
                    contributions=wf.contributions,
                    heads_abstained=wf.abstained,
                    fusion_version=self.model.version,
                    calibration_versions=wf.calibration_versions,
                    model_versions=wf.model_versions,
                )
            )
        return fused, risk

    def _window_state(self, p: float) -> RiskState:
        return (
            RiskState.HIGH
            if p >= self.profile.high
            else RiskState.ELEVATED if p >= self.profile.elevated else RiskState.LOW
        )

    def drivers(self, top: int = 3) -> list[RiskDriver]:
        total = sum(self._contrib_sum.values())
        if total <= 0:
            return []
        t = self.temporal
        ranked = sorted(self._contrib_sum.items(), key=lambda kv: -kv[1])[:top]
        out = []
        for h, v in ranked:
            direction = "towards synthetic" if self._signed_sum[h] > 0 else "towards genuine"
            out.append(
                RiskDriver(
                    factor=FACTORS.get(h, h),
                    weight=round(v / total, 3),
                    detail=f"head {h}: net evidence {direction}",
                )
            )
        if (
            t.segment_max > t.p_hmm
            and t.segment_max >= self.profile.elevated
            and t.segment_max_window is not None
        ):
            out.append(
                RiskDriver(
                    factor="any_segment_trigger",
                    weight=round(t.segment_max, 3),
                    detail=f"short high-risk segment ending at window {t.segment_max_window}",
                )
            )
        return out

    def session_risk(self, state: RiskState | None = None) -> SessionRisk:
        t = self.temporal
        mv = dict(self.model_versions)
        mv["fusion"] = self.model.version
        return SessionRisk(
            session_id=self.session_id,
            updated_at=datetime.now(tz=UTC),
            risk_score=int(round(100 * t.p_session)),
            state=state or self.states.state,
            # Descriptive statistics of fused window scores (max >= mean). The risk
            # decision uses the evidence-based HMM / segment posteriors (risk_score).
            p_spoof_session_max=round(max(t.segment_mean_max, t.p_mean), 6),
            p_spoof_session_mean=round(t.p_mean, 6),
            drivers=self.drivers(),
            timeline=list(self.timeline),
            model_versions=mv,
            abstain_ratio=round(t.abstain_ratio, 6),
        )
