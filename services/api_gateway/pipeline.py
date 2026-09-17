"""In-process session pipeline shared by every gateway front-end (gRPC, WebSocket, REST file analysis).

    AudioChunk bytes → decode/resample (B2) → 3 s / 1 s windows → VAD + quality gate
    → sample store → detection heads (B4–B8) → RiskEngine (B9)
    → [ContextEngine (B10)] → policy decision on state change / close (B11)

Emits ordered events: ``window_score``, ``session_risk``, ``context_signals``,
``policy_decision``. Raw audio only ever lives in memory (sample store, I5).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from packages.vg_audio.quality import assess_quality
from packages.vg_audio.resample import resample_chunk
from packages.vg_audio.vad import EnergyVAD
from packages.vg_audio.windowing import Windower
from packages.vg_core.head_api import DetectionHead
from packages.vg_core.logging import get_logger
from packages.vg_core.models import (
    AnalysisWindow,
    CallMetadata,
    RiskState,
    SessionContext,
    SessionRisk,
)
from packages.vg_core.sample_store import drop_session, put_samples
from services.context.asr import Segment
from services.context.engine import ContextEngine
from services.fusion.engine import RiskEngine
from services.fusion.persistence import TimelineStore
from services.policy.engine import decide
from services.policy.evidence import EvidenceStore
from services.policy.notify import Notifier
from services.policy.profiles import ProfileStore

log = get_logger(__name__)


@dataclass
class Event:
    type: str  # window_score | session_risk | context_signals | policy_decision
    data: Any  # pydantic model
    extra: dict[str, Any] = field(
        default_factory=dict
    )  # UI-facing additions (not part of the proto)

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data.model_dump(mode="json"), **self.extra}


def default_heads() -> list[DetectionHead]:
    """Heads from ``VG_GATEWAY_HEADS`` (default A,B,C,F). Untrained heads abstain honestly."""
    wanted = [
        h.strip().upper() for h in os.getenv("VG_GATEWAY_HEADS", "A,B,C,F").split(",") if h.strip()
    ]
    heads: list[DetectionHead] = []
    if wanted == [
        "STUB"
    ]:  # dev/demo only: deterministic pseudo-random scores, clearly versioned "stub"
        from packages.vg_core.stub_head import StubHead

        log.warning("gateway_stub_heads", note="scores are pseudo-random; for UI demos only")
        return [StubHead(h, abstain_fraction=0.05) for h in "ABC"]
    for h in wanted:
        if h == "A":
            from packages.vg_models.heads.head_a_ssl.head import HeadA

            heads.append(HeadA())
        elif h == "B":
            from packages.vg_models.heads.head_b_dsp.head import HeadB

            heads.append(HeadB())
        elif h == "C":
            from packages.vg_models.heads.head_c_prosody.head import HeadC

            heads.append(HeadC())
        elif h == "F":
            from packages.vg_models.heads.head_f_watermark.head import HeadF

            heads.append(HeadF())
    return heads


@dataclass
class Services:
    """Long-lived, shared objects (one per gateway process)."""

    profiles: ProfileStore = field(default_factory=ProfileStore)
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    notifier: Notifier = field(default_factory=Notifier)
    timeline: TimelineStore | None = None
    heads_factory: Callable[[], list[DetectionHead]] = default_heads


class SessionPipeline:
    def __init__(
        self,
        meta: CallMetadata,
        services: Services,
        heads: list[DetectionHead] | None = None,
        with_context: bool = True,
        on_audio_seconds: Callable[[float], None] | None = None,
    ) -> None:
        self.meta = meta
        self.services = services
        self.heads = heads if heads is not None else services.heads_factory()
        for h in self.heads:
            h.warmup()
        self.ctx = SessionContext(
            session_id=meta.session_id,
            tenant_id=meta.tenant_id,
            claimed_identity_id=meta.claimed_identity_id,
            shadow_mode=meta.shadow_mode,
            call_metadata=meta,
        )
        self.engine = RiskEngine(meta.session_id, store=services.timeline)
        self.context = ContextEngine(meta) if with_context else None
        self.windower = Windower(meta.session_id, meta.source_sample_rate)
        self.vad = EnergyVAD()
        self.last_risk: SessionRisk | None = None
        self.last_decision_state: RiskState | None = None
        self.last_labels: list[str] = []
        self.head_scores: list[Any] = []
        self.window_scores: list[Any] = []
        self.decisions: list[Any] = []
        self._on_audio = on_audio_seconds
        self.closed = False

    # ------------------------------------------------------------------ input
    def push_chunk(self, payload: bytes, encoding: str, sample_rate: int) -> list[Event]:
        pcm = resample_chunk(payload, encoding, sample_rate)
        if self._on_audio:
            self._on_audio(len(pcm) / 16000)
        return self._run(self.windower.push(pcm))

    def push_pcm16k(self, pcm: np.ndarray) -> list[Event]:
        if self._on_audio:
            self._on_audio(len(pcm) / 16000)
        return self._run(self.windower.push(np.asarray(pcm, dtype=np.float32)))

    def add_transcript(self, segments: list[Segment]) -> list[Event]:
        if self.context is None:
            return []
        self.context.add_segments(segments)
        rep = self.context.report()
        events = [Event("context_signals", rep.signals)]
        if sorted(rep.signals.intent_labels) != self.last_labels and self.last_risk is not None:
            self.last_labels = sorted(rep.signals.intent_labels)
            events += self._decide()
        return events

    def close(self) -> list[Event]:
        if self.closed:
            return []
        self.closed = True
        events = self._run(self.windower.flush())
        if self.last_risk is None:
            self.last_risk = self.engine.session_risk()
            events.append(Event("session_risk", self.last_risk))
        events += self._decide(force=True)
        drop_session(self.meta.session_id)
        return events

    # ------------------------------------------------------------------ core
    def _run(self, windows: list[Any]) -> list[Event]:
        events: list[Event] = []
        for w in windows:
            ratio, _ = self.vad.process_window(w.pcm)
            q = assess_quality(
                pcm=w.pcm, voiced_ratio=ratio, window_duration_s=(w.end_ms - w.start_ms) / 1000
            )
            ref = put_samples(f"shm://{self.meta.session_id}/{w.window_id}", w.pcm)
            window = AnalysisWindow(
                session_id=self.meta.session_id,
                window_id=w.window_id,
                start_ms=w.start_ms,
                end_ms=w.end_ms,
                samples_ref=ref,
                voiced_ms=q.voiced_ms,
                snr_db=q.snr_db,
                clipping_ratio=q.clipping_ratio,
                quality_ok=q.quality_ok,
                quality_flags=q.quality_flags,
                original_sample_rate=w.original_sample_rate,
            )
            scores = [
                (
                    h.score_with_budget(window, self.ctx)
                    if hasattr(h, "score_with_budget")
                    else h.score(window, self.ctx)
                )
                for h in self.heads
            ]
            self.head_scores.extend(scores)
            fused, risk = self.engine.update(w.window_id, scores)
            self.window_scores.append(fused)
            self.last_risk = risk
            events += [Event("window_score", fused), Event("session_risk", risk)]
            if risk.state != self.last_decision_state and risk.state in (
                RiskState.ELEVATED,
                RiskState.HIGH,
            ):
                events += self._decide()
        return events

    def _decide(self, force: bool = False) -> list[Event]:
        if self.last_risk is None:
            return []
        report = self.context.report() if self.context is not None else None
        if not force and self.last_risk.state == self.last_decision_state and report is None:
            return []
        profile = self.services.profiles.get(self.meta.tenant_id)
        res = decide(
            profile,
            self.last_risk,
            report,
            head_scores=self.head_scores[-60:],
            window_scores=self.window_scores[-60:],
            call_metadata=self.meta,
            store=self.services.evidence,
            notifier=self.services.notifier,
        )
        self.last_decision_state = self.last_risk.state
        self.decisions.append(res)
        return [
            Event(
                "policy_decision",
                res.decision,
                {
                    "agent_prompt": res.agent_prompt,
                    "band": res.band,
                    "tier": res.tier,
                    "risk": res.risk,
                },
            )
        ]
