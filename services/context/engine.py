"""ContextSignals assembly (B10-T06) and the combined-risk function.

``ContextEngine`` (one per session) ingests ASR segments, redacts them first
(B10-T07), tracks language / code-switching, runs rule intent synchronously and
the local LLM asynchronously (bounded by a timeout — I10), scores metadata and
transaction context, and produces ``ContextSignals`` plus per-signal
explanations.

``final_risk`` implements the plan's
    final_risk = f(voice_authenticity, speaker_mismatch, metadata_risk,
                   transaction_sensitivity, intent_risk)
as a noisy-OR over independent risk sources, with transaction sensitivity
amplifying (never creating) the other evidence. B11 consumes it.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
from dataclasses import dataclass, field

from packages.vg_core.logging import get_logger
from packages.vg_core.models import CallMetadata, ContextSignals, SessionRisk, TranscriptSnippet
from services.context import metadata_risk, transaction
from services.context.asr import RollingTranscript, Segment
from services.context.intent import (
    IntentClassifier,
    IntentResult,
    LLMIntentClassifier,
    RuleIntentClassifier,
    merge,
)
from services.context.langid import detect
from services.context.redact import redact

log = get_logger(__name__)
_POOL = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="intent-llm")


@dataclass
class ContextReport:
    signals: ContextSignals
    explanations: dict[str, dict[str, str] | list[str]] = field(default_factory=dict)
    transaction_tier: str = "low"


class ContextEngine:
    def __init__(
        self,
        meta: CallMetadata,
        history: metadata_risk.CallHistory | None = None,
        banking: transaction.TransactionConnector | None = None,
        llm: LLMIntentClassifier | IntentClassifier | None = None,
        hours: metadata_risk.TenantHours | None = None,
        llm_every_s: float = 10.0,
    ) -> None:
        self.meta = meta
        self.history = history or metadata_risk.CallHistory()
        self.banking = banking
        self.rules = RuleIntentClassifier()
        self.llm = llm
        self.transcript = RollingTranscript()
        self._redacted_segments: list[Segment] = []
        self._meta_risk = metadata_risk.score(meta, self.history, hours)
        self._llm_future: cf.Future[IntentResult] | None = None
        self._llm_result: IntentResult | None = None
        self._llm_last_ms = -(10**9)
        self._llm_every_ms = int(llm_every_s * 1000)

    # ------------------------------------------------------------------ ingest
    def add_segments(self, segments: list[Segment]) -> None:
        clean = []
        for s in segments:
            text, _ = redact(s.text)
            words = [(redact(w)[0], a, b) for w, a, b in s.words]
            clean.append(
                Segment(s.start_ms, s.end_ms, text, s.language, s.final, s.confidence, words)
            )
        self.transcript.add(clean)  # only redacted text is ever held (B10-T07)
        self._maybe_run_llm()

    def _maybe_run_llm(self) -> None:
        if self.llm is None:
            return
        if self._llm_future is not None and self._llm_future.done():
            try:
                self._llm_result = self._llm_future.result()
            except Exception as exc:  # noqa: BLE001 - LLM failures never affect the call path
                log.warning("intent_llm_failed", error=str(exc), session_id=self.meta.session_id)
            self._llm_future = None
        segs = self.transcript.segments(include_partial=False)
        end_ms = segs[-1].end_ms if segs else 0
        if self._llm_future is None and end_ms - self._llm_last_ms >= self._llm_every_ms and segs:
            self._llm_last_ms = end_ms
            text = self.transcript.text(include_partial=False)
            self._llm_future = _POOL.submit(self.llm.classify, text, self.meta.language_hint)

    # ------------------------------------------------------------------ output
    def report(self, as_of_ms: int | None = None, wait_llm_s: float = 0.0) -> ContextReport:
        if wait_llm_s and self._llm_future is not None:
            try:
                self._llm_result = self._llm_future.result(timeout=wait_llm_s)
                self._llm_future = None
            except Exception:  # noqa: BLE001, S110 - timeout or error: rules stand
                pass
        segs = self.transcript.segments()
        text = self.transcript.text()
        lang = detect(text, self.meta.language_hint)
        intent = merge(self.rules.classify(text, lang.language), self._llm_result)

        tx_risk = transaction.TransactionRisk(0.0, "low", {})
        if self.banking is not None:
            tx = self.banking.pending(self.meta.session_id)
            prof = (
                self.banking.profile(self.meta.claimed_identity_id or "")
                if self.meta.claimed_identity_id
                else None
            )
            tx_risk = transaction.score(tx, prof)

        snippets = []
        for h in intent.hits:
            seg = next(
                (s for s in segs if h.evidence and h.evidence.lower() in s.text.lower()), None
            )
            if seg is not None and all(sn.label != h.label for sn in snippets):
                snippets.append(
                    TranscriptSnippet(start_ms=seg.start_ms, text=seg.text[:300], label=h.label)
                )

        as_of = as_of_ms if as_of_ms is not None else (segs[-1].end_ms if segs else 0)
        signals = ContextSignals(
            session_id=self.meta.session_id,
            as_of_ms=max(as_of, 0),
            metadata_risk=self._meta_risk.risk,
            transaction_risk=tx_risk.risk,
            intent_risk=intent.risk,
            intent_labels=intent.labels,
            transcript_snippets=snippets,
            language_detected=lang.language,
            code_switched=lang.code_switched,
        )
        return ContextReport(
            signals,
            {
                "metadata": self._meta_risk.signals,
                "transaction": tx_risk.signals,
                "intent": [f"{h.label}: “{h.evidence}” ({intent.source})" for h in intent.hits],
                "language": {k: f"{v:.0%}" for k, v in lang.shares.items()},
            },
            tx_risk.tier,
        )


# ---------------------------------------------------------------- combined risk
TIER_AMPLIFIER = {"low": 0.8, "medium": 1.0, "high": 1.25}


def final_risk(
    voice: SessionRisk | None, ctx: ContextReport | None, speaker_mismatch: float | None = None
) -> dict[str, float]:
    """Combine acoustic and contextual evidence into one 0–1 risk with a per-factor breakdown.

    evidence = noisy-OR(voice authenticity, speaker mismatch, intent, 0.6·metadata)
    final    = 1 − (1 − min(1, evidence · tier_amplifier)) · (1 − 0.5·transaction_risk)

    The transaction tier scales the other evidence (a high-value wire makes the
    same doubt matter more) but cannot create risk by itself beyond its own term.
    ABSTAIN voice risk contributes nothing — never implies safety either.
    """
    voice_p = 0.0
    if voice is not None and voice.state.value != "ABSTAIN":
        voice_p = voice.risk_score / 100
    parts = {
        "voice_authenticity": voice_p,
        "speaker_mismatch": speaker_mismatch or 0.0,
        "intent": ctx.signals.intent_risk if ctx else 0.0,
        "metadata": 0.6 * ctx.signals.metadata_risk if ctx else 0.0,  # circumstantial
    }
    clean = 1.0
    for v in parts.values():
        clean *= 1 - min(max(v, 0.0), 1.0)
    tier = ctx.transaction_tier if ctx else "low"
    amp = TIER_AMPLIFIER[tier]
    tx_part = 0.5 * ctx.signals.transaction_risk if ctx else 0.0
    final = 1 - (1 - min(1.0, (1 - clean) * amp)) * (1 - tx_part)
    return {
        "final": round(final, 4),
        **{k: round(v, 4) for k, v in parts.items()},
        "transaction": round(tx_part, 4),
        "tier_amplifier": amp,
    }


def now_ms(meta: CallMetadata) -> int:
    return int((dt.datetime.now(dt.UTC) - meta.started_at).total_seconds() * 1000)
