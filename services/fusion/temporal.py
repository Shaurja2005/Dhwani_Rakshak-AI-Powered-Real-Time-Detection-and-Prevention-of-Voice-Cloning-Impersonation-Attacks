"""Temporal fusion (B9-T03) and the any-segment trigger (B9-T04).

**Two-state HMM** (genuine ↔ synthetic) run as a forward filter over windows.
Each scored window contributes a log-likelihood ratio

    LLR_t = logit(p_fused_t) − logit(window_prior)

(the fused window probability already includes the window prior, so it is
removed to get pure evidence). Abstained windows contribute no evidence — only
the transition step — so bad audio never moves the session score. A small
switch probability makes the posterior *firm up* as consistent evidence
accumulates instead of flickering window to window, while still letting it move
if the call genuinely changes.

**Any-segment trigger.** A spliced attack (a few synthetic seconds inside a
genuine call) is diluted by a session-level filter. We therefore also track the
maximum *segment posterior*: for every run of ``segment_windows`` consecutive
scored windows, sigmoid(logit(segment_prior) + scale · Σ LLR). Using summed
evidence (not averaged probabilities) means genuine-leaning windows pull a
segment down, so chance fluctuations over a long call do not trigger it.
Session risk takes the maximum of the HMM posterior and the segment maximum;
both are exposed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from packages.vg_models.calibration import logit, sigmoid


@dataclass
class TemporalConfig:
    session_prior: float = 0.02  # P(synthetic) at call start; tenant/base-rate dependent
    window_prior: float = 0.5  # prior implicit in fused window probabilities
    switch_prob: float = 0.02  # per-window P(state change)
    llr_clip: float = 6.0  # cap single-window evidence (robustness to one wild window)
    evidence_scale: float = 0.6  # windows overlap (3 s / 1 s hop) -> evidence is correlated
    segment_windows: int = 3
    segment_prior: float = 0.1


@dataclass
class TemporalState:
    cfg: TemporalConfig = field(default_factory=TemporalConfig)
    log_odds: float = field(init=False)
    scored: list[float] = field(default_factory=list)  # fused p of scored windows, in order
    segment_max: float = 0.0  # max evidence-based segment posterior (drives risk)
    segment_mean_max: float = 0.0  # max mean fused p over a segment (descriptive, >= p_mean)
    segment_max_window: int | None = None
    n_windows: int = 0
    n_abstained: int = 0
    _recent: deque[float] = field(default_factory=deque)
    _recent_p: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.log_odds = float(logit(self.cfg.session_prior))

    @property
    def p_hmm(self) -> float:
        return float(sigmoid(self.log_odds))

    @property
    def p_mean(self) -> float:
        return float(np.mean(self.scored)) if self.scored else 0.0

    @property
    def abstain_ratio(self) -> float:
        return self.n_abstained / self.n_windows if self.n_windows else 1.0

    @property
    def p_session(self) -> float:
        return max(self.p_hmm, self.segment_max)

    def _transition(self) -> None:
        p = self.p_hmm
        e = self.cfg.switch_prob
        p = p * (1 - e) + (1 - p) * e
        self.log_odds = float(logit(p))

    def update(self, window_id: int, p_fused: float | None) -> None:
        self.n_windows += 1
        self._transition()
        if p_fused is None:
            self.n_abstained += 1
            return
        c = self.cfg
        llr = float(np.clip(logit(p_fused) - logit(c.window_prior), -c.llr_clip, c.llr_clip))
        self.log_odds += c.evidence_scale * llr
        self.log_odds = float(np.clip(self.log_odds, -30, 30))
        self.scored.append(p_fused)

        self._recent.append(llr)
        self._recent_p.append(p_fused)
        if len(self._recent) > c.segment_windows:
            self._recent.popleft()
            self._recent_p.popleft()
        self.segment_mean_max = max(self.segment_mean_max, float(np.mean(self._recent_p)))
        if len(self._recent) == c.segment_windows:
            seg = float(sigmoid(logit(c.segment_prior) + c.evidence_scale * sum(self._recent)))
            if seg > self.segment_max:
                self.segment_max, self.segment_max_window = seg, window_id


def state_changes_per_minute(states: list[str], hop_s: float = 1.0) -> float:
    """Score-stability metric (B9 DoD): transitions between non-identical states per minute."""
    if len(states) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(states, states[1:], strict=False) if a != b)
    return changes / (len(states) * hop_s / 60)


def clip_prob(p: float) -> float:
    return min(max(p, 0.0), 1.0) if not math.isnan(p) else 0.0
