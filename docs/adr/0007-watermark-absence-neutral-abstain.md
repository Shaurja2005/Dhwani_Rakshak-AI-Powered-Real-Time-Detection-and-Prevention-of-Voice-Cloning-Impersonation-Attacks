# ADR 0007 — Watermark absence is a neutral abstain

**Status:** Accepted (2026-09-17)

## Context

SOURCE_OF_TRUTH §6 describes Head F as "never abstains; absence ⇒ neutral", and
invariant I4 says watermark absence must change the fused score by *zero*
(B8 Definition of Done). The `HeadScore` contract requires a `p_spoof` whenever
`abstain=false`. Any `p_spoof` value — 0.0 (exoneration, explicitly forbidden),
0.5, or a prior — shifts a weighted or averaged fused score, so "not abstaining"
and "exactly neutral" cannot both hold.

## Decision

- Watermark **present**: `abstain=false`, `p_spoof >= 0.95`, `evidence.watermark = "present"`.
- Watermark **absent** (detectors ran, nothing found): `abstain=true`,
  `abstain_reason=null`, `evidence.watermark = "absent"`, `evidence.neutral = true`.
- **No detector available**: `abstain=true`, `abstain_reason="untrained"` (ADR 0006),
  `evidence.watermark = "not_checked"`.
- `packages/vg_models/heads/head_f_watermark/asymmetry.py::fusion_inputs` removes
  every Head F score that is not a positive detection; B9 fusion must use it.

"Never abstains" in §6 is reinterpreted as "always runs when a detector is
available"; the §6 row is updated accordingly.

## Alternatives considered

- **Emit p_spoof = 0.5 with confidence 0.** Still moves unweighted means; relies
  on every consumer honouring confidence (the dev replay fusion did not).
- **New abstain reason `watermark_absent`.** Clearer, but a second contract change
  for something the `evidence` field already expresses; revisit if B13 needs it.

## Consequences

- Abstain-rate dashboards (B17) must not count `evidence.watermark == "absent"`
  as an audio-quality abstain.
- B9 tests must include the DoD check: an unwatermarked window changes the fused
  score by exactly zero.

## Blocks affected

B8 (emitter), B9 (fusion), B13 (UI wording), B17 (metrics).
