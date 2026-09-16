# ADR 0006 — `untrained` as an abstain reason

**Status:** Accepted (2026-09-17, project owner)

## Context

Heads that need learned weights (A, and the B classifier) can be deployed
before those weights exist: during the structure-first build, when a new head
is added, or after a rollback to a head with no valid checkpoint. Such a head
has no basis to score (I2) and must abstain. The existing reasons
(`no_enrollment`, `insufficient_speech`, `timeout`, `quality_gate`,
`no_challenge`) all describe the *audio* or the *session*, not the *model*, so
Head A was abstaining with `abstain_reason: null`.

## Decision

Add `untrained` to `AbstainReason` in the JSON schema, the protobuf enum
(`ABSTAIN_REASON_UNTRAINED = 6`) and the Pydantic model. A head emits it when it
has no trained weights loaded. The raw (meaningless) score may be recorded in
`evidence` for debugging, never in `p_spoof`.

This is **not** temporary: it stays valid after training, because any head can
again be without weights (new head, failed checkpoint load, rollback).

## Alternatives considered

- **Keep `null`.** Contract-legal but ambiguous; fusion and the UI cannot tell
  "no model" from "unspecified".
- **Emit scores with `confidence = 0`.** Rejected: consumers that ignore
  confidence (the replay fusion did) turn random scores into alerts.
- **Reuse `quality_gate`.** Wrong semantics; would pollute abstain-rate
  monitoring, which B17 uses to detect audio problems.

## Consequences

- B9 fusion can exclude untrained heads explicitly and report them.
- B13 UI can show "head not trained" instead of "insufficient audio".
- B17 should alert if `untrained` appears in production traffic.
- Additive enum value: old consumers that validate against the previous enum
  will reject it — all in-repo consumers are updated in the same change.

## Blocks affected

B0 (contract), B4, B5 (emitters), B9, B13, B17 (consumers).
