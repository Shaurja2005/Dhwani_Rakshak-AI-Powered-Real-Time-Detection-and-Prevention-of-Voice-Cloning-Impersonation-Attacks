# ADR 0008 — Add `credential_request` and `payment_request` intent labels

**Status:** Accepted (2026-09-17)

## Context

`schemas/context_signals.schema.json` restricted `intent_labels` to the six
coercion cues named in IMPLEMENTATION_PLAN B10-T03. The B10 intent classifier
also detects two further cues that are among the strongest and most common in
Indian banking fraud calls:

* **credential_request** — asking the customer or agent for an OTP, PIN, CVV or password;
* **payment_request** — an explicit instruction to move money (including gift cards / crypto).

The proto (`repeated string`) and Pydantic model (`list[str]`) already accepted
any string, so the drift was only caught when B11 validated evidence bundles
against the JSON schema — which rejected real B10 output.

## Decision

Add both labels to the `intent_labels` enum in the JSON schema. The label set is
now defined in one place in code (`services/context/intent.py::LABELS`), and a
test asserts the schema enum equals that tuple so they cannot drift again.

## Alternatives considered

- **Drop the two labels.** Loses the strongest single fraud cue (OTP requests).
- **Map them onto existing labels** (e.g. credential_request → secrecy_demand).
  Wrong semantics; misleads analysts and the UI.
- **Remove the enum.** Loses contract validation of a field the UI keys off.

## Consequences

- Additive enum change; consumers validating against the old schema must update.
- B13 UI must render labels for both (plain-English hints exist in B11 `actions.py`).

## Blocks affected

B0 (schema), B10 (emitter), B11 (evidence validation), B13 (UI).
