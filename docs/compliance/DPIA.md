# Data Protection Impact Assessment: VoiceGuard (template + completed example)

> **Status:** completed example for the reference deployment (an Indian bank's contact centre,
> on-prem). **This is an engineering draft, not legal advice.** It must be reviewed and signed
> by the controller's Data Protection Officer / legal counsel before production traffic.
> Jurisdictions considered: India DPDP Act 2023 (+ RBI data-localisation directions for payment
> and banking data), EU GDPR (Art. 35), California CCPA/CPRA.

How to use: copy this file per tenant deployment. Keep sections 1–9. Replace the example text in
*italics* where the deployment differs, and re-run the compliance suite (`pytest -m compliance`)
as evidence for section 6.

---

## 1. Processing description

| Item | Reference deployment |
|---|---|
| Controller | *The bank (tenant)*. The VoiceGuard operator is a processor under a data processing agreement. |
| Purpose | Detect synthetic-voice (clone) impersonation and social-engineering patterns on live calls, and raise friction (banner, call-back, step-up). **Advisory only** (invariant I1): it never auto-declines a customer. |
| Data subjects | Customers and callers claiming to be customers; contact-centre agents (their voice is present on the call). |
| Personal data | Call audio (**transient, memory only**); per-window scores; codec and quality telemetry; redacted transcript snippets and intent labels; call metadata (caller number, claimed customer id); analyst feedback labels. |
| Special category / biometric | **Voiceprints** (speaker embeddings) for customers who explicitly enrolled. Biometric data under DPDP 2023, GDPR Art. 9 and CCPA. |
| Sources | Telephony/SIP streams of the tenant; the tenant's CRM (claimed identity); the customer (enrolment). |
| Recipients | Tenant agents and fraud analysts (UI); the tenant's own systems (signed webhooks). **No third parties, no external APIs** (I6). |
| Location | On-prem at the tenant, in India. No offshore processing (`config/privacy/tenants/*.yaml: region: IN, central_processing: false`). |
| Retention | Raw audio: none (default). Evidence bundles 365 d, score timeline 90 d, feedback 365 d, voiceprints until withdrawal / 730 d inactive (`config/privacy/tenants/default.yaml`). *Values need DPO confirmation (PROJECT_STATUS Q10).* |

Data flow (all inside the tenant network):

```
SIP/RTP ─▶ capture (B1) ─▶ conditioner (B2) ─▶ in-memory window store (wiped after use)
                                                  │
                             heads A–F (B4–B8) ◀──┘   ASR + intent (B10, local model)
                                   │                          │
                              fusion (B9) ── scores only ─────┴─▶ policy (B11) ─▶ agent UI (B13)
                                   │                                   │
                        score timeline DB                 evidence bundle DB (no audio, hash chain)
```

## 2. Lawful basis (consent matrix)

Encoded as configuration in `config/privacy/consent_matrix.yaml` and enforced in code (the
gateway refuses a session whose basis does not permit `fraud_detection`: HTTP 403 / gRPC
PERMISSION_DENIED).

| Purpose | Basis | Consent prompt | Bundling | Notes |
|---|---|---|---|---|
| Fraud detection on the call | Legitimate use (DPDP §7) / legitimate interests (GDPR 6(1)(f)) | No interruptive prompt | allowed | Must be in the privacy notice |
| Transcript intent analysis | as above | No | allowed | Redacted before storage (B10) |
| Speaker enrolment (voiceprint) | **Explicit, verifiable consent** + notice | Yes | **never bundled** | `consent_ref` stored with the voiceprint; enrolment without it is refused |
| Speaker verification | Relies on the enrolment consent | No | allowed | Only against voiceprints enrolled with consent |
| Raw audio retention (flagged calls) | Legitimate use, only if the tenant enables it | No | allowed | Off by default; encrypted; short window |
| Model improvement | Explicit consent | Yes | never | Scores/labels only by default |
| Marketing / analytics | — | — | never | **Disabled in this product** |

## 3. Necessity and proportionality

- **Why audio:** voice-clone fraud can only be detected from the voice. Content-only signals
  (intent) are used in addition, not instead (layered defence).
- **Minimisation:** raw audio is processed in memory in 3 s windows and overwritten after
  scoring (`packages/vg_core/sample_store.py`). Only scores, versions and redacted text are persisted.
- **No automated decision:** outputs are advisory. Actions add friction (call-back, step-up) and
  never deny service on their own (I1). `ABSTAIN` is shown distinctly and never implies "genuine".
- **Shadow mode first:** new tenants start in shadow mode (B11-T07), so thresholds are tuned on
  live traffic before any alert reaches an agent.
- **Alternatives considered:** a cloud detection API was rejected (data residency, I6). Keeping
  all call recordings for later analysis was rejected as disproportionate.

## 4. Risks to individuals

| # | Risk | Likelihood | Severity |
|---|---|---|---|
| R1 | A genuine customer is flagged as synthetic (false positive) → friction, embarrassment, denial of a transaction | Medium | Medium |
| R2 | Accent / language / gender bias: false positives concentrate on some groups | Medium | High |
| R3 | Breach of voiceprints (irrevocable biometric) | Low | High |
| R4 | Function creep: audio or transcripts reused for marketing or surveillance | Low | High |
| R5 | Raw audio persisted by accident (logs, temp files, debug dumps) | Low | High |
| R6 | Offshore transfer through a cloud API or LLM | Low | High |
| R7 | Excess retention of evidence / timelines | Medium | Medium |
| R8 | Data subjects cannot get access or erasure | Low | Medium |
| R9 | Over-reliance by agents on the score (automation bias) | Medium | Medium |

## 5. Mitigations

| Risk | Mitigation | Where |
|---|---|---|
| R1, R9 | Advisory-only copy; abstain state; call-back rather than decline; analyst feedback loop; calibrated thresholds from a cost model; shadow mode | B11, B13, I1 |
| R2 | Per-gender / per-language FPR gaps in every benchmark report; fairness release gate blocks promotion of a failing model | B15 `fairness.py`, B17 registry |
| R3 | AES-256-GCM per-tenant keys (envelope encryption, rotation, crypto-shredding on offboarding); consent reference required; deletion API | B7 vault, B16 `keys.py` |
| R4 | Purpose limitation in the consent matrix (unknown purposes refused, marketing disabled); audit log of every privacy action | B16 `policy.py`, `audit.py` |
| R5 | Memory-only processing with buffer overwrite; feature-only logging processor strips audio-like fields; automated test scans every written file for audio | B16 `feature_log.py`, `tests/compliance/` |
| R6 | Local ASR / LLM; egress guard (default-deny host allowlist, audio payload blocking, offshore refusal without opt-in) | B10, B16 `egress.py`, `DATA_RESIDENCY.md` |
| R7 | Retention job per tenant policy; evidence bodies erased while the hash chain proves what existed; legal holds explicit | B16 `retention.py` |
| R8 | DSAR export and erasure by customer id / caller number / speaker id; audited with hashed subject refs | B16 `retention.py` (DSAR), privacy API |

## 6. Evidence of controls (re-run before sign-off)

```bash
pytest -m compliance -q
python -m services.privacy.main retention --tenants <tenant> --dry-run
```

The suite covers: no raw PCM at rest on the default path (every file written during an
end-to-end run is scanned), buffers overwritten, retention erasure keeps a verifiable chain,
legal holds respected, consent enforced per purpose (gateway 403 without a lawful basis,
enrolment without a consent reference refused), DSAR export/erasure scoped to one subject,
audit log tamper-evident.

## 7. Residual risk

| Risk | Residual | Rationale |
|---|---|---|
| R1/R2 | **Medium** until the fairness gate threshold is set (Q9) and measured on the tenant's own traffic in shadow mode | Cannot be closed by engineering alone |
| R3 | Low | Encryption and consent controls; key custody depends on the tenant's KMS |
| R5, R6 | Low | Enforced and tested |
| R7, R8 | Low | Automated; values pending DPO confirmation (Q10) |

## 8. Consultation

*DPO: ___. InfoSec: ___. Contact-centre operations: ___. Consultation with data subjects or their
representatives: customer-panel review of the privacy-notice wording (planned).*

## 9. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Data Protection Officer | | ☐ approve ☐ approve with conditions ☐ reject | |
| CISO | | | |
| Product owner | | | |

Review this DPIA again on: any new processing purpose, a new model family, a change of hosting
location, or 12 months after sign-off.

---

## Appendix A: audit-log schema

Privacy-relevant actions are written to an append-only, hash-chained audit log
(`services/privacy/audit.py`). Events follow `services/privacy/audit_event.schema.json`:
`at, action, actor, tenant_id, purpose, lawful_basis, subject_ref (hashed), outcome, details`.
Actions: retention_run, evidence_erased, dsar_export, dsar_erasure, voiceprint_deleted,
key_created, key_rotated, key_shredded, consent_refused, raw_audio_retained, raw_audio_purged,
egress_denied, model_promoted, model_rolled_back.
