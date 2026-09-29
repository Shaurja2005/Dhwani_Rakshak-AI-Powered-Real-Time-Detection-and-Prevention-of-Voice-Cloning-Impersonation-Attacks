# VoiceGuard: submission write-up (B18-T06)

## The problem

Voice cloning has turned "I recognise your voice" into a vulnerability. A few seconds of
public audio is enough to impersonate a customer calling their bank, or an executive
instructing a transfer. Indian contact centres face two extra problems:

- 12+ languages, often code-switched;
- narrowband, heavily compressed telephony that destroys the artefacts most detectors rely on.

## What we built

A real-time, on-premises system that scores each live call with five independent defence
layers. It fuses them into one calibrated risk score and turns that score into plain
instructions for the agent: a caution banner, a call-back on the registered number, a
liveness challenge. See `docs/ARCHITECTURE.md`.

## Two things we want to say out loud

> **1. Detection is advisory, never authoritative.** Genuine and synthetic speech overlap under
> realistic conditions, so any detector has a non-zero error floor. VoiceGuard raises friction
> and triggers out-of-band verification. It must never be the sole gate on a transaction.

> **2. The strongest defences are partly non-acoustic.** Speaker verification,
> transaction-context anomaly detection, transcript-level intent analysis and an enforced
> call-back all degrade gracefully when the audio detector is fooled. A system with five
> layers beats one with a slightly better EER on one.

## Evidence

- **Detection accuracy:** only as reported in `docs/benchmarks/REPORT.md`. That covers
  leave-one-generator-out, cross-dataset, per language and accent, fairness (per-group
  false-positive gaps), codec and SNR sweeps, and laundering and adversarial attacks against
  our own detector. We publish the degradation on unseen generators and cross-dataset data
  because it is the honest measure of field performance.
- **Latency and capacity:** `docs/benchmarks/LOADTEST.md`, measured on named hardware and
  cited by row id.
- **Privacy and compliance:**
  - `docs/compliance/DPIA.md`;
  - `docs/compliance/DATA_RESIDENCY.md`;
  - the automated compliance suite (`pytest -m compliance`), which covers no raw audio at rest,
    retention, consent per purpose, and DSAR.

## What is not finished (honest status)

Detection heads need their training data and GPU runs, which are listed in
`docs/SETUP_PENDING.md`. Until then:

- they **abstain** with reason `untrained` rather than emit guesses;
- the benchmark report contains no accuracy numbers;
- the latency figures use a model of the same shape as the planned student.

Everything else runs end to end today:
- capture, conditioning, fusion, intent, policy, UI, API and SDKs;
- serving, evaluation harness, privacy controls and deployment.

## Responsible use

We clone only voices whose owners signed a consent form (`docs/ETHICS.md`, enforced in code).
The system never stores call audio by default, never sends data to external services, and
presents its output as advice to a human.
