# VoiceGuard architecture (B18-T04)

Real-time, **advisory** detection of voice-clone impersonation on phone and VoIP calls,
built as five independent defence layers fused into one calibrated risk score. It runs
entirely on the bank's premises: no audio, transcript or embedding leaves (I6).

```mermaid
flowchart LR
  subgraph Capture["Capture (B1)"]
    PBX[PBX / SBC / Twilio / WebRTC] --> ING[ingest adapters]
  end
  subgraph Condition["Conditioning (B2)"]
    ING --> DEC[decode + resample 16 kHz] --> WIN[3 s windows / 1 s hop]
    WIN --> VAD[VAD + quality gate]
  end
  VAD --> STORE[(in-memory window store<br/>wiped after use, I5)]
  subgraph Heads["Detection heads (B4–B8)"]
    STORE --> A[A · SSL anti-spoof<br/>XLS-R + Nes2Net, batched B14]
    STORE --> B[B · DSP + room / codec forensics]
    STORE --> C[C · prosody, breath, rhythm]
    STORE --> D[D · speaker verification<br/>encrypted voiceprints]
    STORE --> E[E · liveness challenge]
    STORE --> F[F · watermark]
  end
  subgraph Context["Context (B10, local only)"]
    ASR[local ASR] --> INT[intent + redaction]
    META[call metadata + transaction risk]
  end
  A & B & C & D & E & F --> FUS[fusion + calibration + temporal model (B9)]
  FUS --> POL[policy engine (B11)<br/>tiered thresholds, shadow mode]
  INT --> POL
  META --> POL
  POL --> UI[agent / analyst UI (B13)]
  POL --> EVD[(evidence bundles<br/>hash chain, no audio)]
  POL --> WH[signed webhooks]
  GW[API gateway (B12)<br/>REST · WebSocket · gRPC · SDKs] -.-> ING
  subgraph Ops["Operations (B14–B17)"]
    INF[batching inference / Triton] --- A
    OBS[Prometheus metrics · drift monitor] --- FUS
    REG[model registry<br/>blue/green + rollback] --- INF
    PRIV[privacy service<br/>retention · DSAR · keys · audit] --- EVD
  end
  EVAL[evaluation harness (B15)<br/>REPORT.md] --- REG
```

## The five layers, and why there are five

| Layer | What it catches | What fools it | Block |
|---|---|---|---|
| Synthetic-voice detection (heads A, B, C) | TTS / voice-conversion artefacts, codec and room inconsistencies, unnatural prosody | Unseen generators, heavy channel degradation | B4–B6 |
| Speaker verification (head D) | The voice is not the enrolled customer | A good clone *of that customer* | B7 |
| Liveness challenge (head E) | Replayed or pre-generated audio | Real-time voice conversion with a fast operator | B8 |
| Conversation intent (B10) | Urgency, secrecy, OTP and payment requests, callback resistance | A patient, careful fraudster | B10 |
| Policy + call-back (B11) | Anything risky: verification moves to an independent channel | Nothing at the voice level. It's a process control. | B11 |

A detector that has been fooled still leaves the other layers standing. Detection is advisory:
the output is *friction* (banner, call-back, step-up), never an automatic decline.

## Contracts and invariants

- Data contracts: `schemas/*.json`, `proto/voiceguard.proto` (SOURCE_OF_TRUTH §4). Every head
  returns a `HeadScore` or abstains with a reason. ABSTAIN is first-class and never means "genuine".
- Invariants I1–I10 (SOURCE_OF_TRUTH §2), in brief:
  - advisory only;
  - no raw audio at rest;
  - no external APIs;
  - licence lineage enforced;
  - consented cloning only;
  - never block the call path.

## Where things run

- **Single node:** `deploy/docker-compose.prod.yml` (`make prod-up`).
- **Kubernetes:** `deploy/k8s` (`make k8s-render`).
- **Topology and data residency:** `docs/compliance/DATA_RESIDENCY.md`.
- **Latency and capacity:** `docs/benchmarks/LOADTEST.md`.
- **Accuracy:** `docs/benchmarks/REPORT.md`, the only citable source of numbers.
