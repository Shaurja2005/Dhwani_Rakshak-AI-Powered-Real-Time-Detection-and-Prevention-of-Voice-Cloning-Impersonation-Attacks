# PROJECT STATUS — VoiceGuard

**Board version:** 0.1.0 · **Last updated:** — · **Updated by:** —

> This is the shared task board. Humans and agents both write here. Read `AGENTS.md` §3 for the exact update protocol before editing.
>
> **Rules:** never delete a row · never renumber a task ID · never mark `DONE` without the Definition of Done in `IMPLEMENTATION_PLAN.md` being satisfied and an artifact path recorded · always append to §5 Changelog in the same commit.

---

## 1. Status legend

| Status | Meaning |
|---|---|
| `TODO` | Not started |
| `WIP` | In progress, owner assigned |
| `BLOCKED` | Cannot proceed — the blocker MUST be named in Notes |
| `REVIEW` | Complete, awaiting review/merge |
| `DONE` | Merged, DoD satisfied, artifact recorded |
| `DROPPED` | Deliberately not doing — reason required in Notes |

## 2. Milestone rollup

| Milestone | Target | Status | % |
|---|---|---|---|
| M0 — Walking skeleton runs end to end | — | TODO | 0% |
| M1 — Real detection head + speaker verification in the pipeline | — | TODO | 0% |
| M2 — Indic + telephony corpus built, before/after chart produced | — | TODO | 0% |
| M3 — Context/intent + policy + UI complete | — | TODO | 0% |
| M4 — Eval report generated, all claims traceable | — | TODO | 0% |
| M5 — Deployable stack + compliance suite passing | — | TODO | 0% |
| M6 — Demo rehearsed with offline fallback | — | TODO | 0% |

## 3. Block rollup

| Block | Title | Owner | Status | Done / Total |
|---|---|---|---|---|
| B0 | Repo, contracts, CI | agent-antigravity | DONE | 9/9 |
| B1 | Capture adapters | agent-antigravity | DONE | 9/9 |
| B2 | Stream conditioning | — | TODO | 0/8 |
| B3 | Data & corpus engineering | — | TODO | 0/11 |
| B4 | Head A — SSL anti-spoof | — | TODO | 0/10 |
| B5 | Head B — DSP & scene | — | TODO | 0/6 |
| B6 | Head C — prosody | — | TODO | 0/6 |
| B7 | Head D — speaker verification | — | TODO | 0/6 |
| B8 | Heads E/F — liveness & watermark | — | TODO | 0/6 |
| B9 | Fusion & risk engine | — | TODO | 0/8 |
| B10 | Context & intent | — | TODO | 0/7 |
| B11 | Policy & alerting | — | TODO | 0/8 |
| B12 | APIs & SDKs | — | TODO | 0/9 |
| B13 | Agent/analyst UI | — | TODO | 0/7 |
| B14 | Serving & optimization | — | TODO | 0/6 |
| B15 | Evaluation harness | — | TODO | 0/10 |
| B16 | Privacy & compliance | — | TODO | 0/9 |
| B17 | Deployment & ops | — | TODO | 0/8 |
| B18 | Demo & submission | — | TODO | 0/6 |

## 4. Task board

Columns: **ID · Task · Owner · Status · Depends on · Artifact · Notes**
Artifact = the path, PR, or report that proves the task is done.

### B0 — Repo, contracts, CI
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B0-T01 | Monorepo scaffold | agent-antigravity | DONE | — | bootstrap_repo.sh | |
| B0-T02 | proto + generated stubs | agent-antigravity | DONE | B0-T01 | proto/voiceguard.proto | |
| B0-T03 | JSON Schemas + Pydantic models | agent-antigravity | DONE | B0-T01 | schemas/*.json | |
| B0-T04 | vg_core: config, logging, tracing | agent-antigravity | DONE | B0-T01 | packages/vg_core/ | |
| B0-T05 | DetectionHead ABC + StubHead | agent-antigravity | DONE | B0-T03 | packages/vg_core/head_api.py | unblocks B9/B11/B13 |
| B0-T06 | docker-compose dev stack | agent-antigravity | DONE | B0-T01 | deploy/docker-compose.dev.yml | |
| B0-T07 | CI: lint, types, tests, schema-compat | agent-antigravity | DONE | B0-T03 | .github/workflows/ci.yml | |
| B0-T08 | Makefile targets | agent-antigravity | DONE | B0-T06 | Makefile | |
| B0-T09 | Seed SoT / STATUS / AGENTS docs | agent-antigravity | DONE | B0-T01 | PROJECT_STATUS.md | |

### B1 — Capture adapters
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B1-T01 | WAV replay adapter (real-time + jitter/loss) | agent-antigravity | DONE | B0-T04 | services/ingest/adapters/replay_wav.py | |
| B1-T02 | Generic WebSocket PCM adapter | agent-antigravity | DONE | B1-T01 | services/ingest/adapters/websocket_pcm.py | |
| B1-T03 | Asterisk AudioSocket adapter + configs | agent-antigravity | DONE | B1-T02 | services/ingest/adapters/asterisk_audiosocket.py | |
| B1-T04 | FreeSWITCH mod_audio_stream adapter | agent-antigravity | DONE | B1-T02 | services/ingest/adapters/freeswitch_ws.py | |
| B1-T05 | Twilio Media Streams adapter | agent-antigravity | DONE | B1-T02 | services/ingest/adapters/twilio_stream.py | 8k µ-law |
| B1-T06 | SIPREC receiver | agent-antigravity | DONE | B1-T03 | services/ingest/adapters/siprec.py | stretch |
| B1-T07 | Browser/WebRTC capture | agent-antigravity | DONE | B1-T02 | services/ingest/adapters/websocket_pcm.py | via WS adapter |
| B1-T08 | Metadata envelope extraction | agent-antigravity | DONE | B0-T03 | services/ingest/metadata.py | |
| B1-T09 | Session lifecycle + orphan reaper | agent-antigravity | DONE | B1-T01 | services/ingest/session.py | |

### B2 — Stream conditioning
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B2-T01 | Jitter buffer + PLC | — | TODO | B1-T01 | | |
| B2-T02 | Resampler incl. 8k µ-law/A-law | — | TODO | B1-T01 | | |
| B2-T03 | VAD integration with hysteresis | — | TODO | B2-T02 | | Silero |
| B2-T04 | Windowing 3.0s/1.0s | — | TODO | B2-T03 | | |
| B2-T05 | **Quality gate / abstain logic** | — | TODO | B2-T04 | | highest FP-reduction value |
| B2-T06 | Loudness normalization + pre-norm logging | — | TODO | B2-T02 | | |
| B2-T07 | Music/tone/IVR detector | — | TODO | B2-T03 | | |
| B2-T08 | Redis per-window feature cache | — | TODO | B2-T04 | | |

### B3 — Data & corpus engineering
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B3-T01 | Dataset registry with license field | — | TODO | B0-T01 | | |
| B3-T02 | Downloaders: core anti-spoof corpora | — | TODO | B3-T01 | | start EULA requests day 1 |
| B3-T03 | Downloaders: Indic corpora | — | TODO | B3-T01 | | |
| B3-T04 | Unified manifest schema + writer | — | TODO | B3-T01 | | |
| B3-T05 | Clone zoo containers (TTS + VC) | — | TODO | B3-T04 | | 1 image per generator |
| B3-T06 | Batch cloning job | — | TODO | B3-T05 | | GPU-heavy |
| B3-T07 | Channel destruction simulator | — | TODO | B3-T04 | | |
| B3-T08 | **Symmetry test in CI** | — | TODO | B3-T07 | | invariant I3 |
| B3-T09 | License gate in data loader | — | TODO | B3-T01 | | invariant I7 |
| B3-T10 | Speaker/generator-disjoint splits | — | TODO | B3-T04 | | |
| B3-T11 | Consent + ethics register | — | TODO | — | | invariant I9 |

### B4 — Head A (SSL anti-spoof)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B4-T01 | Stage 0: reproduce published baseline | — | TODO | B3-T02 | | sanity gate |
| B4-T02 | SSL front-end wrapper + layer selection | — | TODO | B4-T01 | | layers 5–9 |
| B4-T03 | Nes2Net back-end (+AASIST secondary) | — | TODO | B4-T02 | | |
| B4-T04 | OC-Softmax / AM-Softmax loss | — | TODO | B4-T03 | | |
| B4-T05 | RawBoost + channel aug in train loop | — | TODO | B3-T07 | | |
| B4-T06 | Stage 1 training run | — | TODO | B4-T04 | | |
| B4-T07 | Stage 2 multi-corpus pooling | — | TODO | B4-T06, B3-T06 | | balance by family |
| B4-T08 | Stage 3 robustness techniques | — | TODO | B4-T07 | | |
| B4-T09 | Real-time inference wrapper | — | TODO | B4-T06 | | budget + timeout abstain |
| B4-T10 | Continual-learning update procedure | — | TODO | B4-T08 | | |

### B5 — Head B (DSP & scene)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B5-T01 | CQT/LFCC/MGD/LTAS/bicoherence extractors | — | TODO | B2-T04 | | |
| B5-T02 | Vocoder/NAC fingerprinting | — | TODO | B5-T01 | | |
| B5-T03 | GBDT/CNN classifier | — | TODO | B5-T01, B3-T04 | | |
| B5-T04 | RIR / scene-consistency analyzer | — | TODO | B5-T01 | | distinctive component |
| B5-T05 | Codec chain identifier | — | TODO | B5-T01 | | |
| B5-T06 | Human-readable explanations | — | TODO | B5-T03 | | |

### B6 — Head C (prosody)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B6-T01 | F0 dynamics, jitter, shimmer | — | TODO | B2-T04 | | |
| B6-T02 | Breath-group detection | — | TODO | B6-T01 | | |
| B6-T03 | Pause distribution & rate variance | — | TODO | B6-T01 | | |
| B6-T04 | Disfluency detection from ASR | — | TODO | B10-T01 | | |
| B6-T05 | Temporal model + calibrated score | — | TODO | B6-T03 | | |
| B6-T06 | Indic prosody normalization + per-language eval | — | TODO | B6-T05, B15-T04 | | FP risk on retroflex/code-switch |

### B7 — Head D (speaker verification)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B7-T01 | Embedding model benchmark & selection | — | TODO | B2-T04 | | test on Indian-accented audio |
| B7-T02 | Enrollment service API | — | TODO | B0-T03 | | |
| B7-T03 | Encrypted voiceprint vault | — | TODO | B7-T02, B16-T07 | | biometric data |
| B7-T04 | Cosine scoring + AS-norm | — | TODO | B7-T01 | | |
| B7-T05 | Channel compensation for enrollment | — | TODO | B7-T04, B3-T07 | | |
| B7-T06 | No-enrollment abstain path | — | TODO | B7-T04 | | invariant I2 |

### B8 — Heads E/F (liveness & watermark)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B8-T01 | Challenge generator (code-switched nonce) | — | TODO | B0-T05 | | |
| B8-T02 | Response verifier (ASR match + latency) | — | TODO | B8-T01, B10-T01 | | |
| B8-T03 | Operator-triggered challenge hook | — | TODO | B8-T01, B13-T03 | | |
| B8-T04 | AudioSeal watermark detector | — | TODO | B2-T04 | | |
| B8-T05 | Additional commercial watermark detectors | — | TODO | B8-T04 | | |
| B8-T06 | Enforce absence-is-neutral rule | — | TODO | B8-T04, B9-T02 | | invariant I4 |

### B9 — Fusion & risk engine
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B9-T01 | Per-head calibration | — | TODO | B0-T05 | | |
| B9-T02 | Cross-head fusion (LR/GBDT) + abstain handling | — | TODO | B9-T01 | | |
| B9-T03 | Temporal fusion (Bayesian / HMM) | — | TODO | B9-T02 | | |
| B9-T04 | Any-segment trigger for partial splicing | — | TODO | B9-T03 | | |
| B9-T05 | Four-state emission incl. ABSTAIN | — | TODO | B9-T03 | | |
| B9-T06 | Cost-model operating point selection | — | TODO | B9-T01, B15-T09 | | not EER |
| B9-T07 | Per-head contribution attribution | — | TODO | B9-T02 | | |
| B9-T08 | Score timeline persistence + replay | — | TODO | B9-T05 | | |

### B10 — Context & intent
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B10-T01 | Streaming ASR (IndicConformer/Whisper) | — | TODO | B2-T04 | | |
| B10-T02 | Language ID + code-switch detection | — | TODO | B10-T01 | | |
| B10-T03 | Social-engineering intent classifier (local LLM) | — | TODO | B10-T01 | | invariant I6 |
| B10-T04 | Metadata risk scorer | — | TODO | B1-T08 | | |
| B10-T05 | Transaction context connector (mock core banking) | — | TODO | B0-T03 | | |
| B10-T06 | ContextSignals assembly | — | TODO | B10-T03, B10-T04 | | |
| B10-T07 | PII redaction on transcripts | — | TODO | B10-T01 | | |

### B11 — Policy & alerting
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B11-T01 | Rules engine + tenant threshold profiles | — | TODO | B9-T05 | | |
| B11-T02 | Action library | — | TODO | B11-T01 | | |
| B11-T03 | Transaction-sensitivity tiered thresholds | — | TODO | B11-T01, B10-T05 | | |
| B11-T04 | Evidence bundle (immutable, hashed) | — | TODO | B9-T08 | | |
| B11-T05 | Multi-channel notification + SIEM webhook | — | TODO | B11-T02 | | |
| B11-T06 | Pre-transaction warning prompts | — | TODO | B11-T02 | | |
| B11-T07 | Shadow mode (default for new tenants) | — | TODO | B11-T01 | | |
| B11-T08 | Analyst feedback loop → eval + replay buffer | — | TODO | B11-T04, B13-T06 | | |

### B12 — APIs & SDKs
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B12-T01 | gRPC bidirectional streaming | — | TODO | B0-T02 | | |
| B12-T02 | REST API surface | — | TODO | B0-T03 | | |
| B12-T03 | Signed webhooks + retry | — | TODO | B11-T05 | | |
| B12-T04 | AuthN/AuthZ, quotas, rate limits | — | TODO | B12-T02 | | |
| B12-T05 | Python SDK | — | TODO | B12-T01 | | |
| B12-T06 | JS/TS SDK | — | TODO | B12-T01 | | |
| B12-T07 | Edge/mobile SDK stub (ONNX/TFLite) | — | TODO | B14-T03 | | privacy story |
| B12-T08 | OpenAPI docs + 10-minute quickstart | — | TODO | B12-T02 | | |
| B12-T09 | Reference connectors | — | TODO | B1-T03, B1-T05 | | |

### B13 — UI
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B13-T01 | Live call view (gauge, timeline, head bars) | — | TODO | B0-T05 | | build on stub stream |
| B13-T02 | Live transcript with intent highlights | — | TODO | B10-T06 | | |
| B13-T03 | Alert banner + one-click step-up | — | TODO | B11-T02 | | |
| B13-T04 | Post-call forensics view + PDF export | — | TODO | B11-T04 | | |
| B13-T05 | Admin console (profiles, enrollment, shadow) | — | TODO | B11-T01, B7-T02 | | |
| B13-T06 | Analyst feedback buttons | — | TODO | B13-T03 | | |
| B13-T07 | Demo mode (genuine vs cloned side by side) | — | TODO | B13-T01 | | |

### B14 — Serving & optimization
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B14-T01 | Distill ensemble → student model | — | TODO | B4-T08 | | |
| B14-T02 | INT8 quantization + accuracy delta report | — | TODO | B14-T01 | | publish the cost |
| B14-T03 | ONNX export + parity check | — | TODO | B14-T01 | | |
| B14-T04 | Triton (server) + TFLite/ORT-Mobile (edge) | — | TODO | B14-T03 | | |
| B14-T05 | Load test + cost per 1k call-minutes | — | TODO | B14-T04 | | |
| B14-T06 | Backpressure & graceful degradation | — | TODO | B14-T04 | | invariant I10 |

### B15 — Evaluation harness
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B15-T01 | Harness skeleton (adopt AUDDT/DeepFense) | — | TODO | B3-T04 | | week 1 |
| B15-T02 | Leave-one-generator-out protocol | — | TODO | B15-T01 | | |
| B15-T03 | Cross-dataset protocol | — | TODO | B15-T01 | | report degradation honestly |
| B15-T04 | Per-language / per-accent breakdown | — | TODO | B15-T01, B3-T03 | | key chart |
| B15-T05 | Fairness: per-gender/per-language FPR gaps | — | TODO | B15-T04 | | release gate |
| B15-T06 | Per-codec / per-SNR curves | — | TODO | B15-T01, B3-T07 | | |
| B15-T07 | Operational metrics | — | TODO | B14-T05 | | |
| B15-T08 | Adversarial / laundering robustness | — | TODO | B15-T01 | | |
| B15-T09 | Metrics: EER, t-DCF, a-DCF, pAUC, ECE | — | TODO | B15-T01 | | |
| B15-T10 | Auto-generated benchmark report in CI | — | TODO | B15-T09 | | every claim traces here |

### B16 — Privacy & compliance
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B16-T01 | Ephemeral processing + no-raw-audio-at-rest test | — | TODO | B2-T04 | | invariant I5 |
| B16-T02 | Feature-only logging | — | TODO | B16-T01 | | |
| B16-T03 | On-prem/edge default topology | — | TODO | B17-T02 | | invariant I6 |
| B16-T04 | Consent matrix as configuration | — | TODO | B1-T08 | | |
| B16-T05 | Data residency documentation & enforcement | — | TODO | B10-T03 | | RBI |
| B16-T06 | Retention/erasure automation + DSAR | — | TODO | B16-T02 | | |
| B16-T07 | Per-tenant keys + encrypted vault | — | TODO | B0-T04 | | |
| B16-T08 | DPIA template + completed example | — | TODO | B16-T04 | | |
| B16-T09 | Model cards + data statements | — | TODO | B15-T10 | | |

### B17 — Deployment & ops
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B17-T01 | Dockerfiles per service | — | TODO | B0-T06 | | |
| B17-T02 | Single-node on-prem compose | — | TODO | B17-T01 | | realistic BFSI start |
| B17-T03 | Helm / k8s manifests | — | TODO | B17-T01 | | |
| B17-T04 | Prometheus metrics | — | TODO | B17-T01 | | |
| B17-T05 | Grafana dashboards (ops + fraud) | — | TODO | B17-T04 | | |
| B17-T06 | Score-distribution drift monitoring | — | TODO | B17-T04, B9-T08 | | new-generator early warning |
| B17-T07 | Model registry + blue/green rollout | — | TODO | B14-T04 | | |
| B17-T08 | On-call runbook | — | TODO | B17-T05 | | |

### B18 — Demo & submission
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B18-T01 | Live clone-the-judge demo script | — | TODO | M1 | | consent required |
| B18-T02 | Human-reading-fraud-script demo | — | TODO | B10-T03 | | layered-defense proof |
| B18-T03 | Before/after Indic EER chart | — | TODO | B15-T04 | | headline chart |
| B18-T04 | Architecture diagram, README, video | — | TODO | — | | |
| B18-T05 | Offline demo fallback | — | TODO | B18-T01 | | assume wifi fails |
| B18-T06 | Framing statements in the writeup | — | TODO | — | | advisory-not-authoritative |

## 5. Changelog

Append one line per status change. Newest at the top. Never edit an existing line.

```
YYYY-MM-DDTHH:MMZ | <agent-or-human> | <TASK-ID> | <OLD> -> <NEW> | <artifact/PR> | <one-line note>
```

| When | Who | Task | Change | Artifact | Note |
|---|---|---|---|---|---|
| 2026-08-30T16:47Z | agent-antigravity | B1-T01-09 | WIP -> DONE | services/ingest/ | 36 tests pass, 79% cov |
| 2026-08-30T10:35Z | agent-antigravity | B0-T01-09 | TODO -> DONE | vg_core/, schemas/ | Scaffolded repo, contracts, pipelines |
| — | — | — | board created | — | initial seed |

## 6. Open questions

Anything that needs a human decision. Agents append here rather than guessing.

| # | Question | Raised by | Blocks | Answer |
|---|---|---|---|---|
| Q1 | Which datasets have we actually been granted access to? | — | B3, B4 | |
| Q2 | Target deployment hardware for the latency claim (T4? L4? CPU-only?) | — | B14 | |
| Q3 | Commercial or research lineage for the primary demo model? | — | B3-T09 | |
| Q4 | Which telephony stack does the pilot tenant actually run? | — | B1 | |
| Q5 | Which Indic languages are in scope for v1 (all 12, or 3–4 done well)? | — | B3, B15 | |

## 7. Blocked items

| Task | Blocked since | Blocked by | Owner of the blocker | Escalation |
|---|---|---|---|---|
| — | — | — | — | — |
