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
| B2 | Stream conditioning | agent-antigravity | DONE | 8/8 |
| B3 | Data & corpus engineering | claude-b3 | WIP | 7/11 |
| B4 | Head A — SSL anti-spoof | claude-b4 | WIP | 3/10 |
| B5 | Head B — DSP & scene | claude-b5 | WIP | 4/6 |
| B6 | Head C — prosody | claude-b6 | WIP | 3/6 |
| B7 | Head D — speaker verification | claude-b7 | WIP | 4/6 |
| B8 | Heads E/F — liveness & watermark | claude-b8 | WIP | 2/6 |
| B9 | Fusion & risk engine | claude-b9 | WIP | 4/8 |
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
| B2-T01 | Jitter buffer + PLC | agent-antigravity | DONE | B1-T01 | packages/vg_audio/jitter_buffer.py | |
| B2-T02 | Resampler incl. 8k µ-law/A-law | agent-antigravity | DONE | B1-T01 | packages/vg_audio/resample.py, codecs.py | |
| B2-T03 | VAD integration with hysteresis | agent-antigravity | DONE | B2-T02 | packages/vg_audio/vad.py | Silero + Energy |
| B2-T04 | Windowing 3.0s/1.0s | agent-antigravity | DONE | B2-T03 | packages/vg_audio/windowing.py | |
| B2-T05 | **Quality gate / abstain logic** | agent-antigravity | DONE | B2-T04 | packages/vg_audio/quality.py | highest FP-reduction value |
| B2-T06 | Loudness normalization + pre-norm logging | agent-antigravity | DONE | B2-T02 | packages/vg_audio/quality.py | |
| B2-T07 | Music/tone/IVR detector | agent-antigravity | DONE | B2-T03 | packages/vg_audio/quality.py | Goertzel + HF ratio |
| B2-T08 | Redis per-window feature cache | agent-antigravity | DONE | B2-T04 | packages/vg_audio/features.py | log-mel, LFCC, MFCC, F0 |

### B3 — Data & corpus engineering
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B3-T01 | Dataset registry with license field | claude-b3 | DONE | B0-T01 | ml/data/registry.yaml, ml/data/registry.py | 22 corpora; unverified licenses default commercial_use:false |
| B3-T02 | Downloaders: core anti-spoof corpora | claude-b3 | REVIEW | B3-T01 | ml/data/download.py | tooling done; no corpus fetched yet, sha256 unpinned, EULAs pending (Q1) |
| B3-T03 | Downloaders: Indic corpora | claude-b3 | REVIEW | B3-T01 | ml/data/download.py | same as T02; sea_spoof/indic_codecfake/rtcfake URLs need confirming |
| B3-T04 | Unified manifest schema + writer | claude-b3 | DONE | B3-T01 | ml/data/manifest.py | + corpus report (make corpus-report) |
| B3-T05 | Clone zoo containers (TTS + VC) | claude-b3 | WIP | B3-T04 | ml/data/generators/, ml/data/generators.yaml | contract + xtts_v2 image written (not yet built); 12 images remain |
| B3-T06 | Batch cloning job | claude-b3 | REVIEW | B3-T05 | ml/data/clone_job.py, ml/data/degrade.py | planner + dry-run tested; no GPU run yet |
| B3-T07 | Channel destruction simulator | claude-b3 | DONE | B3-T04 | ml/data/channel/ | G.711, Opus, AMR-NB, GSM, G.722, loss, jitter, RIR, noise, AGC, NS. No G.729/EVS (no open encoder) |
| B3-T08 | **Symmetry test in CI** | claude-b3 | DONE | B3-T07 | ml/data/channel/test_symmetry.py | pass acc 0.48 vs 0.50 baseline; negative control catches leak at 0.90 |
| B3-T09 | License gate in data loader | claude-b3 | DONE | B3-T01 | ml/data/license_gate.py | also blocks eval-only corpora from training |
| B3-T10 | Speaker/generator-disjoint splits | claude-b3 | DONE | B3-T04 | ml/data/splits.py | speaker-disjoint + leave-generator-out |
| B3-T11 | Consent + ethics register | claude-b3 | DONE | — | ml/data/consent_register.yaml, ml/data/consent.py, docs/ETHICS.md | register empty: clone_job refuses everything until a human adds entries |

### B4 — Head A (SSL anti-spoof)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B4-T01 | Stage 0: reproduce published baseline | claude-b4 | BLOCKED | B3-T02 |  | deferred setup: needs ASVspoof 2019 LA + XLS-R weights + CUDA torch (docs/SETUP_PENDING.md) |
| B4-T02 | SSL front-end wrapper + layer selection | claude-b4 | REVIEW | B4-T01 | packages/vg_models/heads/head_a_ssl/frontend.py | HF path untested: transformers import broken in dev env; tiny front-end tested |
| B4-T03 | Nes2Net back-end (+AASIST secondary) | claude-b4 | DONE | B4-T02 | packages/vg_models/heads/head_a_ssl/backends.py, docs/adr/0001-nes2net-primary-backend.md | re-implementations; parity check is part of B4-T01 |
| B4-T04 | OC-Softmax / AM-Softmax loss | claude-b4 | DONE | B4-T03 | ml/training/losses.py | OC-Softmax + AM-Softmax |
| B4-T05 | RawBoost + channel aug in train loop | claude-b4 | DONE | B3-T07 | ml/training/augment.py | RawBoost 1-4 + B3 channel simulator, label-blind |
| B4-T06 | Stage 1 training run | claude-b4 | REVIEW | B4-T04 | ml/training/train_head_a.py, ml/training/configs/head_a_stage1.yaml | loop verified on synthetic smoke config; real run deferred |
| B4-T07 | Stage 2 multi-corpus pooling | claude-b4 | REVIEW | B4-T06, B3-T06 | ml/training/dataset.py, ml/training/configs/head_a_stage2.yaml | family-balanced sampler tested; 12 languages configured; run deferred |
| B4-T08 | Stage 3 robustness techniques | claude-b4 | WIP | B4-T07 | ml/training/robust.py, ml/training/configs/head_a_stage3.yaml | SAM, mixup, consistency, GRL, LoRA done; 2-front-end ensemble deferred to B9 |
| B4-T09 | Real-time inference wrapper | claude-b4 | REVIEW | B4-T06 | packages/vg_models/heads/head_a_ssl/head.py, packages/vg_core/sample_store.py | budget/timeout/never-raise tested; untrained mode abstains; p95 on target hw pending |
| B4-T10 | Continual-learning update procedure | claude-b4 | REVIEW | B4-T08 | ml/training/continual.py | EWC + replay buffer + procedure; not exercised on a real new family |

### B5 — Head B (DSP & scene)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B5-T01 | CQT/LFCC/MGD/LTAS/bicoherence extractors | claude-b5 | DONE | B2-T04 | packages/vg_models/heads/head_b_dsp/features.py | LFCC, CQT-approx, MGD, LTAS, roll-off, bicoherence, breath band; ~10 ms/window CPU incl. all B5 analysis |
| B5-T02 | Vocoder/NAC fingerprinting | claude-b5 | REVIEW | B5-T01 | packages/vg_models/heads/head_b_dsp/vocoder.py | upsampling tones, brick-wall edge, phase regularity; validated on synthetic artifacts only, needs real HiFi-GAN/EnCodec outputs |
| B5-T03 | GBDT/CNN classifier | claude-b5 | REVIEW | B5-T01, B3-T04 | packages/vg_models/heads/head_b_dsp/gbdt.py, ml/training/train_head_b.py, ml/training/configs/head_b.yaml | numpy GBDT (vectorised predict); trained on synthetic smoke data only |
| B5-T04 | RIR / scene-consistency analyzer | claude-b5 | DONE | B5-T01 | packages/vg_models/heads/head_b_dsp/scene.py | onset-based late-decay RT60; flags dry voice over RT60 0.4 s background incl. after G.711; RT60 >~0.7 s needs longer pauses than a 3 s window |
| B5-T05 | Codec chain identifier | claude-b5 | DONE | B5-T01 | packages/vg_models/heads/head_b_dsp/codec_id.py | wideband clean/compressed vs narrowband; cannot separate AMR from G.711 |
| B5-T06 | Human-readable explanations | claude-b5 | DONE | B5-T03 | packages/vg_models/heads/head_b_dsp/explain.py | 1-3 plain-English reasons, always includes channel |

### B6 — Head C (prosody)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B6-T01 | F0 dynamics, jitter, shimmer | claude-b6 | DONE | B2-T04 | packages/vg_models/heads/head_c_prosody/pitch.py | vectorised YIN, <2% F0 error 85-350 Hz; frame-level jitter/shimmer |
| B6-T02 | Breath-group detection | claude-b6 | DONE | B6-T01 | packages/vg_models/heads/head_c_prosody/breath_rhythm.py | inhalations before phrase onsets |
| B6-T03 | Pause distribution & rate variance | claude-b6 | DONE | B6-T01 | packages/vg_models/heads/head_c_prosody/breath_rhythm.py | pause/phrase/syllable-rate CV + over_regularity; acoustic filled-pause detector |
| B6-T04 | Disfluency detection from ASR | claude-b6 | REVIEW | B10-T01 | packages/vg_models/heads/head_c_prosody/disfluency.py | lexical fillers (en + 5 Indic), repetitions, restarts; not fed until B10 ASR exists |
| B6-T05 | Temporal model + calibrated score | claude-b6 | REVIEW | B6-T03 | packages/vg_models/heads/head_c_prosody/model.py, ml/training/train_head_c.py, ml/training/configs/head_c.yaml | BiGRU + global features; trained on synthetic smoke data only |
| B6-T06 | Indic prosody normalization + per-language eval | claude-b6 | REVIEW | B6-T05, B15-T04 | packages/vg_models/heads/head_c_prosody/normalize.py | Indic min-pause 220 ms, per-language z-norm, per-language FPR gate fails training run; real per-language eval needs data + B15 |

### B7 — Head D (speaker verification)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B7-T01 | Embedding model benchmark & selection | claude-b7 | REVIEW | B2-T04 | packages/vg_models/heads/head_d_speaker/embedders.py, packages/vg_models/heads/head_d_speaker/benchmark.py | ECAPA / WeSpeaker / TitaNet wrappers (lazy, not installed) + per-language EER-with-CI benchmark; selection needs Indian-accented data. MFCC baseline is plumbing-only (not a verifier) |
| B7-T02 | Enrollment service API | claude-b7 | DONE | B0-T03 | services/enrollment/core.py, services/enrollment/api.py, services/enrollment/main.py | REST: enrol/status/list/erase; quality gates, multi-session, consistency check, expiry, consent required. gRPC Enroll via B12 |
| B7-T03 | Encrypted voiceprint vault | claude-b7 | DONE | B7-T02, B16-T07 | packages/vg_models/heads/head_d_speaker/vault.py | AES-256-GCM per tenant key, tenant+speaker bound as AAD, embeddings only, key rotation, erasure; env key provider is dev-only (KMS in B16-T07) |
| B7-T04 | Cosine scoring + AS-norm | claude-b7 | REVIEW | B7-T01 | packages/vg_models/heads/head_d_speaker/scoring.py | cohort-centred cosine + AS-norm, calibration fit, EER/gap bootstrap CIs; needs real cohort + calibration data |
| B7-T05 | Channel compensation for enrollment | claude-b7 | DONE | B7-T04, B3-T07 | services/enrollment/core.py, packages/vg_models/heads/head_d_speaker/head.py | enrolment embedded through G.711 (+AMR-NB) channel; 8 kHz calls scored against narrowband centroid |
| B7-T06 | No-enrollment abstain path | claude-b7 | DONE | B7-T04 | packages/vg_models/heads/head_d_speaker/head.py | no claim / not enrolled / expired / embedder mismatch -> no_enrollment abstain |

### B8 — Heads E/F (liveness & watermark)
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B8-T01 | Challenge generator (code-switched nonce) | claude-b8 | DONE | B0-T05 | packages/vg_models/heads/head_e_liveness/challenge.py | secrets-based code-switched digits (en/hi/ta), nonce phrases, reverse order; per-session registry |
| B8-T02 | Response verifier (ASR match + latency) | claude-b8 | REVIEW | B8-T01, B10-T01 | packages/vg_models/heads/head_e_liveness/verifier.py | ordered content match (any script/romanisation), latency, rate, gap regularity; heuristic weights need shadow-mode tuning; needs B10 ASR tokens |
| B8-T03 | Operator-triggered challenge hook | claude-b8 | REVIEW | B8-T01, B13-T03 | packages/vg_models/heads/head_e_liveness/api.py | REST hook: issue / prompt-end / response / active; UI button is B13, auth is B12 |
| B8-T04 | AudioSeal watermark detector | claude-b8 | REVIEW | B2-T04 | packages/vg_models/heads/head_f_watermark/detectors.py, packages/vg_models/heads/head_f_watermark/head.py | AudioSeal wrapper (not installed); keyed spread-spectrum reference detector tested: z~50 marked, ~34 after G.711, ~44 after Opus 16k, ~4 clean/wrong key |
| B8-T05 | Additional commercial watermark detectors | claude-b8 | REVIEW | B8-T04 | packages/vg_models/heads/head_f_watermark/detectors.py | partner API detectors gated on tenant opt-in (I6); no vendor integrations obtained |
| B8-T06 | Enforce absence-is-neutral rule | claude-b8 | DONE | B8-T04, B9-T02 | packages/vg_models/heads/head_f_watermark/asymmetry.py, docs/adr/0007-watermark-absence-neutral-abstain.md | absence = neutral abstain; fusion_inputs filter; DoD test: unwatermarked clip changes fused score by exactly 0 |

### B9 — Fusion & risk engine
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B9-T01 | Per-head calibration | claude-b9 | REVIEW | B0-T05 | packages/vg_models/calibration.py | Platt/temperature per (head, model_version) + ECE; ECE<0.05 on synthetic held-out; needs deployment-like data |
| B9-T02 | Cross-head fusion (LR/GBDT) + abstain handling | claude-b9 | REVIEW | B9-T01 | services/fusion/fuser.py | LR over calibrated logits + missing indicators (no zero fill), Head F via ADR 0007 filter; prior weights until fitted on real data |
| B9-T03 | Temporal fusion (Bayesian / HMM) | claude-b9 | DONE | B9-T02 | services/fusion/temporal.py | two-state HMM forward filter over window LLRs; abstained windows = transition only |
| B9-T04 | Any-segment trigger for partial splicing | claude-b9 | DONE | B9-T03 | services/fusion/temporal.py, services/fusion/engine.py | evidence-based segment posterior; 4-window splice caught in >=10/12 simulated calls, genuine false trigger <=1/12 |
| B9-T05 | Four-state emission incl. ABSTAIN | claude-b9 | DONE | B9-T03 | services/fusion/operating_point.py | LOW/ELEVATED/HIGH/ABSTAIN with explicit abstain rules + hysteresis; ~1-2 state changes/min vs 25+ naive |
| B9-T06 | Cost-model operating point selection | claude-b9 | REVIEW | B9-T01, B15-T09 | services/fusion/operating_point.py | thresholds at fixed FPR (1%/0.1%) + expected-cost minimiser; per-tenant profile files; needs real score distributions |
| B9-T07 | Per-head contribution attribution | claude-b9 | DONE | B9-T02 | services/fusion/fuser.py, services/fusion/engine.py | normalised contributions (sum 1) + signed drivers; any-segment driver only when it actually fires |
| B9-T08 | Score timeline persistence + replay | claude-b9 | REVIEW | B9-T05 | services/fusion/persistence.py, services/fusion/main.py | SQLite store + GET timeline replay API tested; Timescale DDL provided but not run (B17) |

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
| 2026-09-17T06:00Z | claude-b9 | B9-T01-08 | TODO -> DONE(T03,T04,T05,T07) / REVIEW(T01,T02,T06,T08) | services/fusion/, packages/vg_models/calibration.py, tests/unit/test_b9_fusion.py | 220 tests pass; replay.py now uses RiskEngine; fixed flaky sample-store TTL test (Windows clock resolution) |
| 2026-09-17T05:00Z | claude-b8 | B8-T01-06 | TODO -> DONE(T01,T06) / REVIEW(T02-T05) | packages/vg_models/heads/head_e_liveness/, packages/vg_models/heads/head_f_watermark/, tests/unit/test_b8_liveness_watermark.py | 202 tests pass; ADR 0007; AudioSeal + B10 ASR deferred |
| 2026-09-17T04:00Z | claude-b7 | B7-T01-06 | TODO -> DONE(T02,T03,T05,T06) / REVIEW(T01,T04) | packages/vg_models/heads/head_d_speaker/, services/enrollment/, tests/unit/test_b7_speaker.py | 188 tests pass; neural embedder + cohort deferred to setup |
| 2026-09-17T03:00Z | claude-b6 | B6-T01-06 | TODO -> DONE(T01-03) / REVIEW(T04-06) | packages/vg_models/heads/head_c_prosody/, ml/training/train_head_c.py, tests/unit/test_b6_head_c.py | 174 tests pass; training deferred |
| 2026-09-17T02:00Z | claude-b5 | B5-T01-06 | TODO -> DONE(T01,T04-06) / REVIEW(T02,T03) | packages/vg_models/heads/head_b_dsp/, ml/training/train_head_b.py, tests/unit/test_b5_head_b.py | CPU ~10 ms/window; training deferred |
| 2026-09-17T01:30Z | claude-b4 | Q6/Q7 | answered | docs/adr/0006-untrained-abstain-reason.md, schemas/head_score.schema.json, proto/voiceguard.proto | `untrained` abstain reason; cross-block edits accepted; fixed pre-warmup model_version bug in HeadA |
| 2026-09-17T01:00Z | claude-b4 | B4-T01-10 | TODO -> DONE(T03-05) / REVIEW(T02,06,07,09,10) / WIP(T08) / BLOCKED(T01) | packages/vg_models/heads/head_a_ssl/, ml/training/, tests/unit/test_b4_head_a.py | 135 tests pass; training deferred (docs/SETUP_PENDING.md) |
| 2026-09-17T00:00Z | claude-b3 | Q1 | answered | PROJECT_STATUS.md §6 | ASVspoof 5 on hand; IndicSynth planned; build structure first, data setup deferred |
| 2026-09-16T19:41Z | claude-b3 | B3-T01-11 | TODO -> DONE(T01,04,07-11) / REVIEW(T02,03,06) / WIP(T05) | ml/data/, tests/unit/test_b3_data.py | 106 tests pass; corpus not built (needs data access + GPU) |
| 2026-08-30T17:50Z | agent-antigravity | B2-T01-08 | WIP -> DONE | packages/vg_audio/, services/conditioner/ | 82 tests pass, 81.8% cov |
| 2026-08-30T16:47Z | agent-antigravity | B1-T01-09 | WIP -> DONE | services/ingest/ | 36 tests pass, 79% cov |
| 2026-08-30T10:35Z | agent-antigravity | B0-T01-09 | TODO -> DONE | vg_core/, schemas/ | Scaffolded repo, contracts, pipelines |
| — | — | — | board created | — | initial seed |

## 6. Open questions

Anything that needs a human decision. Agents append here rather than guessing.

| # | Question | Raised by | Blocks | Answer |
|---|---|---|---|---|
| Q1 | Which datasets have we actually been granted access to? | — | B3, B4 | 2026-09-17 (user): ASVspoof 5 available locally (132 GB). IndicSynth planned, not yet obtained. Nothing else requested. |
| Q2 | Target deployment hardware for the latency claim (T4? L4? CPU-only?) | — | B14 | |
| Q3 | Commercial or research lineage for the primary demo model? | — | B3-T09 | Implied research: ASVspoof 5 EULA + IndicSynth (CC BY-NC 4.0) are both non-commercial. Needs explicit human confirmation. |
| Q4 | Which telephony stack does the pilot tenant actually run? | — | B1 | |
| Q5 | Which Indic languages are in scope for v1 (all 12, or 3–4 done well)? | — | B3, B15 | 2026-09-17 (user): all 12, subject to IndicSynth coverage. |
| Q6 | Add `untrained` to AbstainReason? Untrained Head A currently abstains with reason null. Contract change, needs ADR. | claude-b4 | B0, B9 | 2026-09-17 (user): implement if useful beyond setup. Done: ADR 0006, `untrained` added to schema/proto/models; used by heads A, B, C. |
| Q7 | Accept cross-block edits from B4: `packages/vg_core/sample_store.py` (B0; resolves samples_ref) and `scripts/replay.py` (--real-heads, real model versions)? | claude-b4 | B0, B2 | 2026-09-17 (user): accepted. |

## 7. Blocked items

| Task | Blocked since | Blocked by | Owner of the blocker | Escalation |
|---|---|---|---|---|
| B4-T01 | 2026-09-17 | deferred user setup (datasets, weights, CUDA torch) | project owner | docs/SETUP_PENDING.md |
