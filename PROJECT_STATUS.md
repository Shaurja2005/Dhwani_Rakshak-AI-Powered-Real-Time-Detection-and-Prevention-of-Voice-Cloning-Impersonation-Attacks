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
| B3 | Data & corpus engineering | claude-b3 | WIP | 7/13 |
| B4 | Head A — SSL anti-spoof | claude-b4 | WIP | 3/13 |
| B5 | Head B — DSP & scene | claude-b5 | WIP | 4/6 |
| B6 | Head C — prosody | claude-b6 | WIP | 3/6 |
| B7 | Head D — speaker verification | claude-b7 | WIP | 4/6 |
| B8 | Heads E/F — liveness & watermark | claude-b8 | WIP | 2/6 |
| B9 | Fusion & risk engine | claude-b9 | WIP | 4/8 |
| B10 | Context & intent | claude-b10 | WIP | 4/7 |
| B11 | Policy & alerting | claude-b11 | WIP | 6/8 |
| B12 | APIs & SDKs | claude-b12 | WIP | 6/9 |
| B13 | Agent/analyst UI | claude-b13 | REVIEW | 6/7 |
| B14 | Serving & optimization | claude-b14 | WIP | 1/6 |
| B15 | Evaluation harness | claude-b15 | WIP | 1/10 |
| B16 | Privacy & compliance | claude-b16 | WIP | 7/9 |
| B17 | Deployment & ops | claude-b17 | WIP | 4/8 |
| B18 | Demo & submission | claude-b18 | WIP | 1/6 |

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
| B3-T12 | ASVspoof 5 + ASVspoof 2019 LA importers (protocols -> manifests + official splits) | claude-b3 | REVIEW | B3-T01 | ml/data/importers/, scripts/setup/extract_asvspoof5.py | md5-verified resumable extraction; attack label = generator family, codec conditions kept; tested on a miniature layout; real run done (182,357 train / 140,950 dev / 680,774 eval rows; train+dev contain no codec conditions). ASVspoof 5 LICENSE.txt says ODC-By 1.0 (+CC BY 4.0 bona fide) while the registry says non-commercial EULA: needs human sign-off (Q12) |
| B3-T13 | IndicSynth + Kathbath per-language importer (sampled parquet row groups, speaker-disjoint splits) | claude-b3 | REVIEW | B3-T01, B3-T10 | ml/data/importers/indic.py, tests/unit/test_indic_continual.py | random row groups over the whole language (IndicSynth rows are stored sorted by generator/gender), pinned revisions, resumable parts, 16 kHz FLAC; clone + genuine voice share a speaker id (Kathbath ids) so splits are speaker-disjoint, clip-balanced per label; import report lists original sample rates and cross-split VC source speakers; Hub layer tested on local parquet (needs pyarrow); Kathbath is gated (user login) |

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
| B4-T11 | Base-model training path for an 8 GB GPU (frozen XLS-R feature cache + cached back-end trainer) | claude-b4 | REVIEW | B4-T06, B3-T12 | ml/training/feature_cache.py, ml/training/train_head_a_cached.py, ml/training/configs/head_a_base.yaml, scripts/train_base_model.py, docs/TRAINING_GUIDE.md, REPORT.md#r2b2ae19f | first real run done on the RTX 4060 (early stop at epoch 6, best dev epoch 0): r2b2ae19f.overall.all; codec-free eval slice r2b2ae19f.codec.none vs worst r2b2ae19f.codec.codec_C07q1 / codec_C04q1 (Encodec) -> next: codec-augmented cache. fp16 extraction of only the layers used (truncate_to_used_layers: identical states, 10/24 layers), resumable memmap cache, AMP + balanced sampling + early stop + resume; standard checkpoint (cached-vs-audio scores equal in tests); eval scorer on GPU; waits for the user's GPU run |
| B4-T12 | Channel-augmented views in the feature cache + v0.2 model (label-blind codec/noise/RawBoost chains before XLS-R) | claude-b4 | REVIEW | B4-T11, B3-T07 | ml/training/channel_aug.py, ml/training/feature_cache.py, ml/training/train_head_a_cached.py, ml/training/configs/head_a_aug.yaml, scripts/train_augmented_model.py, TRAINING_AUGMENTED.md | augmented splits (source + view) with chains.json provenance; multi-set training with pooled dev early stop + per-set dev EER; eval notes that its codec/SNR sweeps share encoders with training; chain seeded by (seed, view, utt id) only (I3); tested on a miniature layout + real-data check (13 codecs, ~68 clips/s with 8 workers); waits for the user's GPU run |
| B4-T13 | Per-language continual fine-tuning (replay caches, init from previous model, mean-by-set early stop, before/after eval per language) | claude-b4 | REVIEW | B4-T12, B3-T13, B15-T04 | scripts/train_indic_language.py, ml/training/replay_cache.py, ml/training/configs/indic_continual.yaml, TRAINING_INDIC.md | ledger fixes the language order; language k starts from k-1 and trains on its clean + augmented cache plus ASVspoof 5 + all earlier replays; eval scores the previous and the new model on all test sets so far (tagged run records); compare shows per-language deltas, pass/fail limit pending Q13; two-language chain tested end to end on a miniature layout; EWC (ml/training/continual.py) not used: replay only |

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
| B10-T01 | Streaming ASR (IndicConformer/Whisper) | claude-b10 | REVIEW | B2-T04 | services/context/asr.py | streaming buffer + rolling transcript + word tokens; Whisper / IndicConformer wrappers lazy (not installed); scripted backend tested |
| B10-T02 | Language ID + code-switch detection | claude-b10 | DONE | B10-T01 | services/context/langid.py | 12 Indic scripts + Hinglish lexicon, matrix-language rule, segment-level hi/mr disambiguation, code-switch flag |
| B10-T03 | Social-engineering intent classifier (local LLM) | claude-b10 | REVIEW | B10-T01 | services/context/intent.py, services/context/seed.py, services/context/data/intent_seed.jsonl | rules (en/hi/Hinglish) + local-only LLM (I6 enforced, async I10); 196-scenario seed set (self-authored: regression only, not accuracy evidence); LLM not run |
| B10-T04 | Metadata risk scorer | claude-b10 | DONE | B1-T08 | services/context/metadata_risk.py | first-time, CLI mismatch, intl-spoofed CLI, trunk reputation, off-hours, velocity; history backend is in-memory interface |
| B10-T05 | Transaction context connector (mock core banking) | claude-b10 | REVIEW | B0-T03 | services/context/transaction.py | mock core banking: unusual amount, new/recent beneficiary, first high-value, privileged request, sensitivity tier; real connector per tenant |
| B10-T06 | ContextSignals assembly | claude-b10 | DONE | B10-T03, B10-T04 | services/context/engine.py, services/context/main.py | ContextSignals + explanations + final_risk; DoD test: genuine voice LOW + fraud script intent HIGH, both in breakdown |
| B10-T07 | PII redaction on transcripts | claude-b10 | DONE | B10-T01 | services/context/redact.py | card(Luhn)/OTP/Aadhaar/phone/IFSC/PAN/UPI/email/account + spoken digits (en/hi); applied before storage and before the LLM |

### B11 — Policy & alerting
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B11-T01 | Rules engine + tenant threshold profiles | claude-b11 | DONE | B9-T05 | services/policy/profiles.py, services/policy/engine.py | profiles are data (JSON) per tenant; 3 built-in profiles; band from final_risk |
| B11-T02 | Action library | claude-b11 | DONE | B11-T01 | services/policy/actions.py | all 12 contract actions with audience + agent wording; label-triggered minimum actions |
| B11-T03 | Transaction-sensitivity tiered thresholds | claude-b11 | DONE | B11-T01, B10-T05 | services/policy/profiles.py | per-tier bands (low/medium/high transaction sensitivity), e.g. balance inquiry HIGH at 0.95, wire at 0.60 |
| B11-T04 | Evidence bundle (immutable, hashed) | claude-b11 | DONE | B9-T08 | services/policy/evidence.py | sha256 content hash, schema-validated, append-only SQLite with trigger + hash chain; no audio |
| B11-T05 | Multi-channel notification + SIEM webhook | claude-b11 | REVIEW | B11-T02 | services/policy/notify.py | WebSocket pub/sub + HMAC-signed SIEM webhook with backoff tested; SMS/email/push are recording providers (real providers per tenant) |
| B11-T06 | Pre-transaction warning prompts | claude-b11 | DONE | B11-T02 | services/policy/actions.py | plain, advisory agent prompts; ABSTAIN wording never implies safety |
| B11-T07 | Shadow mode (default for new tenants) | claude-b11 | DONE | B11-T01 | services/policy/profiles.py, services/policy/notify.py | shadow mode default for new tenants: decisions + bundles recorded, nothing dispatched |
| B11-T08 | Analyst feedback loop → eval + replay buffer | claude-b11 | REVIEW | B11-T04, B13-T06 | services/policy/feedback.py | analyst labels -> eval rows + replay candidates; UI buttons are B13 |

### B12 — APIs & SDKs
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B12-T01 | gRPC bidirectional streaming | claude-b12 | DONE | B0-T02 | services/api_gateway/grpc_server.py, services/api_gateway/proto_codec.py, services/api_gateway/pipeline.py | bidi AnalyzeStream + AnalyzeFile + Enroll with key auth; tested against a real in-process gRPC server; fixed invalid proto option that broke codegen |
| B12-T02 | REST API surface | claude-b12 | DONE | B0-T03 | services/api_gateway/app.py | sessions, WS /v1/stream, analyze/file, evidence, feedback, profile, shadow, keys, enrollment + liveness mounted with ownership checks |
| B12-T03 | Signed webhooks + retry | claude-b12 | DONE | B11-T05 | services/api_gateway/webhooks.py | https-only tenant subscriptions, HMAC-SHA256 signed, async delivery with backoff (signing from B11 notify) |
| B12-T04 | AuthN/AuthZ, quotas, rate limits | claude-b12 | REVIEW | B12-T02 | services/api_gateway/auth.py | hashed per-tenant keys, scopes, default-deny, tenant + session isolation, token-bucket rate limit, request + audio quotas; in-memory key store (persist + mTLS at ingress in B17) |
| B12-T05 | Python SDK | claude-b12 | DONE | B12-T01 | sdks/python/voiceguard/ | analyze_file, stream (session REST), gRPC stream, enroll, evidence, feedback, webhooks, verify signature |
| B12-T06 | JS/TS SDK | claude-b12 | DONE | B12-T01 | sdks/js/ | TS SDK (fetch + WebSocket stream, PCM helper, WebCrypto webhook verification); 5 node:test tests |
| B12-T07 | Edge/mobile SDK stub (ONNX/TFLite) | claude-b12 | BLOCKED | B14-T03 | sdks/python/voiceguard/edge.py | ONNX Runtime edge scorer stub abstains "untrained" until B14 exports a distilled model |
| B12-T08 | OpenAPI docs + 10-minute quickstart | claude-b12 | DONE | B12-T02 | docs/api/QUICKSTART.md, docs/api/openapi.json, scripts/export_openapi.py, examples/quickstart.py | quickstart path exercised in tests end to end |
| B12-T09 | Reference connectors | claude-b12 | REVIEW | B1-T03, B1-T05 | services/api_gateway/connectors.py | Twilio / Asterisk AudioSocket / FreeSWITCH bridges via SDK (Twilio tested end to end); collaboration-platform browser extension not built |

### B13 — UI
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B13-T01 | Live call view (gauge, timeline, head bars) | claude-b13 | DONE | B0-T05 | services/ui/public/views/live.js, services/ui/public/lib/components.js, services/ui/public/lib/state.js | gauge, timeline sparkline, named head bars, state badge with distinct ABSTAIN; watch live calls or simulate from a file; verified visually (desktop + 375px) |
| B13-T02 | Live transcript with intent highlights | claude-b13 | DONE | B10-T06 | services/ui/public/lib/components.js | transcript with inline highlighted intent phrases + plain-language chips; numbers pre-redacted by B10 |
| B13-T03 | Alert banner + one-click step-up | claude-b13 | DONE | B11-T02 | services/ui/public/views/live.js | alert banner with agent prompt, recommended actions, shadow-mode note; call-back + challenge-phrase buttons (challenge via B8 API) |
| B13-T04 | Post-call forensics view + PDF export | claude-b13 | DONE | B11-T04 | services/ui/public/views/forensics.js | timeline scrub, evidence bundle with integrity check, model versions, full rationale, PDF via print stylesheet |
| B13-T05 | Admin console (profiles, enrollment, shadow) | claude-b13 | DONE | B11-T01, B7-T02 | services/ui/public/views/admin.js | profile bands table + JSON edit, shadow toggle, voiceprint list/enrol/delete with consent reference |
| B13-T06 | Analyst feedback buttons | claude-b13 | DONE | B13-T03 | services/ui/public/views/live.js, services/ui/public/views/forensics.js | true/false positive buttons -> /v1/evidence/{id}/feedback |
| B13-T07 | Demo mode (genuine vs cloned side by side) | claude-b13 | REVIEW | B13-T01 | services/ui/public/views/demo.js | side-by-side genuine vs cloned analysis with playback-synchronised gauges; needs trained heads + consented clone to be meaningful |

### B14 — Serving & optimization
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B14-T01 | Distill ensemble → student model | claude-b14 | REVIEW | B4-T08 | ml/export/distill.py, ml/training/configs/distill_head_a.yaml | truncated-SSL student (first k layers) + score/embedding/OC-Softmax distillation from 1..n teachers; CLI tested end to end on synthetic data; student checkpoints reload truncated. Needs trained B4 teacher (SETUP_PENDING B14) |
| B14-T02 | INT8 quantization + accuracy delta report | claude-b14 | REVIEW | B14-T01 | ml/export/quantize.py, docs/benchmarks/loadtest/ | dynamic INT8 (Linear/GRU/LSTM) + report of score drift, decision flips, EER fp32 vs int8, size, latency. Latency measured on proxy; accuracy delta needs labelled dev set + trained student |
| B14-T03 | ONNX export + parity check | claude-b14 | REVIEW | B14-T01 | ml/export/to_onnx.py, ml/export/parity_check.py, services/inference/backends.py | calibration baked into graph (spoof_logit, raw_score), dynamic batch/time axes, parity over varied lengths + decision flips. onnx/onnxruntime not installed -> parity test skips; run after `pip install onnx onnxruntime` |
| B14-T04 | Triton (server) + TFLite/ORT-Mobile (edge) | claude-b14 | REVIEW | B14-T03 | deploy/triton/, services/inference/, ml/export/edge.py | Triton config (ORT backend, dynamic batching, reject-on-timeout) + HTTP client backend; in-process cross-session batcher + BatchedHeadA wired into gateway via VG_INFERENCE_BACKEND; CPU inference service; ORT-mobile + ONNX INT8 wrappers. Not run against a real Triton/GPU; TFLite not wired |
| B14-T05 | Load test + cost per 1k call-minutes | claude-b14 | REVIEW | B14-T04 | docs/benchmarks/LOADTEST.md, docs/benchmarks/loadtest/, scripts/loadtest.py | full pipeline, real-time paced calls on i7-13620H CPU: 6 calls (FP32) / 8 calls (INT8) within p95 < 700 ms, with XLS-R-shaped 6-layer proxy weights; cost formula given, no price. Needs trained student + GPU (T4/L4) run for the plan target (Q2) |
| B14-T06 | Backpressure & graceful degradation | claude-b14 | DONE | B14-T04 | services/inference/degrade.py, services/inference/batching.py, services/api_gateway/pipeline.py, tests/unit/test_b14_serving.py | bounded batcher queue (reject, never queue) + deadline drop; LoadController with hysteresis sheds Head A (abstain timeout, evidence.load_shed) and marks cheap heads degraded with reduced confidence; window_score events carry degraded=true; VG_LOAD_SHEDDING=1 |

### B15 — Evaluation harness
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B15-T01 | Harness skeleton (adopt AUDDT/DeepFense) | claude-b15 | REVIEW | B3-T04 | ml/eval/run_eval.py, ml/eval/scorers.py, ml/eval/configs/ | own harness: any scorer spec (head_a ckpt, deployed heads, full pipeline, baseline, registry version) -> all protocols -> run record -> REPORT.md; runs end to end on synthetic data (`make eval-smoke`); real run needs B3 manifests + trained model |
| B15-T02 | Leave-one-generator-out protocol | claude-b15 | REVIEW | B15-T01 | packages/vg_eval/protocols.py | per-family EER vs full bona fide pool, seen/unseen from checkpoint meta, pooled ALL_UNSEEN row, fold specs for true LOGO retraining; needs data |
| B15-T03 | Cross-dataset protocol | claude-b15 | REVIEW | B15-T01 | packages/vg_eval/protocols.py, ml/eval/configs/default.yaml | per-dataset EER + eer_minus_in_domain; In-the-Wild eval-only via license gate purpose=eval; needs data |
| B15-T04 | Per-language / per-accent breakdown | claude-b15 | REVIEW | B15-T01, B3-T03 | packages/vg_eval/protocols.py, packages/vg_eval/report.py (plot_before_after) | per language/accent rows + before/after chart generator; needs Indic data + two model versions |
| B15-T05 | Fairness: per-gender/per-language FPR gaps | claude-b15 | REVIEW | B15-T04 | packages/vg_eval/fairness.py | FPR per group at one global threshold with Wilson CIs, gap + ratio; release gate reports not_configured until the max gap is decided (Q9); registry refuses promotion on fail |
| B15-T06 | Per-codec / per-SNR curves | claude-b15 | REVIEW | B15-T01, B3-T07 | ml/eval/run_eval.py, ml/eval/adversarial.py | manifest codec/SNR slices + codec sweep (g711u/a, amr_nb, gsm, opus, g722 re-encode) + SNR sweep on the same audio; G.729/EVS need encoders ffmpeg lacks |
| B15-T07 | Operational metrics | claude-b15 | REVIEW | B14-T05 | packages/vg_eval/protocols.py (operational), packages/vg_eval/report.py (render_serving) | time-to-first-alert, score stability (std, flips/min) at a data-derived window threshold; B14 load-test rows imported into REPORT.md as citable lt-* rows; throughput per GPU pending GPU run |
| B15-T08 | Adversarial / laundering robustness | claude-b15 | REVIEW | B15-T01 | ml/eval/adversarial.py | laundering on spoofs only (codec re-encode, speed, pitch, noise) + white-box PGD (L-inf) and a Malacopula-style universal FIR filter learnt on half the spoofs; needs trained model |
| B15-T09 | Metrics: EER, t-DCF, a-DCF, pAUC, ECE | claude-b15 | DONE | B15-T01 | packages/vg_eval/metrics.py, tests/unit/test_b15_eval.py | EER (checked vs analytic Phi(-1)), ASVspoof5 minDCF, 2019 t-DCF, a-DCF, AUC, McClish pAUC, ECE, bootstrap CI; abstentions excluded + coverage reported. a-DCF constants to be re-checked against the eval plan in use |
| B15-T10 | Auto-generated benchmark report in CI | claude-b15 | REVIEW | B15-T09 | packages/vg_eval/report.py, .github/workflows/eval.yml, docs/benchmarks/REPORT.md | REPORT.md rebuilt from run records with stable row ids; synthetic runs refused; CI smoke job + full-eval on a self-hosted data runner (runner not provisioned) |

### B16 — Privacy & compliance
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B16-T01 | Ephemeral processing + no-raw-audio-at-rest test | claude-b16 | DONE | B2-T04 | packages/vg_core/sample_store.py, packages/vg_audio/windowing.py, tests/compliance/test_no_raw_audio_at_rest.py | store owns + zero-overwrites buffers on drop/evict/TTL; windower wiped on close; end-to-end test scans every file written (data dir, temp dir, open() spy) for the input audio in 3 encodings |
| B16-T02 | Feature-only logging | claude-b16 | DONE | B16-T01 | services/privacy/feature_log.py, services/privacy/raw_retention.py | structlog processor strips audio-shaped values (installed by the gateway); flagged-call raw audio vault is off by default, consent-checked, AES-GCM, expiring, audited |
| B16-T03 | On-prem/edge default topology | claude-b16 | DONE | B17-T02 | docs/compliance/DATA_RESIDENCY.md, services/privacy/egress.py, deploy/docker-compose.prod.yml | on-prem default; internal no-egress network (compose) + default-deny NetworkPolicy (k8s); egress guard: host allowlist, offshore refused without opt-in, audio payloads blocked (wired into webhooks) |
| B16-T04 | Consent matrix as configuration | claude-b16 | DONE | B1-T08 | config/privacy/consent_matrix.yaml, services/privacy/policy.py | purposes x lawful bases, bundling rules, purpose limitation; enforced in SessionPipeline (REST 403 / gRPC PERMISSION_DENIED without a basis for fraud_detection; transcript analysis gated separately) |
| B16-T05 | Data residency documentation & enforcement | claude-b16 | DONE | B10-T03 | docs/compliance/DATA_RESIDENCY.md, services/privacy/egress.py | region IN per tenant, central processing opt-in only, enforcement table with tests |
| B16-T06 | Retention/erasure automation + DSAR | claude-b16 | DONE | B16-T02 | services/privacy/retention.py, services/policy/evidence.py, services/privacy/main.py | per-tenant retention over evidence/timeline/feedback/voiceprints/flagged audio with legal holds + dry run; DSAR export/erasure by customer id, caller number or speaker id; evidence bodies erasable while the hash chain still verifies (v1 DBs migrated in place). Retention periods need DPO sign-off (Q10) |
| B16-T07 | Per-tenant keys + encrypted vault | claude-b16 | DONE | B0-T04 | services/privacy/keys.py | envelope encryption (DEKs wrapped by a KEK), versioned rotation that re-encrypts the voiceprint + flagged-audio vaults and shreds the old key, tenant crypto-shredding; production KEK comes from the bank KMS/HSM (kek_provider) |
| B16-T08 | DPIA template + completed example | claude-b16 | REVIEW | B16-T04 | docs/compliance/DPIA.md, services/privacy/audit.py, services/privacy/audit_event.schema.json | completed example + audit-log schema (hash-chained, append-only, hashed subject refs); needs DPO / legal sign-off |
| B16-T09 | Model cards + data statements | claude-b16 | REVIEW | B15-T10 | docs/model_cards/TEMPLATE.md, services/privacy/model_card.py | generated from registry + eval run + data registry, numbers carry REPORT.md row ids; no released model yet |

### B17 — Deployment & ops
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B17-T01 | Dockerfiles per service | claude-b17 | REVIEW | B0-T06 | deploy/docker/render_dockerfiles.py, services/*/Dockerfile, .dockerignore | 9 generated multi-stage slim non-root images with healthchecks, drift-checked in CI/tests; not built here (Docker daemon off; base images need a pull) |
| B17-T02 | Single-node on-prem compose | claude-b17 | REVIEW | B17-T01 | deploy/docker-compose.prod.yml, deploy/.env.example, Makefile (prod-*) | gateway, privacy API + retention sidecar, prometheus, grafana, optional inference (split) and replayer; internal no-egress network, read-only roots, tmpfs /tmp; validated with `docker compose config`; `make prod-up` on a clean VM not yet run (DoD) |
| B17-T03 | Helm / k8s manifests | claude-b17 | REVIEW | B17-T01 | deploy/k8s/ | kustomize base + prod overlay (Helm not installed): hardened pods, PVCs, blue/green inference + HPA + PDB, retention CronJob, default-deny NetworkPolicies; renders offline with `kubectl kustomize`; not applied to a cluster |
| B17-T04 | Prometheus metrics | claude-b17 | DONE | B17-T01 | services/observability/metrics.py, services/api_gateway/app.py (/metrics), deploy/observability/prometheus.prod.yml, deploy/observability/alerts.yml | per-head latency, abstain by reason, alert rate, windows by state, sessions, queue depth, degraded windows, model version, drift PSI, GPU; verified end to end (dev gateway + replayed calls) |
| B17-T05 | Grafana dashboards (ops + fraud) | claude-b17 | REVIEW | B17-T04 | deploy/observability/grafana/ | ops + fraud-analytics dashboards generated from one builder; every query checked against exported metric names; provisioning for compose; not yet viewed in a running Grafana |
| B17-T06 | Score-distribution drift monitoring | claude-b17 | DONE | B17-T04, B9-T08 | services/observability/drift.py | PSI + KS on the bona fide proxy (sessions that stayed LOW / confirmed genuine) vs a frozen reference, exported + alerted; alert levels are operational defaults to tune in shadow mode (Q11) |
| B17-T07 | Model registry + blue/green rollout | claude-b17 | DONE | B14-T04 | packages/vg_models/registry.py, scripts/model_rollout.py, services/inference/served_head.py | immutable versions + sha256, promotion requires a real eval run and a non-failing fairness gate (I7 check for commercial), instant rollback; VG_INFERENCE_BACKEND=registry hot-swaps the served model on the next batch |
| B17-T08 | On-call runbook | claude-b17 | DONE | B17-T05 | docs/runbooks/ONCALL.md | abstain spike, latency, alert-rate x3, drift, rollout, gateway down, privacy incidents; every alert links to its section (tested) |

### B18 — Demo & submission
| ID | Task | Owner | Status | Depends | Artifact | Notes |
|---|---|---|---|---|---|---|
| B18-T01 | Live clone-the-judge demo script | claude-b18 | REVIEW | M1 | docs/demo/DEMO_SCRIPT.md, docs/demo/scenarios.yaml, scripts/demo_scenarios.py | timed script with consent protocol and failure-mode recoveries; rehearsal runner checks each layer; needs trained heads, a consented volunteer and a rehearsal |
| B18-T02 | Human-reading-fraud-script demo | claude-b18 | REVIEW | B10-T03 | docs/demo/scenarios.yaml, scripts/demo_scenarios.py | intent layer verified (credential_request, urgency, secrecy -> ELEVATED + verify prompt); acoustic-LOW half needs trained heads and a recorded human |
| B18-T03 | Before/after Indic EER chart | claude-b18 | REVIEW | B15-T04 | scripts/make_headline_chart.py | chart from two run records (refuses synthetic or mismatched eval data); needs the two real evaluation runs |
| B18-T04 | Architecture diagram, README, video | claude-b18 | REVIEW | — | docs/ARCHITECTURE.md, README.md, docs/demo/VIDEO_SCRIPT.md | mermaid architecture, rewritten README, 3-minute shot list; the video itself must be recorded by people |
| B18-T05 | Offline demo fallback | claude-b18 | REVIEW | B18-T01 | scripts/offline_demo.py, Makefile (demo-*) | three fallback tiers; tier 3 plays a recorded pack of scores/events with no network or models; pack to be recorded after the final rehearsal (`make demo-pack`) |
| B18-T06 | Framing statements in the writeup | claude-b18 | DONE | — | docs/demo/WRITEUP.md, README.md, docs/demo/DEMO_SCRIPT.md, tests/unit/test_b18_demo.py | both statements in README, write-up and demo script (tested); pitch docs may not contain an uncited percentage (tested) |

## 5. Changelog

Append one line per status change. Newest at the top. Never edit an existing line.

```
YYYY-MM-DDTHH:MMZ | <agent-or-human> | <TASK-ID> | <OLD> -> <NEW> | <artifact/PR> | <one-line note>
```

| When | Who | Task | Change | Artifact | Note |
|---|---|---|---|---|---|
| 2026-10-01T09:00Z | claude-b4 | B4-T13 | REVIEW (note) | scripts/train_indic_language.py, TRAINING_INDIC.md | queue command (several languages in a row); run commands + download estimates for all 12 languages; manual disk clean-up table; first real run stopped at the Kathbath gate (no HF login yet) |
| 2026-10-01T08:00Z | claude-b4 | B4-T12 | REVIEW (note) | docs/benchmarks/runs/A_xlsr300m-nes2net-v0.2.0.json, REPORT.md#r9ffe9c63 | v0.2 evaluated: r9ffe9c63.overall.all within CI of r2b2ae19f.overall.all; pAUC@1% up (r9ffe9c63.overall.all), codec.none and most per-codec rows worse, own codec/SNR sweeps not improved, operational.bona_fide worse; augmentation likely too strong (dev_aug1 14.4% at best epoch). Not a clear win; Indic init choice left to the user |
| 2026-09-30T18:00Z | claude-b3 | B3-T13 | new -> REVIEW | ml/data/importers/indic.py | per-language IndicSynth + Kathbath sampler; registry: IndicSynth access public, languages incl. sa (no as), 845 GB; Kathbath access gated_hf |
| 2026-09-30T18:00Z | claude-b4 | B4-T13 | new -> REVIEW | scripts/train_indic_language.py, TRAINING_INDIC.md | continual per-language training; trainer: train.init_from + train.early_stop=mean_by_set; user decisions: 12k+12k per language, manual cache deletion, Bengali first |
| 2026-09-30T18:00Z | claude-b15 | B15-T04 | REVIEW (note) | packages/vg_eval/report.py, ml/eval/run_eval.py | optional run tag (--tag) so one model can be evaluated on several eval-set collections; not part of run_id (existing row ids unchanged) |
| 2026-09-30T15:30Z | claude-b4 | B4-T12 | new -> REVIEW | ml/training/channel_aug.py, TRAINING_AUGMENTED.md | augmented-view cache + v0.2 config; Augmenter.run returns the applied chain |
| 2026-09-30T15:30Z | claude-b3 | B3-T07 | DONE (note) | ml/data/channel/codecs.py, ml/data/channel/webrtc_chain.py | + opus_nb, amr_wb, mp3, aac, speex, speex_nb via ffmpeg; per-codec bitrate choices; AMR bitrates snap to valid modes; defaults unchanged |
| 2026-09-30T13:10Z | claude-b4 | B4-T11 | REVIEW (note) | docs/benchmarks/runs/A_xlsr300m-nes2net-v0.1.0.json, REPORT.md#r2b2ae19f | first GPU run + eval; fixed frozen-frontend input gradients (PGD) and chunked universal-filter learning; status now checks last.pt for a finished run |
| 2026-09-30T19:40Z | claude-b4 | B4-T11 | new -> REVIEW | ml/training/feature_cache.py, ml/training/train_head_a_cached.py, docs/TRAINING_GUIDE.md | base-model path for the RTX 4060 laptop; tiny front-end made seed-deterministic (packages/vg_models/heads/head_a_ssl/frontend.py) |
| 2026-09-30T19:40Z | claude-b3 | B3-T12 | new -> REVIEW | ml/data/importers/ | ASVspoof 5 / 2019 LA importers + md5-verified extraction; licence discrepancy raised as Q12 |
| 2026-09-24T01:27Z | claude-b18 | B18-T01-06 | TODO -> DONE(T06) / REVIEW(T01-T05) | docs/demo/, scripts/demo_scenarios.py, scripts/offline_demo.py, scripts/make_headline_chart.py, docs/ARCHITECTURE.md, README.md | 13 tests; demo needs trained heads, consented volunteer, rehearsal, video recording |
| 2026-09-24T01:27Z | claude-b17 | B17-T01-08 | TODO -> DONE(T04,T06,T07,T08) / REVIEW(T01,T02,T03,T05) | deploy/, services/observability/, packages/vg_models/registry.py, scripts/model_rollout.py, docs/runbooks/ONCALL.md | compose + kustomize validated offline; metrics verified with replayed calls; images not built (Docker daemon off) |
| 2026-09-24T01:27Z | claude-b16 | B16-T01-09 | TODO -> DONE(T01-T07) / REVIEW(T08,T09) | services/privacy/, config/privacy/, docs/compliance/, tests/compliance/ | compliance suite passes; DPIA needs DPO sign-off; retention values Q10 |
| 2026-09-24T01:27Z | claude-b15 | B15-T01-10 | TODO -> DONE(T09) / REVIEW(T01-T08,T10) | packages/vg_eval/, ml/eval/, .github/workflows/eval.yml | harness runs end to end on synthetic data; real runs need data + trained model; fairness gate threshold Q9 |
| 2026-09-24T01:27Z | claude-b16 | B11/B12 cross-block | edit | services/policy/evidence.py, services/api_gateway/{pipeline,app,grpc_server,webhooks,auth}.py | evidence bodies erasable (chain over body hashes, v1 migration); consent enforced at session start; egress guard on webhooks; /metrics public route |
| 2026-09-23T20:15Z | claude-b14 | B14-T01 | TODO -> REVIEW | ml/export/distill.py | student distillation + CLI; student checkpoints now reload truncated (packages/vg_models/heads/head_a_ssl/model.py, frontend.py) |
| 2026-09-23T20:15Z | claude-b14 | B14-T02 | TODO -> REVIEW | ml/export/quantize.py | dynamic INT8 + report; accuracy delta pending trained student |
| 2026-09-23T20:15Z | claude-b14 | B14-T03 | TODO -> REVIEW | ml/export/to_onnx.py, ml/export/parity_check.py | export + parity; skipped until onnx/onnxruntime installed |
| 2026-09-23T20:15Z | claude-b14 | B14-T04 | TODO -> REVIEW | deploy/triton/, services/inference/ | Triton config + client, cross-call batcher, BatchedHeadA, CPU inference service, edge wrappers |
| 2026-09-23T20:15Z | claude-b14 | B14-T05 | TODO -> REVIEW | docs/benchmarks/LOADTEST.md | CPU i7-13620H: 6 calls FP32 / 8 INT8 at p95 < 700 ms (proxy weights) |
| 2026-09-23T20:15Z | claude-b14 | B14-T06 | TODO -> DONE | services/inference/degrade.py, tests/unit/test_b14_serving.py | load shedding: Head A shed, cheap heads degraded; bounded queue, no audio queued |
| 2026-09-23T20:15Z | claude-b14 | B2 (Q8) | fix | packages/vg_audio/quality.py | tone/DTMF energy fraction mis-scaled by ~N (flagged most audio) + vectorised Goertzel 1087 -> 19 ms; needs human acceptance (Q8) |
| 2026-09-23T20:15Z | claude-b14 | B6 | fix | packages/vg_models/heads/head_c_prosody/head.py | Head C warmup no longer caps process-wide torch threads at 2 (throttled Head A) |
| 2026-09-17T10:00Z | claude-b13 | B13-T01-07 | TODO -> DONE(T01-T06) / REVIEW(T07) | services/ui/, services/api_gateway/app.py (watch WS, session list, /ui static), tests/unit/test_b12_api.py, services/ui/test/ | 267 py + 6 UI + 5 JS tests; no-build ES modules; DoD (non-technical user test) still needs real people |
| 2026-09-17T09:00Z | claude-b12 | B12-T01-09 | TODO -> DONE(T01-T03,T05,T06,T08) / REVIEW(T04,T09) / BLOCKED(T07) | services/api_gateway/, sdks/, docs/api/, examples/, tests/unit/test_b12_api.py | 266 py + 5 JS tests pass; removed invalid `option python_package` from proto (B0 bug: codegen never worked) |
| 2026-09-17T08:00Z | claude-b11 | B11-T01-08 | TODO -> DONE(T01-T04,T06,T07) / REVIEW(T05,T08) | services/policy/, tests/unit/test_b11_policy.py, docs/adr/0008-intent-labels-credential-payment.md | 253 tests pass; ADR 0008 fixes B10 label/schema drift caught by bundle validation |
| 2026-09-17T07:00Z | claude-b10 | B10-T01-07 | TODO -> DONE(T02,T04,T06,T07) / REVIEW(T01,T03,T05) | services/context/, tests/unit/test_b10_context.py | 239 tests pass; ASR models + local LLM deferred to setup |
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
| Q2 | Target deployment hardware for the latency claim (T4? L4? CPU-only?) | — | B14 | 2026-09-24 (claude-b14): CPU numbers measured on the dev laptop (docs/benchmarks/LOADTEST.md); GPU tier still needs a decision + hardware. |
| Q3 | Commercial or research lineage for the primary demo model? | — | B3-T09 | Implied research: ASVspoof 5 EULA + IndicSynth (CC BY-NC 4.0) are both non-commercial. Needs explicit human confirmation. |
| Q4 | Which telephony stack does the pilot tenant actually run? | — | B1 | |
| Q5 | Which Indic languages are in scope for v1 (all 12, or 3–4 done well)? | — | B3, B15 | 2026-09-17 (user): all 12, subject to IndicSynth coverage. |
| Q6 | Add `untrained` to AbstainReason? Untrained Head A currently abstains with reason null. Contract change, needs ADR. | claude-b4 | B0, B9 | 2026-09-17 (user): implement if useful beyond setup. Done: ADR 0006, `untrained` added to schema/proto/models; used by heads A, B, C. |
| Q7 | Accept cross-block edits from B4: `packages/vg_core/sample_store.py` (B0; resolves samples_ref) and `scripts/replay.py` (--real-heads, real model versions)? | claude-b4 | B0, B2 | 2026-09-17 (user): accepted. |
| Q8 | Accept B14's fix to the B2 quality gate (`packages/vg_audio/quality.py`)? The tone/DTMF checks compared raw Goertzel power with sum(x^2), off by a factor of ~N (48000), so white noise and much ordinary speech were flagged `hold_music_or_tone` and every head abstained. Fix normalises to a true bin energy fraction (thresholds 0.5 / 0.02 unchanged) and vectorises Goertzel (1087 -> 19 ms per window, it was also holding the GIL). | claude-b14 | B2, B14 | Recommended: accept. Tradeoff: gate now passes far more real audio to the heads, which is its intended behaviour; re-check false-accept of tones on recorded IVR/hold audio once B3 telephony data exists. |
| Q9 | Maximum per-group false-positive-rate gap for the fairness release gate (B15-T05)? The gate reports `not_configured` until set. | claude-b15 | B15, B17-T07 | Recommended: decide with risk/compliance per attribute (gender, language, accent), e.g. as an absolute FPR gap at the shipped operating point. Not guessed in code (AGENTS §8). |
| Q10 | Confirm retention periods in config/privacy/tenants/default.yaml (evidence 365 d, timeline 90 d, feedback 365 d, raw audio 0 h, voiceprints 730 d inactive). | claude-b16 | B16-T06 | Engineering placeholders; the DPO must set them per tenant and lawful basis. |
| Q11 | Operational alert levels (PSI 0.1/0.25 drift, abstain rate 2x yesterday, alert rate 3x, p95 700 ms) in deploy/observability/alerts.yml and services/observability/drift.py. | claude-b17 | B17-T04, B17-T06 | Start with these conventional values, re-baseline after 2 weeks of shadow mode. They are not detection thresholds. |
| Q12 | ASVspoof 5 licence: the dataset's own LICENSE.txt / README.txt state ODC-By 1.0 (database) and CC BY 4.0 (bona fide speech), both allowing commercial use; ml/data/registry.yaml records 'ASVspoof 5 EULA, commercial_use: false'. Update the registry? | claude-b3 | B3, B4, I7 lineage | Recommended: a human confirms from the shipped licence files and sets license + license_checked_by; lineage of models trained with IndicSynth stays research either way (CC BY-NC). |
| Q13 | Continual Indic training: largest acceptable EER increase on an earlier language / ASVspoof 5 after adding a language (compare.max_regression_eer in ml/training/configs/indic_continual.yaml)? | claude-b4 | B4-T13, B15 | Not set: `compare` reports the deltas without pass/fail. Needs a human decision; tighter limits mean bigger replay samples (disk) or more epochs. |

## 7. Blocked items

| Task | Blocked since | Blocked by | Owner of the blocker | Escalation |
|---|---|---|---|---|
| B4-T01 | 2026-09-17 | deferred user setup (datasets, weights, CUDA torch) | project owner | docs/SETUP_PENDING.md |
| B12-T07 | 2026-09-17 | B14-T03 (ONNX export of distilled Head A) | B14 owner | edge stub abstains until then |
