# VoiceGuard — Block-wise Implementation Plan

**Project:** AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks
**Codename:** VoiceGuard (`vg`) — rename freely, but change it in `SOURCE_OF_TRUTH.md` first, not in code.
**Audience:** the humans and coding agents who will build this. Read `SOURCE_OF_TRUTH.md` before writing any code.

---

## 0. How to read this document

The system is split into **19 blocks (B0–B18)**. Each block is sized so that one person or one agent can own it end-to-end. Each block section has the same shape:

| Field | Meaning |
|---|---|
| **Goal** | What this block is for, in one sentence |
| **Owner profile** | Who should take it |
| **Depends on** | Blocks that must have at least a *stub* merged before this can start |
| **Inputs / Outputs** | The exact data contract — defined in `SOURCE_OF_TRUTH.md §4` |
| **Tasks** | `B<n>-T<nn>` task IDs. These are the same IDs used in `PROJECT_STATUS.md` |
| **Setup required** | Anything a human must install, register for, download, or configure. Full detail in `SETUP_GUIDE.md` |
| **Definition of Done** | The test that proves the block works |

**The single most important rule for parallel work:** every block talks to other blocks *only* through the contracts in `SOURCE_OF_TRUTH.md §4`. Before any real implementation exists, every block ships a **stub** that satisfies the contract with fake data. That means B9 (fusion) can be built and tested on day one against stub heads, and B13 (UI) can be built against a stub risk stream. Nobody waits.

---

## 1. Build order — the walking skeleton first

Do **not** start by training a model. Start by making an end-to-end pipeline that runs with fake scores, then replace fakes with real components one at a time.

**Phase S — Walking Skeleton (target: first 2 days / hackathon hours 0–6)**

```
WAV file replay  →  conditioner (VAD + windows)  →  stub head (random score)
                 →  fusion (EMA)  →  policy (threshold)  →  gateway (gRPC/WS)  →  UI (live gauge)
```

Everything after that is "swap a stub for the real thing." If the skeleton is not working by end of day 2, stop all model work and fix the skeleton.

**Phase 1 — Real detection** (B3 corpus, B4 SSL head, B7 speaker verification, B9 real fusion)
**Phase 2 — Real context** (B10 ASR + intent, B11 policy actions, B13 UI)
**Phase 3 — Real deployment** (B14 ONNX/Triton, B1 telephony adapters, B16 privacy, B17 ops)
**Phase 4 — Proof** (B15 evaluation, B18 demo/submission)

---

## 2. Dependency graph

```
B0 (repo/contracts/CI)
 ├─ B1 capture ──┐
 ├─ B2 conditioning ──┬─ B4 SSL head ──┐
 ├─ B3 corpus ────────┼─ B5 DSP head ──┤
 │                    ├─ B6 prosody ───┼─ B9 fusion+risk ──┬─ B11 policy ── B13 UI
 │                    ├─ B7 speaker ───┤                   │
 │                    └─ B8 liveness/WM┘                   │
 ├─ B10 context/intent ────────────────────────────────────┘
 ├─ B12 API/SDK ── B14 serving ── B17 ops
 ├─ B15 eval harness (parallel from week 1, gates everything)
 ├─ B16 privacy/compliance (parallel, gates B17 sign-off)
 └─ B18 demo/submission (last 15%)
```

**Critical path:** B0 → B2 → B3 → B4 → B9 → B11 → B18. Anything on this path is late if it slips; staff it first.

---

## 3. Suggested team split

| Team | Blocks | Headcount |
|---|---|---|
| **T1 — Audio Platform** | B0, B1, B2, B12, B14, B17 | 1–2 |
| **T2 — ML / Detection** | B3, B4, B5, B6, B7, B8 | 2 |
| **T3 — Risk & Context** | B9, B10, B11 | 1 |
| **T4 — Product Surface** | B13, B18 | 1 |
| **T5 — Evidence & Compliance** | B15, B16 | 1 |

For a 6-person hackathon team, that's exactly one block-group each. For an agent swarm, each block is one agent's scope and `PROJECT_STATUS.md` is the shared board.

---

# BLOCKS

---

## B0 — Repository, contracts, and CI foundation

**Goal.** Make it possible for eight people to write code at once without merge conflicts or interface drift.
**Owner profile.** The most senior engineer. Do this before anything else, in one sitting.
**Depends on.** Nothing.
**Outputs.** Monorepo, proto/JSON schemas, shared `vg_core` package, docker-compose dev stack, CI.

### Tasks
- `B0-T01` Create monorepo per `REPO_LAYOUT.md` (run `scripts/bootstrap_repo.sh`).
- `B0-T02` Write `proto/voiceguard.proto` — `AnalyzeStream`, `AnalyzeFile`, `Enroll`, `RiskEvent`. Generate Python + TS stubs into `packages/vg_core/generated/` and `sdks/js/src/generated/`.
- `B0-T03` Write `schemas/*.json` (JSON Schema draft-2020) for every message in `SOURCE_OF_TRUTH.md §4`. Generate Pydantic models with `datamodel-code-generator`.
- `B0-T04` `packages/vg_core`: config loader (env + YAML), structured JSON logging with `session_id` correlation, `TraceContext`, error taxonomy.
- `B0-T05` `DetectionHead` abstract base class + `HeadRegistry` (entry-point style plugin loading) + `StubHead` that emits a deterministic pseudo-random score from a seed. **This unblocks B9, B11, B13 immediately.**
- `B0-T06` `deploy/docker-compose.dev.yml`: redis, postgres+timescale, minio, prometheus, grafana. One command: `make dev-up`.
- `B0-T07` CI: ruff + black + mypy, pytest with a coverage floor, schema-compat check that fails a PR if a `schemas/` file changed without a version bump.
- `B0-T08` `Makefile` targets: `dev-up`, `dev-down`, `test`, `lint`, `proto`, `fetch-models`, `replay`, `eval`.
- `B0-T09` Seed `SOURCE_OF_TRUTH.md`, `PROJECT_STATUS.md`, `AGENTS.md` into the repo root.

### Setup required
Python 3.11 (pin it — several speech libs break on 3.12), `uv` or conda, Docker + docker-compose, protoc + grpcio-tools, Node 20 for the UI and JS SDK. See `SETUP_GUIDE.md §1`.

### Definition of Done
`make dev-up && make test` passes on a clean clone. A new dev can run `python -m vg.replay --wav sample.wav` and see stub risk events stream to stdout.

---

## B1 — Capture and integration adapters (Layer 0)

**Goal.** Get a normalized PCM stream + call metadata out of any real-world voice channel.
**Owner profile.** Backend/infra engineer with some telephony patience.
**Depends on.** B0.
**Inputs.** RTP / WebSocket / WebRTC media. **Outputs.** `AudioChunk` stream + one `CallMetadata` per session onto the internal bus.

### Tasks
- `B1-T01` **File/WAV replay adapter** (do this first — it unblocks the whole team; nobody should need a PBX to develop). Replays a WAV in real time with configurable jitter and packet loss.
- `B1-T02` **WebSocket audio adapter** — generic `ws://` ingest of 16-bit PCM frames + a JSON header. This is what the browser demo and most cloud telephony vendors will use.
- `B1-T03` **Asterisk adapter** via AudioSocket (or ARI + external media). Ship a working `pjsip.conf` / `extensions.conf` in `deploy/telephony/asterisk/`.
- `B1-T04` **FreeSWITCH adapter** via `mod_audio_stream` (WebSocket fork of the media leg).
- `B1-T05` **Twilio Media Streams adapter** — µ-law 8 kHz base64 frames over WSS; document the ngrok tunnel for local dev.
- `B1-T06` **SIPREC receiver** (stretch, enterprise story) — accept a SIPREC recording session from an SBC, demux the two RTP streams.
- `B1-T07` **Browser/WebRTC capture** for the collaboration-platform demo (`getUserMedia` → AudioWorklet → WS).
- `B1-T08` Metadata envelope extraction: calling number, SIP `P-Asserted-Identity`, trunk ID, source ASN, tenant ID, direction, claimed identity.
- `B1-T09` Session lifecycle: create / heartbeat / teardown, with a reaper for orphaned sessions.

### Setup required (elaborated in `SETUP_GUIDE.md §5`)
- Asterisk 20 in Docker + two softphone accounts (Linphone/Zoiper) to place a test call between two laptops.
- A Twilio trial account, a purchased number, and `ngrok` for local webhook testing. Twilio sends **8 kHz µ-law** — your resampler must be tested against that specifically.
- Codec libraries on the host: `ffmpeg` with G.711/G.722/AMR-NB/Opus, `sox`.

### Definition of Done
The same downstream pipeline produces identical-shaped events whether the source is a WAV replay, a Twilio call, or an Asterisk call. Prove it with one integration test per adapter using a recorded fixture.

---

## B2 — Stream conditioning (Layer 1)

**Goal.** Turn a messy live stream into clean, scoreable analysis windows — and refuse to score bad audio.
**Owner profile.** DSP-minded backend engineer.
**Depends on.** B0. (Consumes B1, but can be developed against the replay adapter.)
**Inputs.** `AudioChunk`. **Outputs.** `AnalysisWindow` (float32 mono 16 kHz array + quality metrics).

### Tasks
- `B2-T01` Jitter buffer + packet-loss concealment for the RTP/WS paths.
- `B2-T02` Resampler to 16 kHz mono, with correct handling of 8 kHz µ-law/A-law inputs. Record the *original* sample rate in metadata — it is a feature, not just plumbing.
- `B2-T03` VAD integration (Silero VAD primary, pyannote 3.x optional) with hysteresis so short pauses don't fragment windows.
- `B2-T04` Windowing: default **3.0 s window, 1.0 s hop**, configurable per tenant. Emit window IDs monotonically per session.
- `B2-T05` **Quality gate** — this is the single highest-value component in the whole pipeline for false-positive control. Reject a window and emit `ABSTAIN` when: voiced speech < 1.5 s, estimated SNR below floor, clipping ratio too high, DTMF/hold-music/ringback detected, or multiple concurrent speakers detected.
- `B2-T06` Loudness normalization (EBU R128 or simple RMS to a fixed dBFS) — but **log the pre-normalization level**, because level statistics themselves carry signal.
- `B2-T07` Music/tone detector to suppress IVR prompts and hold music from being scored.
- `B2-T08` Per-window feature cache (Redis, TTL ≤ 60 s) so multiple heads don't recompute the same spectrogram.

### Definition of Done
Feed a 5-minute call with 40% silence, hold music, and a 3% packet-loss burst. The conditioner emits only well-formed voiced windows, marks the rest `ABSTAIN`, and never crashes. Latency added by this layer < 60 ms p95.

---

## B3 — Data and corpus engineering

**Goal.** Build the training and evaluation data that the entire ML effort depends on — including the proprietary Indic + telephony corpus that is your actual differentiator.
**Owner profile.** ML engineer who is comfortable with data plumbing and GPU batch jobs. **Start this on day 1** — it is on the critical path and it is the longest-lead item.
**Depends on.** B0.
**Outputs.** A versioned manifest (`data/manifests/*.jsonl`) with full provenance per utterance, plus derived audio in object storage.

### Tasks
- `B3-T01` **Dataset registry** `ml/data/registry.yaml`: for each corpus record name, URL, size, license, `commercial_use: true|false`, and the download/verify recipe. **The license field is enforced in code** (see `B3-T09`).
- `B3-T02` Downloaders + checksum verification for: ASVspoof 5, ASVspoof 2019 LA / 2021 LA+DF, In-the-Wild, SpoofCeleb, MLAAD (+ M-AILABS bona fide), PartialSpoof, CodecFake, DFADD, RTCFake, MUSAN, RIRS_NOISES.
- `B3-T03` Downloaders for Indic resources: IndicSynth, SEA-Spoof, Indic-CodecFake, and bona fide corpora — AI4Bharat IndicVoices, Kathbath/IndicSUPERB, Shrutilipi, Svarah, Common Voice Indic subsets.
- `B3-T04` **Unified manifest schema.** One JSONL row per utterance: `{utt_id, path, label(bona_fide|spoof), generator_family, generator_name, language, accent, gender, speaker_id, source_corpus, license, codec_chain, snr_db, rir_id, duration_s, sample_rate}`. Every later split, report, and fairness breakdown reads from this.
- `B3-T05` **Clone zoo** — containerized generators, one Docker image per family because their dependencies conflict violently. Zero-shot TTS: XTTS-v2, OpenVoice v2, F5-TTS, CosyVoice 2, Fish-Speech, StyleTTS2, MeloTTS, Indic Parler-TTS, Bark. Voice conversion: FreeVC, RVC, Seed-VC, kNN-VC. **Diversity of families matters far more than volume from one model.**
- `B3-T06` Batch cloning job: sample bona fide Indic utterances → generate matched spoof utterances → write manifest rows with generator provenance.
- `B3-T07` **Channel destruction simulator** (`ml/data/channel/`). Apply *identically to bona fide and spoof*: 8 kHz downsample; G.711 µ-law/A-law, G.729, AMR-NB, Opus @ 6–24 kbps, EVS; packet loss 1/3/5% with PLC; jitter; RIR convolution; MUSAN noise at 5–20 dB SNR; WebRTC AGC; RNNoise or DeepFilterNet suppression. Each transform records itself into `codec_chain`.
- `B3-T08` **Symmetry test** — an automated test that trains a tiny classifier on *channel features only* and asserts it cannot separate bona fide from spoof above chance. If it can, your augmentation is asymmetric and every downstream number is a lie. This test must be in CI.
- `B3-T09` **License gate.** Training configs carry `allow_noncommercial: bool`. When `false`, the data loader hard-fails if any manifest row has `commercial_use: false`. IndicSynth (CC BY-NC 4.0) and SEA-Spoof are academic-only — fine for the hackathon and a paper, not for a shipped product. Two model lineages must be trainable: `research` and `commercial`.
- `B3-T10` Speaker-disjoint, generator-disjoint split generator with a fixed seed; write `splits/*.json`.
- `B3-T11` Consent + ethics register: record who consented to be cloned for demo voices. Never clone a real executive or official. Ship `docs/ETHICS.md`.

### Setup required (elaborated in `SETUP_GUIDE.md §3`)
This is the most setup-heavy block in the project.
- **Storage:** budget 1.5–3 TB. ASVspoof 5 alone is large; IndicSynth is 4,000+ hours.
- **Access:** ASVspoof 5 requires accepting a EULA / registering for the download; several corpora are on Zenodo with request forms; HF datasets need `huggingface-cli login` and acceptance of gated terms. **Start these requests on day 1 — approvals can take days.**
- **GPU:** corpus generation is a bigger GPU consumer than training. A single A100 or 2× RTX 4090 for a weekend produces a solid 50–200 hour clone corpus.
- **Per-generator environments:** do not try to install XTTS-v2, F5-TTS, and RVC in one Python env. One Docker image per generator, orchestrated by a job runner. Budget a full day just for this.

### Definition of Done
`make corpus-report` prints a table of hours by language × generator family × codec, and the symmetry test in `B3-T08` passes.

---

## B4 — Head A: SSL anti-spoof detector (primary)

**Goal.** The main neural detector: an SSL speech front-end with a lightweight anti-spoof back-end.
**Owner profile.** The strongest ML engineer.
**Depends on.** B0 (head interface), B3 (data), B15 (eval harness — build in parallel).
**Inputs.** `AnalysisWindow`. **Outputs.** `HeadScore(head_id="A")`.

### Tasks
- `B4-T01` **Stage 0 — reproduce.** Get `SSL_Anti-spoofing` (TakHemlata) or AASIST running on ASVspoof 2019 LA and match the published EER. One day, pure sanity check. If you cannot reproduce a known number, your pipeline is broken and every later number is meaningless.
- `B4-T02` Front-end wrapper: `facebook/wav2vec2-xls-r-300m` and `microsoft/wavlm-large`, with **layer selection** — expose hidden states and default to a learned weighted sum over layers 5–9 rather than the final layer. Intermediate layers carry the artifact information; final layers throw it away in favor of semantics.
- `B4-T03` Back-end: **Nes2Net** primary (processes high-dimensional SSL features without a dimensionality-reduction bottleneck; ~87% back-end compute reduction), AASIST as the secondary/ensemble head.
- `B4-T04` Loss: **OC-Softmax / AM-Softmax**, not plain BCE. One-class objectives model bona fide as a compact region and generalize far better to unseen generators.
- `B4-T05` Augmentation in the training loop: RawBoost + the B3 channel simulator applied on the fly.
- `B4-T06` **Stage 1** train on ASVspoof 5 train partition → record baseline.
- `B4-T07` **Stage 2** multi-corpus pooling: + SpoofCeleb, MLAAD, DFADD, CodecFake, RTCFake, IndicSynth, and your own Indic channel-degraded corpus. **Balance sampling by attack family, not by utterance count**, or the largest corpus silently dominates.
- `B4-T08` **Stage 3** robustness, in descending value-per-effort: aggressive channel augmentation → layer-wise weighted sum → SAM optimizer → mixup across real/fake pairs → feature-consistency self-distillation across two augmented views → domain-adversarial training on generator-ID and codec-ID → LoRA adapters instead of full fine-tune → 2-front-end ensemble (XLS-R + WavLM).
- `B4-T09` Inference wrapper implementing `DetectionHead`, batched, with a hard latency budget and a timeout that returns `ABSTAIN` rather than blocking the pipeline.
- `B4-T10` **Continual learning hook** — a documented procedure and script to ingest a new attack family and update without catastrophic forgetting (replay buffer + EWC/region-based constraint). Banks will ask about your model update cadence; have an answer.

### Setup required (`SETUP_GUIDE.md §2, §4`)
- HF auth and ~40 GB of checkpoint downloads. Pin revisions by commit SHA in `models/registry.yaml`.
- GPU: 1× A100 40 GB, or 2× 4090, or Colab Pro/Kaggle T4 for the hackathon lane (freeze the front-end, train only the back-end).
- Experiment tracking: MLflow or W&B in local/offline mode (data residency — see B16).
- Set `PYTORCH_CUDA_ALLOC_CONF`, pin `transformers`/`torch` versions; SSL front-ends are sensitive to version drift in `fairseq`-derived code.

### Definition of Done
Reproduced baseline (B4-T01) plus a cross-dataset table from B15 showing EER on In-the-Wild, RTCFake, and a held-out Indic split — with the degradation reported honestly. Inference p95 within budget on the target hardware.

---

## B5 — Head B: DSP artifact and acoustic-environment analysis

**Goal.** A cheap, explainable detector that fails *differently* from the neural head, plus the scene-consistency check.
**Owner profile.** Signal-processing engineer. Great block for someone who wants depth without huge GPU needs.
**Depends on.** B0, B2.
**Outputs.** `HeadScore(head_id="B")` with a human-readable evidence dict.

### Tasks
- `B5-T01` Feature extractors: CQT, LFCC, modified group delay, long-term average spectrum, spectral roll-off, bicoherence.
- `B5-T02` Vocoder fingerprinting: detect characteristic harmonic/phase signatures of HiFi-GAN, BigVGAN, neural audio codecs (EnCodec/SoundStream-family).
- `B5-T03` Lightweight classifier (GBDT or small CNN) over these features. Must run on CPU in a few milliseconds.
- `B5-T04` **Room Impulse Response / scene consistency.** Estimate the reverberation profile (RT60, direct-to-reverberant ratio) of the *voice track*, and separately of the *background*. A synthetic voice is generated in an anechoic digital void; attackers then mix in downloaded background noise. When the vocal track's decay profile mathematically contradicts the background's reverberation profile, you have a composite stream. This survives codec compression far better than high-frequency artifacts do, and it is a genuinely distinctive component.
- `B5-T05` Channel/codec identifier — infer the likely codec chain and expose it, both as a feature to fusion and as an operator-facing explanation.
- `B5-T06` Explanation strings: every score ships with 1–3 plain-English reasons ("no breath energy below 300 Hz", "anechoic voice over reverberant background").

### Definition of Done
Runs CPU-only under 15 ms per window, and on a held-out set adds measurable value in the B9 fusion ablation (fused EER improves versus Head A alone).

---

## B6 — Head C: prosody and behavioral analysis

**Goal.** Model the *human-ness* of speech rhythm — cues that survive telephony compression when spectral artifacts do not.
**Owner profile.** ML/speech engineer.
**Depends on.** B0, B2.
**Outputs.** `HeadScore(head_id="C")`.

### Tasks
- `B6-T01` F0 extraction and contour dynamics (range, slope entropy, declination), jitter and shimmer.
- `B6-T02` **Breath-group detection** — inhalation events, breath energy, breath-to-phrase ratio. Neural TTS frequently omits or regularizes breathing.
- `B6-T03` Pause-length distribution and speaking-rate variance; measure over-regularity, which is the classic TTS tell.
- `B6-T04` Disfluency and filler detection (`um`, `uh`, restarts, self-corrections) from the ASR stream produced in B10.
- `B6-T05` Temporal model (small BiLSTM or transformer over per-frame prosodic features) producing a calibrated score.
- `B6-T06` **Indic prosody care.** Retroflex consonants, geminates, and code-switched English are routinely misread as synthetic artifacts by English-trained models. Evaluate per-language and add an accent-aware normalization step. This is where naive systems generate catastrophic false-positive rates on Indian speakers.

### Definition of Done
Beats chance on a codec-destroyed evaluation set where the SSL head's advantage has narrowed — that's the whole point of this head.

---

## B7 — Head D: speaker verification and enrollment

**Goal.** Compare the live voice against an enrolled voiceprint. For the CXO-impersonation scenario this is arguably *more* useful than spoof detection.
**Owner profile.** ML engineer + a bit of backend for the enrollment service.
**Depends on.** B0, B2.
**Outputs.** `HeadScore(head_id="D")` carrying cosine similarity and a mismatch probability.

### Tasks
- `B7-T01` Embedding model integration: SpeechBrain `spkrec-ecapa-voxceleb`, WeSpeaker, or NVIDIA TitaNet-L. Benchmark all three on Indian-accented audio before choosing.
- `B7-T02` **Enrollment service:** REST/gRPC endpoint, minimum audio duration, quality checks, multi-session enrollment, re-enrollment policy.
- `B7-T03` **Voiceprint vault:** encrypted at rest, per-tenant key, embeddings only — never raw enrollment audio beyond a short verification window. Voiceprints are biometric data (B16).
- `B7-T04` Scoring: cosine similarity plus score normalization (AS-norm) against a cohort, so thresholds are stable across channels.
- `B7-T05` Channel compensation — an enrollment recorded on a headset and a call arriving over 8 kHz AMR are a domain-mismatch problem. Enroll through the channel simulator too.
- `B7-T06` Handle the no-enrollment case cleanly: emit `ABSTAIN`, never a fabricated score. Most inbound callers will not be enrolled.

### Definition of Done
On your demo set, a cloned "CEO" voice scores measurably lower against the real enrollment than the genuine CEO does, and the gap is reported with a confidence interval.

---

## B8 — Heads E & F: active liveness and watermark probe

**Goal.** Two cheap, high-precision signals: challenge-response and watermark detection.
**Owner profile.** Any engineer; good scoped starter block.
**Depends on.** B0, B2, B10 (ASR for challenge verification).
**Outputs.** `HeadScore(head_id="E")`, `HeadScore(head_id="F")`.

### Tasks
- `B8-T01` **Challenge generator** — random nonce phrases, code-switched digit strings (e.g. Hindi–English mixed), or an unexpected word order. Real-time clone pipelines have measurable pipeline latency and break on unexpected code-switching.
- `B8-T02` Response verifier: ASR-match the challenge, measure response latency and prosodic appropriateness. Latency distribution is often the strongest single tell.
- `B8-T03` Operator UI hook so an agent can trigger a challenge mid-call (B13).
- `B8-T04` **AudioSeal watermark detector** (`facebookresearch/audioseal`). Sample-level localized detection, survives compression and re-encoding, and runs far faster than a classifier. When present it is near-conclusive positive evidence of synthesis.
- `B8-T05` Detectors for any other commercial TTS watermark schemes you can obtain.
- `B8-T06` **Asymmetry rule, enforce in code:** watermark present → strong evidence of synthesis. Watermark absent → *no evidence whatsoever*. The fusion layer must never treat absence as an exoneration.

### Definition of Done
An AudioSeal-watermarked clip is flagged within one window; an unwatermarked clip changes the fused score by exactly zero.

---

## B9 — Fusion, calibration, and the temporal risk engine (Layer 3)

**Goal.** Turn a stream of noisy per-head scores into one stable, calibrated, explainable session risk score.
**Owner profile.** ML engineer with good statistical instincts. This is the most conceptually delicate block.
**Depends on.** B0 (can be built entirely against `StubHead`).
**Inputs.** `HeadScore[]`. **Outputs.** `FusedWindowScore`, `SessionRisk`.

### Tasks
- `B9-T01` **Per-head calibration.** Temperature scaling or Platt scaling on a held-out set that matches deployment conditions. Raw logits from different heads have wildly different scales and cannot be averaged. Store calibration parameters alongside the model version.
- `B9-T02` **Cross-head fusion.** Start with logistic regression or a small GBDT over calibrated scores — auditable and it beats naive averaging. Must handle heads that abstain (missing inputs) without silently substituting zeros.
- `B9-T03` **Temporal fusion.** Sequential Bayesian update or a two-state HMM (genuine ↔ synthetic) over per-window log-likelihood ratios. The score should *firm up* as the call proceeds rather than flicker.
- `B9-T04` **Any-segment trigger** for partial splicing: if an attacker injects a few synthetic seconds into an otherwise genuine call, session risk takes the maximum over segments, not the mean. Keep both statistics; expose both.
- `B9-T05` **Four output states:** `LOW`, `ELEVATED`, `HIGH`, `ABSTAIN`. Abstain is a first-class output, not an error.
- `B9-T06` **Operating point selection from a cost model, not from EER.** In a contact center handling 50,000 calls a day, a 2% false-positive rate is 1,000 wrongly-flagged customers. Pick thresholds at a fixed FPR (0.1%, 1%) with an explicit abstain band.
- `B9-T07` Contribution attribution — per-head contribution to the final score, for the evidence bundle and the UI.
- `B9-T08` Score timeline persistence (Timescale) with a replay endpoint for forensics.

### Definition of Done
Expected calibration error under 0.05 on a held-out deployment-like set; score stability measured (no oscillation between states more than N times per minute); the ablation table showing each head's marginal contribution is generated automatically.

---

## B10 — Contextual enrichment and intent analysis (Layer 4)

**Goal.** The non-acoustic half of the defense — and the part most competing solutions ignore entirely.
**Owner profile.** Backend + LLM engineer.
**Depends on.** B0, B2.
**Outputs.** `ContextSignals`.

### Tasks
- `B10-T01` **Streaming ASR.** AI4Bharat IndicConformer for Indic languages, Whisper large-v3 as fallback/English. Partial-hypothesis streaming with a rolling transcript buffer.
- `B10-T02` Language identification and code-switch detection (essential for Indian call audio, which switches mid-sentence).
- `B10-T03` **Social-engineering intent classifier** over the rolling transcript. Target the coercion script: manufactured urgency, secrecy demands ("don't tell the team"), authority pressure, unusual channel, resistance to call-back, unusual beneficiary. Run a local LLM (Qwen2.5-7B / Llama-3.1-8B via vLLM or Ollama) — **local, not an API, for data residency.** Ship a labeled seed set of ~200 scripted scenarios plus prompt + few-shot examples.
  This head catches attacks where the audio is a *real human* reading a fraud script — an entire attack class the acoustic detectors are blind to.
- `B10-T04` **Metadata risk scorer:** first-time caller, CLI vs registered-name mismatch, international origin presenting a domestic CLI, VoIP trunk reputation, off-hours, velocity of calls from the same source.
- `B10-T05` **Transaction context connector** (mock a core-banking API for the demo): amount, new beneficiary, deviation from the caller's historical pattern, privileged-access request.
- `B10-T06` Combine into `ContextSignals` with per-signal explanations. Feed to B9/B11 as `final_risk = f(voice_authenticity, speaker_mismatch, metadata_risk, transaction_sensitivity, intent_risk)`.
- `B10-T07` PII redaction on the transcript before it is stored or shown (account numbers, OTPs, card numbers).

### Setup required (`SETUP_GUIDE.md §6`)
- IndicConformer / IndicWhisper checkpoints from AI4Bharat; Whisper large-v3 (~3 GB).
- A local LLM server: Ollama for the fast path, vLLM for throughput. ~16 GB VRAM for a 7B at reasonable latency, or a quantized 7B on CPU for the demo.
- Decide and document the intent-classifier latency budget; it runs *asynchronously* and must never block the audio path.

### Definition of Done
A scripted fraud call with a genuine human voice raises `intent_risk` to HIGH while `voice_authenticity` stays LOW — and your UI shows both. That's the demo moment that separates you from a plain classifier.

---

## B11 — Policy, alerting, and response (Layer 5)

**Goal.** Turn a risk score into an action, configurably, per tenant.
**Owner profile.** Backend engineer.
**Depends on.** B9, B10.
**Outputs.** `PolicyDecision`, `EvidenceBundle`, outbound alerts.

### Tasks
- `B11-T01` Rules engine with per-tenant threshold profiles. A wealth-management desk and a retail helpline need different operating points; make this data, not code.
- `B11-T02` Action library: in-UI banner, forced step-up verification (call-back on the registered number, app-push MFA, dual approval), transaction hold, supervisor escalation, SOC ticket, call recording flag.
- `B11-T03` Risk-tiered thresholds bound to *transaction sensitivity*: e.g. require 0.95 spoof probability to flag a balance inquiry, but only 0.75 to hold a high-value wire transfer.
- `B11-T04` **Evidence bundle** — immutable record: which heads fired, score timeline, transcript snippet, metadata, model versions, the exact threshold profile applied. Content-hashed and stored for audit.
- `B11-T05` Multi-channel notification: WebSocket to the agent UI, SMS/email, in-app push, webhook to the enterprise SIEM.
- `B11-T06` Pre-transaction warning prompts with a recommended secondary verification, worded for a non-technical frontline agent.
- `B11-T07` **Shadow mode** flag — score everything, alert nothing. This is how you deploy into a real tenant before you trust your thresholds. Make it the default for new tenants.
- `B11-T08` Analyst feedback loop: agent marks an alert true/false positive; that label flows back to B15 and B4's continual-learning buffer.

### Definition of Done
The same call, under two tenant profiles, produces two different action sets, and both are fully explained in the evidence bundle.

---

## B12 — Platform APIs and SDKs (Layer 7)

**Goal.** Make the system integrable in an afternoon by someone who has never seen it.
**Owner profile.** Backend/DX engineer.
**Depends on.** B0 contracts.

### Tasks
- `B12-T01` **gRPC bidirectional streaming** — audio in, `RiskEvent` out. The primary real-time interface.
- `B12-T02` **REST API** — session create/query, batch/post-call file analysis, enrollment, evidence retrieval, policy profile CRUD.
- `B12-T03` **Webhooks** for alerts, with HMAC signing and retry-with-backoff.
- `B12-T04` AuthN/AuthZ: per-tenant API keys or mTLS, scopes, rate limits, request quotas.
- `B12-T05` **Python SDK** (`sdks/python`) — the one most integrators will use.
- `B12-T06` **JS/TS SDK** (`sdks/js`) for browser and Node.
- `B12-T07` **Edge/mobile SDK stub** — ONNX Runtime Mobile / TFLite wrapper, showing on-device inference (this is the privacy story made concrete).
- `B12-T08` OpenAPI + generated docs, plus a 10-minute quickstart that runs against the replay adapter with no telephony at all.
- `B12-T09` Reference connectors: Asterisk, FreeSWITCH, Twilio, and a browser extension for collaboration platforms.

### Definition of Done
A developer who has never seen the repo goes from clone to a live risk score on their own WAV file in under 10 minutes, following only the quickstart.

---

## B13 — Agent and analyst UI

**Goal.** The surface the judge/customer actually looks at. Underinvested in most projects; it is what they remember.
**Owner profile.** Frontend engineer.
**Depends on.** B0 contracts (build against the stub stream from day 1).

### Tasks
- `B13-T01` Live call view: risk gauge, score timeline sparkline, per-head contribution bars, current state badge including a clearly-styled ABSTAIN.
- `B13-T02` Live transcript pane with intent flags highlighted inline.
- `B13-T03` Alert banner with the recommended action and one-click step-up ("Verify by call-back", "Trigger challenge phrase").
- `B13-T04` Post-call forensics view: full timeline scrub, evidence bundle, model versions, export to PDF.
- `B13-T05` Admin console: tenant threshold profiles, enrollment management, shadow-mode toggle.
- `B13-T06` Analyst feedback buttons (true/false positive) wired to B11-T08.
- `B13-T07` Demo mode: side-by-side genuine vs cloned call playback with synchronized scoring.

### Definition of Done
Someone non-technical watches the screen during a live cloned call and correctly says "the system is telling me not to trust this caller" without anyone explaining the UI.

---

## B14 — Serving, optimization, and latency

**Goal.** Hit the real-time budget on hardware a customer will actually buy.
**Owner profile.** ML systems engineer.
**Depends on.** B4 (a trained model), B12.

### Tasks
- `B14-T01` **Distillation** — distill the big ensemble into a student: truncated wav2vec2 (first 6–9 layers) plus a small back-end. Keep the large model server-side for post-call forensics and for adjudicating borderline calls.
- `B14-T02` INT8 quantization; measure the accuracy cost and publish it, don't hide it.
- `B14-T03` ONNX export + ONNX Runtime inference path; verify numerical parity with PyTorch within tolerance.
- `B14-T04` NVIDIA Triton deployment with dynamic batching for the server tier; TFLite/ORT-Mobile for edge.
- `B14-T05` Load testing: concurrent sessions per GPU, p50/p95/p99 per-window latency, cost per 1,000 call-minutes.
- `B14-T06` Backpressure and graceful degradation: when overloaded, drop to the cheap heads (B, C, D) and mark reduced confidence rather than queueing audio.

### Targets
First score by **3.5 s of speech**, updates every **1 s**, p95 end-to-end **< 300 ms** per window on a single T4/L4 with batching, or **< 700 ms CPU-only** with the distilled model. Publish the measured numbers, not the targets.

### Definition of Done
A load test report in `docs/benchmarks/` with real numbers on named hardware.

---

## B15 — Evaluation harness and benchmarking

**Goal.** The thing that makes your claims believable. Build it in week 1, before you have anything to evaluate.
**Owner profile.** ML engineer with a skeptical temperament. Ideally *not* the person who trains the model.
**Depends on.** B0, B3.

### Tasks
- `B15-T01` Harness that takes a model script + checkpoint and runs all protocols automatically (AUDDT-style, or adopt AUDDT/DeepFense directly to save weeks).
- `B15-T02` **Leave-one-generator-out** — hold out entire attack families; report per-family EER. This is the number that predicts real-world performance.
- `B15-T03` **Cross-dataset:** train on ASVspoof 5, test on In-the-Wild, SpoofCeleb, MLAAD, IndicSynth, RTCFake. Publish the degradation honestly — it is a finding, not a failure, and honesty here reads as competence.
- `B15-T04` **Per-language and per-accent breakdown** across the 12 Indic languages plus Indian-accented English. Expect a collapse before Indic fine-tuning; the before/after chart is your single best slide. (Published evidence: a detector at ~1.25% EER on standard sets collapsed to ~43.8% EER on SEA-Spoof.)
- `B15-T05` **Fairness:** per-gender and per-language FPR gaps. Uneven false-positive rates across accents is a hard deployment blocker, not a nice-to-have.
- `B15-T06` Per-codec and per-SNR degradation curves.
- `B15-T07` **Operational metrics:** time-to-first-alert in seconds of speech, score stability, latency percentiles, throughput per GPU, cost per 1,000 call-minutes.
- `B15-T08` **Adversarial / laundering robustness:** re-encoding, speed perturbation, pitch shift, additive noise, adversarial perturbation, Malacopula-style attacks. Penetration-test your own detector and report what breaks it.
- `B15-T09` Metrics implementation: EER, min t-DCF and a-DCF (for ASVspoof comparability), AUC, partial-AUC at low FPR, expected calibration error.
- `B15-T10` Auto-generated `docs/benchmarks/REPORT.md` regenerated by CI on every model version.

### Definition of Done
`make eval MODEL=<version>` produces the full report unattended, and every claim anywhere in your submission traces to a row in it.

---

## B16 — Privacy, compliance, and data governance (Layer 6)

**Goal.** Make compliance an architectural property rather than a slide. In Indian BFSI this dictates the architecture; it is not a peripheral feature.
**Owner profile.** Engineer who reads regulations carefully; pairs with a legal reviewer if you have one.
**Depends on.** B0; reviews everything.

### Tasks
- `B16-T01` **Ephemeral processing:** raw audio lives in volatile memory only. Once embeddings and scores are computed, the buffer is overwritten. Write an actual test that asserts no raw PCM reaches disk on the default path.
- `B16-T02` **Feature-only logging:** persist uninvertible embeddings and telemetry (`p_spoof`, codec, intent labels), never waveforms. Raw audio retained only for flagged calls, only for a short window, behind an explicit retention policy and a documented lawful basis.
- `B16-T03` **On-device / on-prem inference by default**; central processing only on explicit tenant opt-in.
- `B16-T04` **Consent matrix** implemented as configuration, distinguishing legal bases: fraud detection and security (legitimate use — no interruptive in-call consent, but must appear in the privacy notice), marketing/analytics (explicit consent required, cannot be bundled), and **speaker enrollment / voiceprinting (explicit verifiable consent + notice)**. Voiceprints are biometric data under DPDP 2023, GDPR, and CCPA.
- `B16-T05` **Data residency:** RBI mandates that banking voice data remain physically in India — which precludes offshore cloud LLM/detection APIs. This is exactly why B10's LLM must be local. Document the deployment topology that satisfies this.
- `B16-T06` Retention/erasure automation, DSAR support, purpose limitation enforcement.
- `B16-T07` Per-tenant encryption keys, key rotation, encrypted voiceprint vault.
- `B16-T08` **DPIA template** and a completed example (`docs/compliance/DPIA.md`), plus an audit-log schema.
- `B16-T09` Model card and data statement for every released model version (training corpora, licenses, known failure modes, fairness results).

### Definition of Done
An automated compliance test suite passes: no raw audio at rest on the default path, retention jobs verified, consent basis enforced per processing purpose, and the DPIA is written.

---

## B17 — Deployment, observability, and operations

**Goal.** Run it somewhere other than a laptop.
**Owner profile.** DevOps/platform engineer.
**Depends on.** B12, B14.

### Tasks
- `B17-T01` Dockerfiles per service; multi-stage, slim, non-root.
- `B17-T02` `docker-compose.prod.yml` for a single-node on-prem deployment (this is the realistic BFSI starting point).
- `B17-T03` Helm chart / k8s manifests for the scaled deployment.
- `B17-T04` Prometheus metrics: per-head latency, abstain rate, alert rate, queue depth, GPU utilization, model version in use.
- `B17-T05` Grafana dashboards: an operations dashboard and a fraud-analytics dashboard.
- `B17-T06` **Drift monitoring** — track the score distribution over time; alert when the bona fide score distribution shifts, which is your early warning that a new generator family has arrived in the wild.
- `B17-T07` Model registry and blue/green model rollout with instant rollback.
- `B17-T08` Runbook: what an on-call engineer does when the abstain rate spikes, when latency degrades, when the alert rate triples.

### Definition of Done
`make prod-up` on a clean VM brings up the full stack; the Grafana dashboard shows live traffic from a replayed call set.

---

## B18 — Demo, documentation, and submission

**Goal.** Convert the work into something a judge or buyer understands in five minutes.
**Owner profile.** Whoever communicates best. Start at 60% completion, not at 95%.
**Depends on.** Everything.

### Tasks
- `B18-T01` **Demo script:** a judge's own voice is cloned (with consent, on the spot) and used to attempt a fund-transfer authorization; the system flags it live. Rehearse the failure modes.
- `B18-T02` Second demo: a genuine human reading a fraud script — acoustic score stays LOW, intent score goes HIGH. This shows layered defense, which is the intellectually serious part of the pitch.
- `B18-T03` Before/after chart: EER on Indic audio before and after Indic + channel fine-tuning. One chart, carries the whole technical story.
- `B18-T04` Architecture diagram, README, and a 3-minute video.
- `B18-T05` Offline demo fallback — pre-recorded calls, no network. Conference wifi will fail; assume it.
- `B18-T06` **The two things to say out loud:**
  1. *Detection is advisory, never authoritative.* There is irreducible overlap between bona fide and synthetic speech under realistic conditions, so any detector has a non-zero error floor. The system raises friction and triggers out-of-band verification; it must never be the sole gate on a transaction. Saying this makes you more credible, not less — it is exactly what a bank's risk team wants to hear.
  2. *The strongest defenses are partly non-acoustic.* Speaker verification, transaction-context anomaly detection, transcript-level intent analysis, and enforced call-back all degrade gracefully when the audio detector is fooled. A team that ships five layers beats a team with a slightly better EER on one.

---

## 4. Milestones

### Hackathon lane (36–48 hours)
| Hours | Deliverable | Blocks |
|---|---|---|
| 0–6 | Walking skeleton: capture → VAD → stub score → dashboard | B0, B1-T01/T02, B2, B13-T01 |
| 6–16 | Real pretrained SSL checkpoint + ECAPA speaker verification vs a pre-enrolled "CEO" voiceprint + temporal smoothing | B4-T01/T09, B7, B9-T03 |
| 16–26 | 3–5 hour Indic clone corpus (XTTS-v2 over Common Voice Hindi/Tamil), codec augmentation, back-end fine-tune; produce the before/after EER chart | B3, B4-T07, B15-T04 |
| 26–36 | Risk engine, thresholds, mock transaction context, agent UI with call-back prompt, LLM intent flag | B9, B10, B11, B13 |
| 36–48 | Privacy story, API docs, rehearsed live demo | B16, B12-T08, B18 |

### 12-week lane
| Weeks | Focus | Blocks |
|---|---|---|
| 1–2 | Repo, contracts, data infrastructure, eval harness, baseline reproduction | B0, B3-T01…T04, B15, B4-T01 |
| 3–6 | Full attack-corpus generation + channel simulation; Stage 2–3 training; all heads | B3, B4–B8 |
| 7–9 | Fusion, context/intent, distillation, ONNX/Triton serving, load testing, SDKs | B9, B10, B14, B12 |
| 10–12 | Policy, UI, one contact-center integration, **shadow-mode** threshold tuning on live traffic before enabling alerts, DPIA | B11, B13, B1, B16, B17 |

---

## 5. Risks and the standing mitigations

| Risk | Mitigation | Owner block |
|---|---|---|
| Detector looks great in-domain, fails in the wild | Leave-one-generator-out + cross-dataset from week 1; never quote a single in-domain EER | B15 |
| Telephony codecs destroy the artifacts the model relies on | Symmetric channel destruction in training; prosody and RIR heads that survive compression | B3, B5, B6 |
| Indic accents produce catastrophic false positives | Indic corpus + per-language fairness gate that blocks release | B3, B15 |
| Non-commercial dataset licenses contaminate a commercial model | Enforced license gate; two model lineages | B3-T09 |
| False positives anger real customers | Quality gate, abstain band, cost-model thresholds, shadow mode | B2, B9, B11 |
| Team blocked waiting on models/data | Stubs for every contract from day 1; replay adapter so nobody needs a PBX | B0, B1-T01 |
| Demo fails live | Offline fallback, rehearsed script | B18-T05 |
