# REPO LAYOUT — initial file structure

Run `scripts/bootstrap_repo.sh` to create this tree with placeholder files. Ownership per directory is in `SOURCE_OF_TRUTH.md §5`.

```
voiceguard/
├── README.md
├── SOURCE_OF_TRUTH.md              # canonical spec — read first
├── PROJECT_STATUS.md               # shared task board — agents update this
├── AGENTS.md                       # working protocol for agents and humans
├── IMPLEMENTATION_PLAN.md
├── SETUP_GUIDE.md
├── REPO_LAYOUT.md
├── Makefile
├── pyproject.toml
├── .env.example
├── .gitignore                      # /data, *.wav, models/*.bin, .venv
│
├── proto/
│   └── voiceguard.proto            # gRPC service + messages          [B0]
│
├── schemas/                        # JSON Schema, one file per message [B0]
│   ├── call_metadata.schema.json
│   ├── audio_chunk.schema.json
│   ├── analysis_window.schema.json
│   ├── head_score.schema.json
│   ├── fused_window_score.schema.json
│   ├── context_signals.schema.json
│   ├── session_risk.schema.json
│   ├── policy_decision.schema.json
│   └── evidence_bundle.schema.json
│
├── packages/                       # importable libraries, no I/O at import time
│   ├── vg_core/                                                       [B0]
│   │   ├── config.py               # env + YAML layered config
│   │   ├── logging.py              # structured JSON, session_id correlation
│   │   ├── models.py               # Pydantic models generated from schemas/
│   │   ├── bus.py                  # redis/kafka abstraction
│   │   ├── head_api.py             # DetectionHead ABC + HeadRegistry
│   │   ├── stub_head.py            # deterministic fake scorer — unblocks B9/B11/B13
│   │   ├── versioning.py           # model_version / calibration_version helpers
│   │   └── generated/              # protoc output, gitignored
│   ├── vg_audio/                                                      [B2]
│   │   ├── resample.py
│   │   ├── vad.py
│   │   ├── windowing.py
│   │   ├── quality.py              # the quality gate / abstain logic
│   │   ├── features.py             # shared spectrogram cache
│   │   └── codecs.py
│   ├── vg_models/
│   │   ├── heads/
│   │   │   ├── head_a_ssl/         # XLS-R/WavLM + Nes2Net             [B4]
│   │   │   ├── head_b_dsp/         # CQT/LFCC/MGD + RIR scene check    [B5]
│   │   │   ├── head_c_prosody/                                        [B6]
│   │   │   ├── head_d_speaker/     # ECAPA/TitaNet + AS-norm           [B7]
│   │   │   ├── head_e_liveness/                                       [B8]
│   │   │   └── head_f_watermark/   # AudioSeal                        [B8]
│   │   ├── calibration.py                                             [B9]
│   │   └── registry.py             # model loading, version pinning
│   └── vg_eval/                                                       [B15]
│       ├── metrics.py              # EER, t-DCF, a-DCF, pAUC, ECE
│       ├── protocols.py            # LOGO, cross-dataset, per-language
│       ├── fairness.py
│       └── report.py               # renders docs/benchmarks/REPORT.md
│
├── services/                       # deployable processes, thin over packages/
│   ├── ingest/                                                        [B1]
│   │   ├── main.py
│   │   └── adapters/
│   │       ├── replay_wav.py       # BUILD THIS FIRST
│   │       ├── websocket_pcm.py
│   │       ├── asterisk_audiosocket.py
│   │       ├── freeswitch_ws.py
│   │       ├── twilio_stream.py
│   │       ├── siprec.py
│   │       └── webrtc.py
│   ├── conditioner/                                                   [B2]
│   ├── inference/                  # hosts the heads, batches windows  [B4-B8]
│   ├── fusion/                     # calibration + temporal + risk     [B9]
│   ├── context/                    # ASR, LID, intent LLM, metadata    [B10]
│   ├── policy/                     # rules, actions, evidence bundles  [B11]
│   ├── enrollment/                 # voiceprint enrol + vault          [B7]
│   ├── privacy/                    # retention, erasure, consent, DSAR [B16]
│   ├── api_gateway/                # gRPC + REST + webhooks            [B12]
│   └── ui/                         # React/Next dashboard              [B13]
│       ├── src/pages/live/
│       ├── src/pages/forensics/
│       ├── src/pages/admin/
│       └── src/components/RiskGauge.tsx
│
├── ml/
│   ├── data/                                                          [B3]
│   │   ├── registry.yaml           # datasets + LICENSE + commercial_use
│   │   ├── download/               # one script per corpus
│   │   ├── manifest.py             # unified manifest schema + writer
│   │   ├── splits.py               # speaker/generator-disjoint splits
│   │   ├── license_gate.py         # hard-fails on restricted rows
│   │   ├── generators/             # one Docker image per TTS/VC family
│   │   │   └── <family>/Dockerfile
│   │   ├── clone_job.py            # batch cloning orchestrator
│   │   └── channel/                # THE channel destruction simulator
│   │       ├── codecs.py           # g711/g729/amr/opus/evs
│   │       ├── packet_loss.py      # loss + PLC
│   │       ├── rir.py              # RIR convolution
│   │       ├── noise.py            # MUSAN mixing at target SNR
│   │       ├── webrtc_chain.py     # AGC + RNNoise/DeepFilterNet
│   │       └── test_symmetry.py    # CI gate for invariant I3
│   ├── training/                                                      [B4]
│   │   ├── configs/                # one YAML per run, versioned
│   │   ├── train_head_a.py
│   │   ├── losses.py               # OC-Softmax / AM-Softmax
│   │   ├── augment.py              # RawBoost + on-the-fly channel aug
│   │   └── continual.py            # replay buffer + EWC/RegO
│   ├── eval/                                                          [B15]
│   │   ├── run_eval.py
│   │   └── adversarial.py          # laundering / perturbation attacks
│   └── export/                                                        [B14]
│       ├── distill.py
│       ├── quantize.py
│       ├── to_onnx.py
│       └── parity_check.py
│
├── sdks/                                                              [B12]
│   ├── python/
│   ├── js/
│   └── edge/                       # ONNX Runtime Mobile / TFLite
│
├── deploy/                                                            [B17]
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   ├── helm/
│   ├── triton/models/
│   ├── telephony/
│   │   ├── asterisk/               # pjsip.conf, extensions.conf
│   │   └── freeswitch/
│   └── observability/
│       ├── prometheus.yml
│       └── grafana/dashboards/
│
├── scripts/
│   ├── bootstrap_repo.sh
│   ├── fetch_models.py             # pinned SHA + hash verification
│   ├── replay.py                   # dev entrypoint: wav -> live scores
│   └── check_status.py             # optional: validate PROJECT_STATUS.md
│
├── tests/
│   ├── unit/
│   ├── integration/                # one per capture adapter
│   ├── fixtures/                   # golden codec fixtures, short WAVs only
│   └── compliance/                 # no-raw-audio-at-rest, retention  [B16]
│
├── docs/
│   ├── adr/                        # architecture decision records
│   ├── benchmarks/REPORT.md        # generated by B15 — cite this, not papers
│   ├── compliance/DPIA.md
│   ├── ETHICS.md
│   ├── REFERENCES.md
│   ├── model_cards/
│   └── demo/
│
├── models/
│   └── registry.yaml               # pinned checkpoints, gitignored weights
│
└── .github/workflows/
    ├── ci.yml
    ├── schema-compat.yml
    └── eval.yml                    # regenerates the benchmark report
```

## Placement rules

1. **`packages/` has no I/O at import time.** Services do I/O; packages do computation. This is what makes the heads unit-testable and reusable in the edge SDK.
2. **Nothing under `services/` imports from another service.** Cross-service communication goes over the bus or gRPC, using the contracts in `SOURCE_OF_TRUTH.md §4`.
3. **No audio in git.** `tests/fixtures/` may contain short (<2 s) WAVs only. Everything else lives in `/data` or MinIO.
4. **No model weights in git.** `models/registry.yaml` pins them; `make fetch-models` retrieves them.
5. **Third-party research repos go in `third_party/`** (gitignored, cloned by a script) — never vendored into `packages/`.
6. **One training run = one YAML in `ml/training/configs/`,** committed. A run that isn't reproducible from a committed config didn't happen.
