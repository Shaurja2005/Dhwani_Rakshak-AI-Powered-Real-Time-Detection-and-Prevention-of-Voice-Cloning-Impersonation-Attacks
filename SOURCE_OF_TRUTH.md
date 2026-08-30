# SOURCE OF TRUTH — VoiceGuard

**Status:** LIVING DOCUMENT · **Version:** 0.1.0 · **Last updated:** (set on first commit)

> This file is the single authoritative reference for the project. If code and this file disagree, **this file is right and the code is a bug** — unless the change went through the ADR process in §12.
>
> **Every agent and every human must read §1–§5 before writing code, and must not change §3, §4, or §6 without an ADR.**

---

## 1. What we are building, in one paragraph

A real-time voice integrity verification framework that ingests live audio from telephony, VoIP, and collaboration platforms; analyzes it with multiple independent detection heads; fuses those signals with call metadata, transaction context, and transcript-level intent analysis; and emits a calibrated impersonation risk score with configurable, tenant-specific responses — while keeping raw biometric audio ephemeral and processing on-premise.

## 2. Non-negotiable invariants

These are architectural laws. Violating any one of them is a release blocker.

| # | Invariant | Why |
|---|---|---|
| **I1** | **Detection is advisory, never authoritative.** The system must never be the sole gate on a transaction. It raises friction and triggers out-of-band verification. | Any detector has a non-zero error floor under realistic conditions. |
| **I2** | **`ABSTAIN` is a first-class output.** When audio quality is inadequate or a head has no basis to score, it abstains. It never guesses. | Guessing on bad audio is the dominant source of false positives. |
| **I3** | **Channel augmentation is symmetric.** Any degradation applied to spoof audio is applied identically to bona fide audio. | Otherwise the model learns "was this codec'd", not "is this fake". Enforced by the CI test in `B3-T08`. |
| **I4** | **Watermark absence is not evidence of authenticity.** Presence is strong positive evidence of synthesis; absence changes the score by zero. | Attackers use generators that don't watermark. |
| **I5** | **Raw audio is ephemeral by default.** Feature-only logging. Raw waveforms persist only for flagged calls, only within an explicit retention window, only with a documented lawful basis. | DPDP Act 2023: voiceprints and processed voice are biometric data. |
| **I6** | **No inference or LLM call leaves the deployment boundary** unless the tenant explicitly opts in. Default is on-prem/edge. | RBI data-residency mandates for banking voice data. |
| **I7** | **No model trained on non-commercially-licensed data may be shipped to a commercial tenant.** Two lineages: `research` and `commercial`. | IndicSynth is CC BY-NC 4.0; SEA-Spoof is academic-only. |
| **I8** | **Every score is versioned and reproducible.** Every `HeadScore` and `SessionRisk` carries `model_version` and `calibration_version`. | Audit, forensics, and rollback. |
| **I9** | **No cloning of real executives, officials, or non-consenting individuals.** Ever, including for testing. | Ethics and law. See `docs/ETHICS.md`. |
| **I10** | **The audio path is never blocked by a slow component.** Any head or enrichment that exceeds its budget returns `ABSTAIN` and the pipeline continues. | Real-time guarantee. |

## 3. Canonical vocabulary

Use these exact terms in code, docs, UI, and commit messages. Do not invent synonyms.

| Term | Meaning |
|---|---|
| **session** | One monitored call, identified by `session_id` (UUIDv7). |
| **chunk** | A raw transport-level packet of audio as received. |
| **window** | A fixed-length analysis span (default 3.0 s) with a hop (default 1.0 s). The unit of scoring. |
| **head** | An independent detector producing one score per window. Heads are `A`–`F` (§6). |
| **bona fide** | Genuine human speech. Never "real" in code identifiers. |
| **spoof** | Synthetic or converted speech. Never "fake" in code identifiers. |
| **p_spoof** | Calibrated probability in `[0,1]` that a window/session is spoof. |
| **risk_score** | Integer `0–100`, the fused session-level risk including non-acoustic context. Distinct from `p_spoof`. |
| **state** | One of `LOW`, `ELEVATED`, `HIGH`, `ABSTAIN`. |
| **abstain** | Insufficient evidence to score. Not an error, not a low score. |
| **evidence bundle** | The immutable record justifying an alert. |
| **tenant** | A customer deployment with its own thresholds, keys, and policies. |
| **enrollment** | A stored voiceprint for a known speaker. |
| **shadow mode** | Scoring is live, alerting is suppressed. |

## 4. Data contracts (authoritative)

JSON Schema lives in `schemas/`; protobuf in `proto/voiceguard.proto`. These two must stay in sync — CI enforces it. Field names below are canonical.

### 4.1 `CallMetadata` — one per session, emitted at session start
```jsonc
{
  "session_id": "uuid7",
  "tenant_id": "string",
  "direction": "inbound|outbound",
  "started_at": "RFC3339",
  "caller_number": "E.164|null",
  "callee_number": "E.164|null",
  "claimed_identity_id": "string|null",   // who the caller says they are; keys the voiceprint lookup
  "channel": "pstn|voip|webrtc|mobile|file",
  "codec_hint": "g711u|g711a|g729|amrnb|opus|evs|pcm|unknown",
  "source_sample_rate": 8000,
  "trunk_id": "string|null",
  "source_asn": "string|null",
  "sip_headers": {},
  "language_hint": "hi|ta|bn|en-IN|...|null",
  "consent_basis": "legitimate_use|explicit_consent|none",
  "shadow_mode": true
}
```

### 4.2 `AudioChunk` — transport level
```jsonc
{ "session_id": "uuid7", "seq": 0, "ts_ms": 0, "sample_rate": 8000,
  "encoding": "pcm_s16le|mulaw|alaw", "payload": "bytes" }
```

### 4.3 `AnalysisWindow` — output of B2, input to every head
```jsonc
{
  "session_id": "uuid7", "window_id": 0,
  "start_ms": 0, "end_ms": 3000,
  "sample_rate": 16000,                 // always 16k after conditioning
  "samples_ref": "shm://... | inline",  // never persisted
  "voiced_ms": 2400, "snr_db": 18.2, "clipping_ratio": 0.001,
  "quality_ok": true,
  "quality_flags": ["hold_music"],      // empty when clean
  "original_sample_rate": 8000,
  "detected_codec": "opus|null"
}
```

### 4.4 `HeadScore` — every head emits exactly this
```jsonc
{
  "session_id": "uuid7", "window_id": 0,
  "head_id": "A|B|C|D|E|F",
  "raw_score": -1.87,                   // head-native scale, uncalibrated
  "p_spoof": 0.83,                      // calibrated; null when abstain=true
  "abstain": false,
  "abstain_reason": "no_enrollment|insufficient_speech|timeout|null",
  "confidence": 0.71,
  "latency_ms": 42,
  "model_version": "A@xlsr300m-nes2net-v0.3.1",
  "calibration_version": "cal-2026-02-11",
  "evidence": { "free": "form, head-specific, human-readable where possible" }
}
```

### 4.5 `FusedWindowScore`
```jsonc
{ "session_id": "uuid7", "window_id": 0, "p_spoof": 0.79, "state": "ELEVATED",
  "contributions": {"A": 0.51, "B": 0.12, "C": 0.09, "D": 0.07},
  "heads_abstained": ["E","F"], "fusion_version": "fuse-lr-v0.2" }
```

### 4.6 `ContextSignals` — output of B10
```jsonc
{ "session_id": "uuid7", "as_of_ms": 42000,
  "metadata_risk": 0.4, "transaction_risk": 0.9, "intent_risk": 0.8,
  "intent_labels": ["manufactured_urgency","secrecy_demand","callback_resistance"],
  "transcript_snippets": [{"start_ms":12000,"text":"redacted text","label":"secrecy_demand"}],
  "language_detected": "hi", "code_switched": true }
```

### 4.7 `SessionRisk` — the headline output
```jsonc
{
  "session_id": "uuid7", "updated_at": "RFC3339",
  "risk_score": 78, "state": "HIGH",
  "p_spoof_session_max": 0.91,     // any-segment trigger
  "p_spoof_session_mean": 0.44,
  "drivers": [{"factor":"speaker_mismatch","weight":0.31,"detail":"cos=0.22 vs enrolled"}],
  "timeline": [{"window_id":0,"p_spoof":0.12,"state":"LOW"}],
  "model_versions": {"A":"...","D":"...","fusion":"..."},
  "abstain_ratio": 0.15
}
```

### 4.8 `PolicyDecision`
```jsonc
{ "session_id":"uuid7","decided_at":"RFC3339","threshold_profile":"bfsi-wealth-v2",
  "actions":["agent_banner","force_callback","hold_transaction"],
  "rationale":"risk_score 78 >= 70 and transaction_sensitivity=high",
  "evidence_bundle_id":"sha256:...","shadow_mode":false,"suppressed":false }
```

### 4.9 Head plugin interface (Python)
```python
class DetectionHead(Protocol):
    head_id: str            # "A".."F"
    model_version: str
    budget_ms: int
    def warmup(self) -> None: ...
    def score(self, window: AnalysisWindow, ctx: SessionContext) -> HeadScore: ...
```
Heads must be pure with respect to session state, must not raise on bad input (return `abstain=True`), and must respect `budget_ms`.

## 5. Repository ownership map

One directory, one owning block. Cross-directory edits require a note in `PROJECT_STATUS.md`.

| Path | Owning block |
|---|---|
| `proto/`, `schemas/`, `packages/vg_core/` | B0 |
| `services/ingest/` | B1 |
| `services/conditioner/`, `packages/vg_audio/` | B2 |
| `ml/data/` | B3 |
| `ml/training/`, `packages/vg_models/heads/head_a_*` | B4 |
| `packages/vg_models/heads/head_b_*` | B5 |
| `packages/vg_models/heads/head_c_*` | B6 |
| `packages/vg_models/heads/head_d_*`, `services/enrollment/` | B7 |
| `packages/vg_models/heads/head_e_*`, `head_f_*` | B8 |
| `services/fusion/` | B9 |
| `services/context/` | B10 |
| `services/policy/` | B11 |
| `services/api_gateway/`, `sdks/` | B12 |
| `services/ui/` | B13 |
| `ml/export/`, `deploy/triton/` | B14 |
| `ml/eval/`, `packages/vg_eval/` | B15 |
| `services/privacy/`, `docs/compliance/` | B16 |
| `deploy/` | B17 |
| `docs/demo/` | B18 |

## 6. Detection head registry

| ID | Name | Primary tech | Budget | Abstains when |
|---|---|---|---|---|
| **A** | SSL anti-spoof | XLS-R-300M or WavLM-large (layers 5–9, weighted sum) + Nes2Net; AASIST secondary | 120 ms GPU / 400 ms CPU | window quality fails, timeout |
| **B** | DSP artifact + scene consistency | CQT/LFCC/MGD/LTAS/bicoherence + GBDT; RIR discordance | 15 ms | insufficient voiced speech |
| **C** | Prosody / behavioral | F0 dynamics, jitter/shimmer, breath groups, pause distribution, disfluency | 25 ms | < 2.0 s voiced speech |
| **D** | Speaker verification | ECAPA-TDNN / TitaNet-L + AS-norm vs enrollment | 30 ms | **no enrollment for `claimed_identity_id`** |
| **E** | Active liveness | Nonce / code-switched challenge + latency and ASR match | on demand | no challenge issued |
| **F** | Watermark probe | AudioSeal + commercial detectors | 5 ms | never abstains; absence ⇒ neutral (I4) |

## 7. Model registry (pin every version)

Recorded in `models/registry.yaml` with a commit SHA and sha256 per artifact.

| Role | Default choice | Alternatives |
|---|---|---|
| SSL front-end | `facebook/wav2vec2-xls-r-300m` | `microsoft/wavlm-large`, `xls-r-1b` |
| Anti-spoof back-end | Nes2Net | AASIST, SLS classifier, RawNet2 |
| Reference repo | `TakHemlata/SSL_Anti-spoofing` | `clovaai/aasist`, DeepFense |
| Speaker embedding | SpeechBrain `spkrec-ecapa-voxceleb` | WeSpeaker, NVIDIA TitaNet-L, 3D-Speaker |
| VAD | Silero VAD | pyannote 3.x |
| ASR (Indic) | AI4Bharat IndicConformer | IndicWhisper, Whisper large-v3 |
| Intent LLM | local Qwen2.5-7B-Instruct or Llama-3.1-8B (vLLM/Ollama) | any local model; **no hosted API** (I6) |
| Watermark | Meta AudioSeal | commercial detectors |
| Noise suppression (simulator) | RNNoise | DeepFilterNet |
| Serving | ONNX Runtime + NVIDIA Triton | TFLite / ORT-Mobile for edge |
| Eval harness | AUDDT | DeepFense |

## 8. Dataset registry and license policy

`ml/data/registry.yaml` is authoritative. Every row carries `commercial_use: true|false`.

| Dataset | Role | Commercial use |
|---|---|---|
| ASVspoof 5 | primary train/dev | check EULA |
| ASVspoof 2019 LA, 2021 LA+DF | comparability only | check EULA |
| In-the-Wild | **honesty check**, eval only | eval |
| SpoofCeleb | train + eval, closest to phone acoustics | check |
| MLAAD (+ M-AILABS) | multilingual train | check |
| RTCFake | VoIP/WebRTC degradation — directly on-point | check |
| PartialSpoof / LlamaPartialSpoof | splicing attacks | check |
| CodecFake, DFADD, SpeechFake | 2024+ generator families | check |
| **IndicSynth** | Indic train | **NO — CC BY-NC 4.0** |
| **SEA-Spoof** | Indic/SEA train+eval | **NO — academic only** |
| Indic-CodecFake | Indic NAC benchmark | check |
| IndicVoices, Kathbath/IndicSUPERB, Shrutilipi, Svarah, Common Voice Indic | bona fide source | per-corpus |
| MUSAN, RIRS_NOISES, DNS noise | augmentation | yes |
| **VG-Indic-Telephony (ours)** | the differentiator | yes, if built from permissive sources |

**Rule:** a training run declares `allow_noncommercial`. When false, the loader hard-fails on any restricted row. Model cards state the lineage.

## 9. Operating points and thresholds

- Thresholds are **per-tenant configuration**, never constants in code.
- Chosen from a **cost model at fixed FPR (0.1%, 1%)** — not from EER. At 50,000 calls/day, a 2% FPR is 1,000 wrongly-flagged customers.
- Default profile `bfsi-default-v1`:

| Band | `risk_score` | Default action |
|---|---|---|
| LOW | 0–39 | log only |
| ELEVATED | 40–69 | agent banner + recommend call-back |
| HIGH | 70–100 | force step-up verification, hold transaction, escalate |
| ABSTAIN | — | show "insufficient audio quality", never imply safety |

- New tenants start in **shadow mode** for a minimum calibration period before alerts are enabled.

## 10. Latency budget

| Stage | p95 target |
|---|---|
| Capture → conditioner | 60 ms |
| Head A (GPU, batched) | 120 ms |
| Heads B/C/D/F (CPU, parallel) | 30 ms |
| Fusion + risk | 10 ms |
| Policy + emit | 20 ms |
| **End-to-end per window** | **< 300 ms GPU / < 700 ms CPU-only** |
| **Time to first score** | 3.5 s of *voiced* speech |
| Score update cadence | every 1.0 s |

Context/intent (B10) runs asynchronously and is **never** in the audio path (I10).

## 11. Environment, ports, and naming

| Service | Port | Env prefix |
|---|---|---|
| api_gateway (REST) | 8080 | `VG_GATEWAY_` |
| api_gateway (gRPC) | 50050 | |
| ingest (WS) | 8081 | `VG_INGEST_` |
| conditioner | 50051 | `VG_COND_` |
| inference / heads | 50052 | `VG_INFER_` |
| fusion | 50053 | `VG_FUSION_` |
| context | 50054 | `VG_CONTEXT_` |
| policy | 8085 | `VG_POLICY_` |
| enrollment | 8086 | `VG_ENROLL_` |
| ui | 3000 | `VG_UI_` |
| postgres/timescale | 5432 | `VG_DB_` |
| redis | 6379 | `VG_REDIS_` |
| minio | 9000/9001 | `VG_S3_` |
| triton | 8000/8001/8002 | `VG_TRITON_` |
| prometheus / grafana | 9090 / 3001 | |

Naming: Python `snake_case`, packages `vg_*`, services `kebab-case` directories, env vars `VG_<SERVICE>_<KEY>`, model versions `<head>@<frontend>-<backend>-v<semver>`.

## 12. Decision log (ADRs)

Any change to §2, §3, §4, or §6 requires a new ADR file in `docs/adr/NNNN-title.md` with: context, decision, alternatives considered, consequences, and the blocks affected. Append the row here.

| ADR | Date | Decision | Status |
|---|---|---|---|
| 0001 | — | Nes2Net as primary back-end over AASIST for the real-time head | Accepted |
| 0002 | — | Window 3.0 s / hop 1.0 s as the default analysis geometry | Accepted |
| 0003 | — | Local LLM only for intent analysis (data residency) | Accepted |
| 0004 | — | Two model lineages (`research` / `commercial`) due to dataset licensing | Accepted |
| 0005 | — | `ABSTAIN` as a first-class pipeline state | Accepted |

## 13. Glossary of external references

Keep a one-line note per external artifact you rely on, with its URL, in `docs/REFERENCES.md`. Every performance claim in the submission must cite a row in `docs/benchmarks/REPORT.md`, which is generated by B15 — never a number remembered from a paper.
