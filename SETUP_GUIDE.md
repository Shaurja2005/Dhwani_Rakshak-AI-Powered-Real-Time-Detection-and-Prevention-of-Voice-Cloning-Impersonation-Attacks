# SETUP GUIDE — the parts a human has to do

This covers everything that cannot be automated away: accounts, licenses, downloads, GPU environments, telephony labs, and local services. Work through it in order. **§3 (datasets) has multi-day lead times — start it on day 1 even if you do nothing else.**

Time estimates assume one competent person and decent bandwidth.

---

## 1. Base development environment (~2 hours)

**Pin these versions.** Speech tooling breaks badly on version drift.

| Component | Version | Notes |
|---|---|---|
| Python | **3.11** | Not 3.12 — several speech/fairseq-derived packages still break |
| Node | 20 LTS | UI + JS SDK |
| Docker + Compose v2 | current | |
| CUDA | 12.1 | match your torch build |
| ffmpeg | 6.x with `--enable-libopus`, G.711/G.722/AMR | codec simulation depends on this |
| sox | 14.4+ | |
| protoc + grpcio-tools | current | |

```bash
# recommended: uv for speed, or conda if you prefer
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements/dev.txt

# verify ffmpeg has the codecs you need
ffmpeg -codecs | grep -Ei 'pcm_mulaw|opus|amr|g729'
```

If `pcm_mulaw` or `libopus` is missing, install `ffmpeg` from a full build (not the minimal distro package) — the channel simulator in B3 will silently produce wrong data otherwise.

**Local service stack** (`make dev-up`) brings up: Postgres+TimescaleDB, Redis, MinIO, Prometheus, Grafana. Ports are listed in `SOURCE_OF_TRUTH.md §11`. Nothing else should be installed globally.

---

## 2. GPU environment (~2–4 hours, plus procurement time)

### What you actually need

| Job | Minimum | Comfortable |
|---|---|---|
| Fine-tuning a 300M SSL front-end + small back-end | 1× 24 GB (4090/A5000), or Colab Pro / Kaggle T4 with the front-end frozen | 1× A100 40 GB |
| **Corpus generation (bigger consumer than training)** | 1× 24 GB, a weekend | 2× 4090 or 1× A100 |
| Real-time inference target | T4 / L4 | L4 |
| Local intent LLM (7B) | 16 GB VRAM, or quantized on CPU for demo | 24 GB |

For the hackathon lane, a single Colab Pro session is enough if you freeze the SSL front-end and train only the back-end. Budget your Colab time — corpus generation will eat it.

### Setup
```bash
nvidia-smi                      # confirm driver
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Experiment tracking
Use **MLflow local** or **W&B in offline mode**. Do not send training telemetry to a hosted service if you intend to make the data-residency claim in your writeup — reviewers will notice the inconsistency.

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns.db
```

---

## 3. Datasets — START DAY 1 (approvals take days; downloads take longer)

### 3.1 Access requests to file immediately

| Corpus | How to get it | Lead time |
|---|---|---|
| **ASVspoof 5** | Register / accept the EULA on the ASVspoof site; resources are freely available but gated | hours–days |
| ASVspoof 2019 LA, 2021 LA+DF | Edinburgh DataShare / Zenodo, license acceptance | hours |
| **In-the-Wild** | Zenodo, direct download | hours |
| SpoofCeleb | Project page / HF | hours–days |
| MLAAD + M-AILABS | Direct + HF | hours |
| **RTCFake** | Project release page | days |
| PartialSpoof, CodecFake, DFADD, SpeechFake | Zenodo / GitHub releases | hours–days |
| **IndicSynth** | `huggingface.co/datasets/vdivyasharma/IndicSynth` — accept CC BY-NC 4.0 | hours |
| **SEA-Spoof** | Project request form — academic use only | **days** |
| Indic-CodecFake | Project page | days |
| AI4Bharat IndicVoices / Kathbath / IndicSUPERB / Shrutilipi / Svarah | ai4bharat.org, some via HF | hours |
| Common Voice Indic subsets | Mozilla, direct | hours |
| MUSAN, RIRS_NOISES | OpenSLR, direct | hours |

```bash
huggingface-cli login          # needed for the gated HF datasets
```

### 3.2 Storage

Budget **1.5–3 TB**. IndicSynth alone is 4,000+ hours; ASVspoof 5 and SpoofCeleb are each large. Put raw corpora on a separate volume from your working manifests, and never copy audio into the repo.

```
/data/raw/<corpus>/        # untouched downloads, read-only after verification
/data/derived/<job_id>/    # cloned + channel-degraded audio
/data/manifests/*.jsonl    # the only thing training reads directly
```

### 3.3 The license trap — read this before training anything

**IndicSynth is CC BY-NC 4.0 and SEA-Spoof is academic-only.** They are the two most valuable Indic resources and neither can be used to train a model you intend to ship commercially. This is not a footnote; it determines your data strategy.

Consequences you must design around now, not later:
1. Maintain **two model lineages**: `research` (everything, best numbers, for the paper and the hackathon) and `commercial` (permissive sources plus your own synthesized corpus).
2. The `commercial` lineage is *why* B3's own-corpus generation exists. It is not optional polish.
3. Every model card states its lineage and the corpora it saw.

### 3.4 Building your own corpus — the setup that surprises people

The clone zoo is the single most annoying setup task in the project. **Do not try to install XTTS-v2, F5-TTS, CosyVoice, and RVC into one Python environment.** Their dependency pins conflict irreconcilably. One Docker image per generator family:

```
ml/data/generators/
├── xtts_v2/Dockerfile
├── openvoice_v2/Dockerfile
├── f5_tts/Dockerfile
├── cosyvoice2/Dockerfile
├── fish_speech/Dockerfile
├── styletts2/Dockerfile
├── melotts/Dockerfile
├── indic_parler/Dockerfile
├── bark/Dockerfile
├── freevc/Dockerfile
├── rvc/Dockerfile
├── seedvc/Dockerfile
└── knnvc/Dockerfile
```

Each image exposes the same tiny CLI so the batch job is generator-agnostic:
```
clone --ref-audio <path> --text <utf8> --lang <code> --out <path> --seed <int>
```

Budget a full day for this. It is worth it: **diversity of generator families matters far more than volume from any single model**, and a uniform interface is what lets you run leave-one-generator-out evaluation later.

Also required for the channel simulator: RNNoise or DeepFilterNet, an Opus encoder at multiple bitrates, an AMR-NB codec, and a packet-loss/PLC simulator. Verify each transform round-trips audibly before generating 500 hours with a broken one.

---

## 4. Model checkpoints (~1 hour of work, several hours of download)

Create `models/registry.yaml` pinning **commit SHA and sha256 for every artifact**. HF main branches move; an unpinned front-end silently changes your results between runs.

```yaml
- role: ssl_frontend
  id: facebook/wav2vec2-xls-r-300m
  revision: <commit-sha>
  sha256: <hash>
  size_gb: 1.2
- role: ssl_frontend_alt
  id: microsoft/wavlm-large
  revision: <commit-sha>
- role: speaker_embedding
  id: speechbrain/spkrec-ecapa-voxceleb
  revision: <commit-sha>
- role: vad
  id: snakers4/silero-vad
  revision: <commit-sha>
- role: asr_indic
  id: ai4bharat/indic-conformer-600m-multilingual
  revision: <commit-sha>
- role: asr_fallback
  id: openai/whisper-large-v3
  revision: <commit-sha>
- role: watermark
  id: facebook/audioseal
  revision: <commit-sha>
```

Fetch with `make fetch-models` (wraps `scripts/fetch_models.py`, verifies hashes, caches to MinIO so teammates don't each re-download 40 GB).

Repos to clone into `third_party/` (do not vendor into your package tree):
- `TakHemlata/SSL_Anti-spoofing` — wav2vec2 + AASIST + RawBoost, the canonical reproduction starting point
- `clovaai/aasist`
- `Liu-Tianchi/Nes2Net`
- AUDDT (evaluation harness) or DeepFense (modular detection framework)

**Total download: roughly 40–60 GB of model weights.**

---

## 5. Telephony lab (~4–8 hours, optional for most of the team)

**Most developers should never need this.** The WAV replay adapter (B1-T01) and the WebSocket adapter (B1-T02) let 80% of the team work without touching a PBX. Set the lab up on one machine, owned by the B1 engineer.

### Option A — Asterisk in Docker (best for the "enterprise PBX" story)
```bash
docker run -d --name asterisk --network host \
  -v $PWD/deploy/telephony/asterisk:/etc/asterisk andrius/asterisk:20
```
Configure two PJSIP endpoints, register two softphones (Linphone on a laptop, Zoiper on a phone), and route the dialplan through an `AudioSocket()` application pointed at your ingest service. Test with a real call between two devices — this is what you'll show live.

### Option B — Twilio Media Streams (fastest to a real PSTN call)
1. Trial account, buy a number (~₹100/month equivalent).
2. `ngrok http 8081` to expose your local ingest.
3. TwiML: `<Start><Stream url="wss://<ngrok>/twilio"/></Start>`.
4. **Twilio streams 8 kHz µ-law base64 frames.** Test your resampler against exactly this; it is the most common integration bug.

### Option C — pure browser (best fallback for a demo with no network)
`getUserMedia` → AudioWorklet → WebSocket. No accounts, no telephony, works offline. Build this even if you build the others.

### Codec fixtures
Record one clean utterance and produce a fixture set through every codec path you claim to support. These become your integration-test golden files:
```bash
ffmpeg -i clean.wav -ar 8000 -acodec pcm_mulaw -f wav g711u.wav
ffmpeg -i clean.wav -ar 8000 -acodec libopus -b:a 12k opus12k.ogg
ffmpeg -i clean.wav -ar 8000 -acodec amr_nb -b:a 12.2k amr.amr
```

---

## 6. ASR and the local intent LLM (~3 hours)

### ASR
- **IndicConformer** (AI4Bharat) for Indic languages — this is the right default for your problem statement.
- **Whisper large-v3** (~3 GB) as fallback and for English. `faster-whisper` (CTranslate2) is substantially faster for streaming.
- Streaming requires chunked decoding with a rolling hypothesis buffer. Do not naively re-decode the whole call every second.

### Intent LLM — must be local
```bash
# fast path
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct

# throughput path
uv pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

**Do not call a hosted LLM API for this.** RBI data-residency rules mean banking voice data must remain in India, which rules out offshore inference endpoints — and your whole privacy story collapses if the transcript leaves the deployment boundary. This is invariant I6.

Prepare a seed set of ~200 labeled scenario transcripts covering: manufactured urgency, secrecy demands, authority pressure, unusual channel, call-back resistance, unusual beneficiary, plus a matched set of benign calls. Write them yourself; they don't need to be real. This is a half-day of work and it is what makes the intent head credible.

---

## 7. Serving stack (~3 hours, do this in Phase 3, not before)

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $PWD/deploy/triton/models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
```

Sequence that actually works: PyTorch → ONNX export → **verify numerical parity within tolerance** → quantize → verify accuracy delta → Triton config with dynamic batching → load test. Skipping the parity check is how teams ship a subtly broken exported model and don't find out until the demo.

---

## 8. First-day checklist

Print this. Tick it.

- [ ] Repo bootstrapped, `make dev-up && make test` green
- [ ] All dataset access requests **submitted** (not received — submitted)
- [ ] HF login done, `models/registry.yaml` drafted
- [ ] GPU confirmed working, torch sees CUDA
- [ ] `ffmpeg` verified to have µ-law, Opus, AMR
- [ ] WAV replay adapter streaming into a stub head, scores printing
- [ ] `SOURCE_OF_TRUTH.md` read by everyone; §4 contracts agreed
- [ ] `PROJECT_STATUS.md` owners assigned for B0–B3 at minimum
- [ ] One person owns the telephony lab; nobody else is blocked on it
- [ ] Ethics/consent register created before any voice is cloned
