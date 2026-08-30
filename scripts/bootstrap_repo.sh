#!/usr/bin/env bash
# bootstrap_repo.sh — create the VoiceGuard monorepo skeleton.
# Usage:  bash bootstrap_repo.sh [target_dir]        (default: ./voiceguard)
# Idempotent: safe to re-run; never overwrites an existing file.

set -euo pipefail
ROOT="${1:-voiceguard}"

say() { printf '  %s\n' "$1"; }
mk()  { mkdir -p "$ROOT/$1"; }
# touch a file with optional content, only if it does not exist
tf()  {
  local path="$ROOT/$1"; shift
  [ -e "$path" ] && return 0
  mkdir -p "$(dirname "$path")"
  if [ "$#" -gt 0 ]; then printf '%s\n' "$@" > "$path"; else : > "$path"; fi
}
pkg() { tf "$1/__init__.py"; }

echo "Bootstrapping VoiceGuard repo at: $ROOT"

# ---------------------------------------------------------------- top level
mk .
tf README.md \
  "# VoiceGuard" "" \
  "Real-time detection and prevention of voice cloning impersonation attacks." "" \
  "**Read \`SOURCE_OF_TRUTH.md\` before writing any code.** Task board: \`PROJECT_STATUS.md\`. Working rules: \`AGENTS.md\`." "" \
  "Quickstart:" "" '```bash' "make dev-up" "make fetch-models" "python scripts/replay.py --wav tests/fixtures/sample.wav" '```'

tf .gitignore \
  ".venv/" "__pycache__/" "*.pyc" ".env" "" \
  "# never commit data or weights" \
  "/data/" "third_party/" "models/*.bin" "models/*.pt" "models/*.onnx" "models/*.safetensors" \
  "*.wav" "*.flac" "*.mp3" "*.ogg" "*.amr" \
  "!tests/fixtures/*.wav" "" \
  "packages/vg_core/generated/" "mlruns/" "node_modules/" ".next/" "dist/"

tf .env.example \
  "VG_ENV=dev" \
  "VG_TENANT_DEFAULT=demo" \
  "VG_DB_URL=postgresql://vg:vg@localhost:5432/vg" \
  "VG_REDIS_URL=redis://localhost:6379/0" \
  "VG_S3_ENDPOINT=http://localhost:9000" \
  "VG_S3_ACCESS_KEY=minioadmin" \
  "VG_S3_SECRET_KEY=minioadmin" \
  "VG_INFER_DEVICE=cuda" \
  "VG_CONTEXT_LLM_BASE_URL=http://localhost:11434" \
  "VG_SHADOW_MODE=true" \
  "HF_TOKEN="

tf Makefile \
  ".PHONY: dev-up dev-down test lint proto fetch-models replay eval corpus-report prod-up" "" \
  "dev-up:" "	docker compose -f deploy/docker-compose.dev.yml up -d" "" \
  "dev-down:" "	docker compose -f deploy/docker-compose.dev.yml down" "" \
  "test:" "	pytest -q" "" \
  "lint:" "	ruff check . && black --check . && mypy packages services" "" \
  "proto:" "	python -m grpc_tools.protoc -Iproto --python_out=packages/vg_core/generated \\" \
  "		--grpc_python_out=packages/vg_core/generated proto/voiceguard.proto" "" \
  "fetch-models:" "	python scripts/fetch_models.py --registry models/registry.yaml" "" \
  "replay:" "	python scripts/replay.py --wav \$(WAV)" "" \
  "eval:" "	python ml/eval/run_eval.py --model \$(MODEL) --out docs/benchmarks/REPORT.md" "" \
  "corpus-report:" "	python ml/data/manifest.py --report" "" \
  "prod-up:" "	docker compose -f deploy/docker-compose.prod.yml up -d"

tf pyproject.toml \
  "[project]" 'name = "voiceguard"' 'version = "0.1.0"' 'requires-python = ">=3.11,<3.12"' "" \
  "[tool.ruff]" "line-length = 100" "" \
  "[tool.mypy]" "python_version = \"3.11\"" "ignore_missing_imports = true" "" \
  "[tool.pytest.ini_options]" 'testpaths = ["tests"]'

for doc in SOURCE_OF_TRUTH.md PROJECT_STATUS.md AGENTS.md IMPLEMENTATION_PLAN.md SETUP_GUIDE.md REPO_LAYOUT.md; do
  tf "$doc" "<!-- Replace this placeholder with the delivered $doc -->"
done
say "top-level files"

# ------------------------------------------------------------ proto/schemas
tf proto/voiceguard.proto \
  'syntax = "proto3";' "package voiceguard.v1;" "" \
  "// Contracts are authoritative in SOURCE_OF_TRUTH.md section 4." \
  "// Keep this file and schemas/*.json in sync; CI enforces it." "" \
  "service VoiceIntegrity {" \
  "  rpc AnalyzeStream(stream StreamRequest) returns (stream RiskEvent);" \
  "  rpc AnalyzeFile(AnalyzeFileRequest) returns (AnalyzeFileResponse);" \
  "  rpc Enroll(stream EnrollChunk) returns (EnrollResponse);" \
  "}" "" \
  "// TODO(B0-T02): define messages per SOURCE_OF_TRUTH.md section 4."

for s in call_metadata audio_chunk analysis_window head_score fused_window_score \
         context_signals session_risk policy_decision evidence_bundle; do
  tf "schemas/${s}.schema.json" \
    '{' '  "$schema": "https://json-schema.org/draft/2020-12/schema",' \
    "  \"title\": \"${s}\"," '  "type": "object",' '  "properties": {},' \
    '  "required": []' '}'
done
say "proto + schemas"

# ------------------------------------------------------------- packages
pkg packages/vg_core
for f in config logging models bus head_api stub_head versioning; do
  tf "packages/vg_core/${f}.py" "\"\"\"vg_core.${f} — TODO(B0). See SOURCE_OF_TRUTH.md section 4.\"\"\""
done
mk packages/vg_core/generated

pkg packages/vg_audio
for f in resample vad windowing quality features codecs; do
  tf "packages/vg_audio/${f}.py" "\"\"\"vg_audio.${f} — TODO(B2).\"\"\""
done

pkg packages/vg_models
pkg packages/vg_models/heads
for h in head_a_ssl head_b_dsp head_c_prosody head_d_speaker head_e_liveness head_f_watermark; do
  pkg "packages/vg_models/heads/${h}"
  tf "packages/vg_models/heads/${h}/head.py" \
    "\"\"\"${h} — implements vg_core.head_api.DetectionHead.\"\"\"" "" \
    "# Contract: return HeadScore; never raise; abstain instead of guessing (invariant I2)."
done
tf packages/vg_models/calibration.py "\"\"\"Per-head calibration — TODO(B9-T01).\"\"\""
tf packages/vg_models/registry.py "\"\"\"Pinned model loading — TODO(B0/B4).\"\"\""

pkg packages/vg_eval
for f in metrics protocols fairness report; do
  tf "packages/vg_eval/${f}.py" "\"\"\"vg_eval.${f} — TODO(B15).\"\"\""
done
say "packages/"

# ------------------------------------------------------------- services
for s in ingest conditioner inference fusion context policy enrollment privacy api_gateway; do
  pkg "services/${s}"
  tf "services/${s}/main.py" "\"\"\"${s} service entrypoint.\"\"\"" "" \
     "def main() -> None:" "    raise NotImplementedError" "" \
     'if __name__ == "__main__":' "    main()"
  tf "services/${s}/Dockerfile" "FROM python:3.11-slim" "WORKDIR /app" "# TODO(B17-T01)"
done

pkg services/ingest/adapters
for a in replay_wav websocket_pcm asterisk_audiosocket freeswitch_ws twilio_stream siprec webrtc; do
  tf "services/ingest/adapters/${a}.py" "\"\"\"Capture adapter: ${a} — TODO(B1).\"\"\""
done

mk services/ui/src/pages/live
mk services/ui/src/pages/forensics
mk services/ui/src/pages/admin
mk services/ui/src/components
tf services/ui/package.json '{ "name": "voiceguard-ui", "private": true, "version": "0.1.0" }'
tf services/ui/src/components/RiskGauge.tsx "// TODO(B13-T01): risk gauge bound to the SessionRisk stream."
say "services/"

# ------------------------------------------------------------------- ml/
pkg ml
pkg ml/data
tf ml/data/registry.yaml \
  "# Dataset registry. Every entry MUST declare commercial_use (invariant I7)." \
  "datasets:" \
  "  - name: asvspoof5" "    role: train" "    license: TODO" "    commercial_use: false" "    url: TODO" \
  "  - name: in_the_wild" "    role: eval_only" "    license: TODO" "    commercial_use: false" "    url: TODO" \
  "  - name: indicsynth" "    role: train_research" "    license: CC-BY-NC-4.0" "    commercial_use: false" "    url: TODO"
mk ml/data/download
for f in manifest splits license_gate clone_job; do
  tf "ml/data/${f}.py" "\"\"\"ml.data.${f} — TODO(B3).\"\"\""
done
pkg ml/data/channel
for f in codecs packet_loss rir noise webrtc_chain; do
  tf "ml/data/channel/${f}.py" "\"\"\"Channel destruction: ${f} — TODO(B3-T07).\"\"\""
done
tf ml/data/channel/test_symmetry.py \
  "\"\"\"CI gate for invariant I3: augmentation must be symmetric.\"\"\"" "" \
  "import pytest" "" \
  "@pytest.mark.skip(reason=\"TODO(B3-T08)\")" \
  "def test_channel_features_cannot_separate_classes() -> None:" \
  "    # Train a tiny classifier on channel features only; assert ~chance accuracy." \
  "    raise NotImplementedError"
for g in xtts_v2 openvoice_v2 f5_tts cosyvoice2 fish_speech styletts2 melotts indic_parler bark freevc rvc seedvc knnvc; do
  tf "ml/data/generators/${g}/Dockerfile" \
    "# Generator: ${g}. One image per family — their deps conflict." \
    "# Must expose: clone --ref-audio <p> --text <s> --lang <c> --out <p> --seed <i>" \
    "FROM python:3.11-slim"
done

pkg ml/training
mk ml/training/configs
for f in train_head_a losses augment continual; do
  tf "ml/training/${f}.py" "\"\"\"ml.training.${f} — TODO(B4).\"\"\""
done
tf ml/training/configs/head_a_stage1.yaml \
  "run_name: head_a_stage1" "lineage: research        # research | commercial" \
  "allow_noncommercial: true   # license gate (invariant I7)" \
  "frontend: facebook/wav2vec2-xls-r-300m" "frontend_layers: [5, 6, 7, 8, 9]" \
  "backend: nes2net" "loss: oc_softmax" "seed: 1337"

pkg ml/eval
tf ml/eval/run_eval.py "\"\"\"Evaluation entrypoint — TODO(B15-T01).\"\"\""
tf ml/eval/adversarial.py "\"\"\"Laundering / perturbation attacks — TODO(B15-T08).\"\"\""
pkg ml/export
for f in distill quantize to_onnx parity_check; do
  tf "ml/export/${f}.py" "\"\"\"ml.export.${f} — TODO(B14).\"\"\""
done
say "ml/"

# ------------------------------------------------------------ sdks + deploy
pkg sdks/python
tf sdks/js/package.json '{ "name": "@voiceguard/sdk", "version": "0.1.0" }'
mk sdks/edge

tf deploy/docker-compose.dev.yml \
  "# Dev stack: postgres+timescale, redis, minio, prometheus, grafana." \
  "# Ports are fixed in SOURCE_OF_TRUTH.md section 11. TODO(B0-T06)." \
  "services: {}"
tf deploy/docker-compose.prod.yml "# Single-node on-prem deployment. TODO(B17-T02)." "services: {}"
mk deploy/helm
mk deploy/triton/models
tf deploy/telephony/asterisk/extensions.conf "; TODO(B1-T03): route the dialplan through AudioSocket()."
tf deploy/telephony/asterisk/pjsip.conf "; TODO(B1-T03): two test endpoints for softphones."
mk deploy/telephony/freeswitch
tf deploy/observability/prometheus.yml "global:" "  scrape_interval: 15s" "scrape_configs: []"
mk deploy/observability/grafana/dashboards
say "sdks/ + deploy/"

# ---------------------------------------------------------------- scripts
tf scripts/fetch_models.py \
  "\"\"\"Download + verify pinned checkpoints from models/registry.yaml.\"\"\"" "" \
  "# Pin revision SHA and sha256 for every artifact: unpinned weights silently" \
  "# change results between runs." 
tf scripts/replay.py \
  "\"\"\"Dev entrypoint: replay a WAV through the full pipeline and print risk events.\"\"\"" "" \
  "# This is how the team works without a PBX. Keep it working. TODO(B1-T01)."
tf scripts/check_status.py "\"\"\"Validate PROJECT_STATUS.md: IDs unique, DONE rows carry artifacts.\"\"\""
cp -n "$0" "$ROOT/scripts/bootstrap_repo.sh" 2>/dev/null || true

# ------------------------------------------------------------------ tests
mk tests/unit
mk tests/integration
mk tests/fixtures
mk tests/compliance
tf tests/compliance/test_no_raw_audio_at_rest.py \
  "\"\"\"Invariant I5: no raw PCM reaches disk on the default path.\"\"\"" "" \
  "import pytest" "" \
  "@pytest.mark.skip(reason=\"TODO(B16-T01)\")" \
  "def test_no_raw_audio_written() -> None:" \
  "    raise NotImplementedError"
tf tests/README.md "Fixtures: short (<2s) WAVs only. Golden codec fixtures live here."

# ------------------------------------------------------------------- docs
mk docs/adr
mk docs/model_cards
mk docs/demo
tf docs/adr/0001-nes2net-primary-backend.md \
  "# ADR 0001 — Nes2Net as the primary anti-spoof back-end" "" \
  "## Context" "" "## Decision" "" "## Alternatives considered" "" "## Consequences" "" "## Blocks affected"
tf docs/benchmarks/REPORT.md \
  "# Benchmark report" "" \
  "Generated by \`make eval\`. **Every performance claim anywhere in this project must cite a row in this file.**" "" \
  "_(empty — no evaluation has been run yet)_"
tf docs/compliance/DPIA.md "# Data Protection Impact Assessment" "" "TODO(B16-T08)."
tf docs/ETHICS.md \
  "# Ethics and consent" "" \
  "- Clone only consented speakers or public-domain corpora." \
  "- Never clone a real executive, official, or non-consenting individual — including for testing." \
  "- The generated corpus stays internal and is documented as defensive research." "" \
  "## Consent register" "" "| Name | Date | Scope of consent | Withdrawal contact |" "|---|---|---|---|"
tf docs/REFERENCES.md "# External references" "" "One line per external artifact we rely on, with its URL."

mk models
tf models/registry.yaml "# Pinned checkpoints. revision = commit SHA, plus sha256 per artifact." "models: []"

mk .github/workflows
tf .github/workflows/ci.yml "name: ci" "on: [push, pull_request]" "jobs: {}"
tf .github/workflows/schema-compat.yml \
  "name: schema-compat" "# Fails a PR if schemas/ changed without a version bump. TODO(B0-T07)." \
  "on: [pull_request]" "jobs: {}"
tf .github/workflows/eval.yml "name: eval" "# Regenerates docs/benchmarks/REPORT.md. TODO(B15-T10)." "on: workflow_dispatch" "jobs: {}"
say "scripts/ tests/ docs/ .github/"

echo
echo "Done. Next steps:"
echo "  1. Copy the delivered SOURCE_OF_TRUTH.md / PROJECT_STATUS.md / AGENTS.md /"
echo "     IMPLEMENTATION_PLAN.md / SETUP_GUIDE.md / REPO_LAYOUT.md over the placeholders."
echo "  2. cd $ROOT && git init && git add -A && git commit -m 'B0-T01: repo scaffold'"
echo "  3. Assign owners for B0-B3 in PROJECT_STATUS.md, then start SETUP_GUIDE.md section 3"
echo "     (dataset access requests) today - approvals take days."
