.PHONY: dev-up dev-down test lint lint-fix proto fetch-models replay eval corpus-report prod-up install install-dev help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development environment
# ---------------------------------------------------------------------------
install:  ## Install runtime dependencies
	pip install -e .

install-dev:  ## Install dev + runtime dependencies
	pip install -e ".[dev,ml]"

dev-up:  ## Start the dev stack (postgres, redis, minio, prometheus, grafana)
	docker compose -f deploy/docker-compose.dev.yml up -d
	@echo "Dev stack ready."
	@echo "  Grafana:    http://localhost:3001  (admin/admin)"
	@echo "  MinIO:      http://localhost:9001  (minioadmin/minioadmin)"
	@echo "  Prometheus: http://localhost:9090"

dev-down:  ## Stop and remove the dev stack containers
	docker compose -f deploy/docker-compose.dev.yml down

dev-logs:  ## Tail logs from the dev stack
	docker compose -f deploy/docker-compose.dev.yml logs -f

# ---------------------------------------------------------------------------
# Testing and linting
# ---------------------------------------------------------------------------
test:  ## Run the full test suite with coverage
	pytest -q --tb=short

test-fast:  ## Run tests excluding slow integration tests
	pytest -q --tb=short -m "not integration and not slow"

lint:  ## Check code quality (ruff + black + mypy)
	ruff check .
	black --check .
	mypy packages services

lint-fix:  ## Auto-fix lint and formatting issues
	ruff check --fix .
	black .

# ---------------------------------------------------------------------------
# Protobuf / code generation
# ---------------------------------------------------------------------------
proto:  ## Generate Python gRPC stubs from proto/voiceguard.proto
	python -m grpc_tools.protoc \
	  -Iproto \
	  --python_out=packages/vg_core/generated \
	  --grpc_python_out=packages/vg_core/generated \
	  --pyi_out=packages/vg_core/generated \
	  proto/voiceguard.proto
	@echo "Generated stubs in packages/vg_core/generated/"

schemas-gen:  ## Re-generate Pydantic models from JSON schemas
	datamodel-codegen \
	  --input schemas/ \
	  --input-file-type jsonschema \
	  --output packages/vg_core/generated/models_generated.py \
	  --class-name-prefix VG \
	  --use-annotated \
	  --target-python-version 3.11
	@echo "Generated models in packages/vg_core/generated/models_generated.py"

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
fetch-models:  ## Download and verify pinned model checkpoints
	python scripts/fetch_models.py --registry models/registry.yaml

# ---------------------------------------------------------------------------
# Development replay
# ---------------------------------------------------------------------------
replay:  ## Replay a WAV through the full pipeline: make replay WAV=path/to/file.wav
ifndef WAV
	$(error WAV is not set. Usage: make replay WAV=path/to/file.wav)
endif
	python scripts/replay.py --wav $(WAV)

replay-sample:  ## Replay the built-in sample fixture
	python scripts/replay.py --wav tests/fixtures/sample.wav

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
eval:  ## Run the full evaluation harness: make eval MODEL=<version>
ifndef MODEL
	$(error MODEL is not set. Usage: make eval MODEL=A@xlsr300m-nes2net-v0.3.1)
endif
	python ml/eval/run_eval.py --model $(MODEL) --out docs/benchmarks/REPORT.md

corpus-report:  ## Print a summary of the training corpus by language × generator × codec
	python ml/data/manifest.py --report

# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------
prod-up:  ## Start the single-node production stack
	docker compose -f deploy/docker-compose.prod.yml up -d
