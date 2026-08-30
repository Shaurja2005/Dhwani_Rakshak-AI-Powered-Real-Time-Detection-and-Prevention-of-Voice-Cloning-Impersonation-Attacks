# VoiceGuard

Real-time detection and prevention of voice cloning impersonation attacks.

**Read `SOURCE_OF_TRUTH.md` before writing any code.** Task board: `PROJECT_STATUS.md`. Working rules: `AGENTS.md`.

Quickstart:

```bash
make dev-up
make fetch-models
python scripts/replay.py --wav tests/fixtures/sample.wav
```
