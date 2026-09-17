# VoiceGuard API — 10-minute quickstart

Goal: from a fresh clone to a risk score for **your own WAV file** in under 10
minutes, with no telephony. (B12-T08)

> Detection is **advisory**, never authoritative. Until models are trained
> (see `docs/SETUP_PENDING.md`), detection heads honestly **abstain** — you will
> see `ABSTAIN` or a low score. The API, evidence and policy flow are all real.

## 1. Install (≈3 min)

```bash
pip install -e ".[dev]"
```

## 2. Start the gateway (≈1 min)

```bash
VG_BOOTSTRAP_KEY=vg_dev_quickstart_key_000000 python -m services.api_gateway.main
```

REST + WebSocket on `http://localhost:8080` (interactive docs at `/docs`),
gRPC on `localhost:50051`. New tenants start in **shadow mode**: decisions are
recorded but no alerts are sent.

## 3. Score a file (≈1 min)

```bash
python examples/quickstart.py path/to/your.wav
```

Or with curl:

```bash
curl -s http://localhost:8080/v1/analyze/file \
  -H "Authorization: Bearer vg_dev_quickstart_key_000000" -H "Content-Type: application/json" \
  -d "{\"call_metadata\": {\"session_id\": \"$(python -c 'import uuid;print(uuid.uuid4())')\", \"tenant_id\": \"demo\", \"direction\": \"inbound\", \"started_at\": \"2026-09-17T10:00:00Z\", \"channel\": \"file\", \"codec_hint\": \"pcm\", \"source_sample_rate\": 16000, \"consent_basis\": \"legitimate_use\"}, \"audio_base64\": \"$(base64 -w0 your.wav)\"}"
```

## 4. Stream in real time (≈3 min)

Python (session REST, works through any proxy):

```python
from sdks.python.voiceguard import VoiceGuardClient
vg = VoiceGuardClient("http://localhost:8080", "vg_dev_quickstart_key_000000")
with vg.stream(tenant_id="demo", sample_rate=16000) as s:
    for chunk in chunks_of_16k_s16le_pcm:        # e.g. 100 ms each
        for event in s.send_audio(chunk):
            print(event["type"], event["data"].get("state"))
```

gRPC bidirectional streaming: `sdks.python.voiceguard.grpc_stream(...)`.
Browser / Node: `@voiceguard/sdk` → `vg.stream({tenantId, onEvent})` over `WS /v1/stream`.

## 5. Next steps

| Task | Endpoint / SDK |
|---|---|
| Evidence bundle for an alert | `GET /v1/evidence/{bundle_id}` · `vg.evidence()` |
| Analyst feedback | `POST /v1/evidence/{id}/feedback` · `vg.feedback()` |
| Signed webhooks | `POST /v1/tenants/{t}/webhooks` → verify `X-VoiceGuard-Signature` |
| Enrol a voiceprint (explicit consent required) | `POST /v1/tenants/{t}/speakers/{id}/enrollments` |
| Leave shadow mode | `POST /v1/tenants/{t}/shadow {"shadow_mode": false}` |
| Telephony | `services/api_gateway/connectors.py` (Twilio, Asterisk, FreeSWITCH) |

Full reference: `docs/api/openapi.json` (regenerate with `python scripts/export_openapi.py`).
