# Data residency and deployment topology (B16-T03, B16-T05)

**Default: everything runs on the tenant's premises (or its in-India private cloud).** No audio,
transcript or embedding leaves that boundary (invariant I6). This follows the RBI requirement
that banking and payment data stay in India. It is also why the intent model (B10) is a local LLM
or rule engine, never a hosted API.

## Topology A: on-prem single node (default, `deploy/docker-compose.prod.yml`)

```
 tenant data centre (India) ─────────────────────────────────────────────────────────┐
 │  PBX / SBC ──SIP/RTP──▶ gateway (B12) ──▶ inference (B14) / heads (B4–B8)         │
 │                             │                 │                                  │
 │                             ├──▶ fusion / policy / context (local ASR + LLM)     │
 │                             ├──▶ postgres (evidence, timeline) · redis           │
 │                             └──▶ agent UI (B13) over TLS inside the LAN          │
 │  privacy job (B16) ─▶ retention / DSAR / key rotation · prometheus + grafana     │
 │  webhooks ─▶ only hosts on the tenant egress allowlist, payload has no audio     │
 └───────────────────────────────────────────────────────────────────────────────────┘
       ▲ no inbound or outbound internet required at run time (models are installed at setup)
```

## Topology B: edge scoring (B12 edge SDK)

The distilled ONNX model (B14) scores on the agent's device or in the branch. Only scores leave
the device, to the tenant's own gateway.

## Topology C: central processing (explicit tenant opt-in only)

Allowed only when the tenant sets `central_processing: true` in
`config/privacy/tenants/<tenant>.yaml`, the processor's data centre is in India, and the contract
and DPIA are updated. The egress guard (`services/privacy/egress.py`) refuses offshore
destinations for any tenant that has not opted in.

## Enforcement in code

| Control | Implementation | Test |
|---|---|---|
| Default-deny egress by host | `EgressGuard.check` (tenant `egress_allowlist` + `VG_EGRESS_ALLOWLIST`) | `tests/unit/test_b16_privacy.py` |
| No audio / embeddings in any outbound payload | `audio_findings` (bytes, arrays, long numeric lists, base64 blobs, audio keys) in the egress guard, webhook delivery and the logging processor | same |
| Offshore refused without opt-in | `EgressGuard.check(..., offshore=True)` | same |
| Local-only LLM for intent | B10 `services/context/intent.py` (local model / rules only) | B10 tests |
| No raw audio at rest | memory-only sample store with overwrite; compliance scan | `tests/compliance/` |

## Setup-time downloads (not run time)

Model weights and datasets are fetched during installation (`scripts/fetch_models.py`,
`ml/data/download.py`). These are inbound downloads of public artefacts, pinned by hash, and never
carry customer data. Production nodes can be air-gapped after setup.
