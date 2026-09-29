# On-call runbook: VoiceGuard (B17-T08)

The detector is **advisory** (I1): when in doubt, degrade to "more friction for customers"
(call-backs, step-up), never to "silently wave calls through". `ABSTAIN` means *could not
judge*. It never means *genuine*.

Dashboards: Grafana, folders **VoiceGuard — Operations** and **VoiceGuard — Fraud analytics**.
Alert rules: `deploy/observability/alerts.yml`. Each alert links to a section below.

First 5 minutes, whatever fired:
1. Open the Operations dashboard. Check windows/s, active calls, p95 latency, abstain ratio,
   load shedding, and model versions in use.
2. `docker compose -f deploy/docker-compose.prod.yml ps` (or `kubectl -n voiceguard get pods`).
   Is everything healthy? Were there restarts?
3. Did something change? Check `python scripts/model_rollout.py status`, recent deploys, and
   telephony or codec changes on the PBX side.

---

## Abstain rate spike

*Alert `VGAbstainRateSpike`: a head abstains more than twice as often as yesterday.*

Look at **Abstentions / s by reason**. The reason tells you where to look:

| Reason | Usual cause | Action |
|---|---|---|
| `quality_gate` | Audio changed upstream: new codec, hold music, one-way audio, level too low | Check the PBX/SBC and the per-codec mix. Replay a sample call (`make replay WAV=...`). Do **not** loosen the quality gate on the call path. |
| `insufficient_speech` | Short calls, IVR prompts, VAD mis-tuned for a new channel | Check call durations. Escalate to B2 if the VAD is misfiring. |
| `timeout` | Overload (see Latency) or a stuck inference backend | Check queue depth and latency. Restart inference if the queue is flat but timeouts keep rising. |
| `untrained` | The head has no trained checkpoint (registry empty or rolled back to none) | `scripts/model_rollout.py status`. Promote a validated model or accept the reduced layer. |
| `no_enrollment` | Head D: customer not enrolled | Expected. Only a problem if enrolments were deleted unexpectedly (check the audit log). |

The fused score still works on the remaining heads. Tell the fraud team which layer is
missing until the head recovers.

## Latency degraded

*Alerts `VGWindowLatencyHigh` (p95 > 700 ms) and `VGLoadShedding`.*

1. **Per-head latency p95** shows which head is slow (almost always Head A).
2. **Queue depth** rising means capacity-bound. Scale inference (`kubectl -n voiceguard scale
   deploy/vg-inference-blue --replicas=N`) or add nodes. Measured capacity per node is in
   `docs/benchmarks/REPORT.md#serving`.
3. Load shedding **is the correct behaviour under overload** (B14-T06). Head A is skipped,
   other heads keep scoring with reduced confidence, and no audio is queued. Leave it on. Fix
   capacity.
4. On Windows hosts check for power throttling (see `docs/benchmarks/LOADTEST.md`, caveat 4).
   On Linux check CPU governor and thermal throttling.
5. If latency rose right after a model promotion, **roll back** (see Model rollout).

## Alert rate tripled

*Alert `VGAlertRateTripled`: ELEVATED/HIGH decisions are 3× yesterday's rate.*

This is either a real attack wave or a false-positive storm. Tell them apart fast:

1. **Fraud analytics → Alerts by tier (shadow vs live)**. Is it one tenant or tier, or everywhere?
2. **Window states** and **p_spoof by head**. Is one head driving it? A single head suddenly
   scoring everything high points to a model or channel problem, not an attack.
3. Ask the analysts for the latest feedback labels: `false_positive` or `true_positive`?
4. **False-positive storm** (a new codec, a new region's accent, or a model change):
   - Roll back the model if one changed recently.
   - Otherwise put the affected tenant in **shadow mode** (`POST /v1/tenants/{id}/shadow`) so
     agents stop seeing alerts while it is investigated. Record the decision.
   - Never raise thresholds on your own (AGENTS.md §9). Thresholds come from the cost model and
     need sign-off.
5. **Real attack wave**: leave alerts on. Notify the fraud team and the bank's SOC. The evidence
   bundles (`/v1/evidence/{id}`) are what the investigators need.

## Score drift

*Alert `VGScoreDrift`: PSI > 0.25 on the bona fide proxy (calls that stayed LOW).*

The distribution of scores on calls we believe are genuine has moved. Causes, most likely first:

1. A channel change (new codec, AGC, noise suppression on the agent side). Correlate with telephony changes.
2. A new population (new tenant, region or language), which is a fairness concern. Check the
   per-language FPR in the latest benchmark report.
3. **A new generator family that the model scores as genuine.** This is the reason the alert exists.
   Pull a sample of recent LOW calls with analyst `false_negative` suspicion. Follow the new-family
   procedure (B4 continual learning, `ml/training/continual.py`) and re-run `make eval`.

Refresh the reference only after the cause is understood:
`observer.drift.freeze_current_as_reference(head)`. Save the reference to
`$VG_DATA_DIR/drift_reference.json` and note it in the change log.

## Model rollout

Promotion and rollback go through the registry. Gateways and inference pods follow it
without a restart.

```bash
python scripts/model_rollout.py status
python scripts/model_rollout.py attach-eval --version A@... --run-id r...   # from docs/benchmarks/runs
python scripts/model_rollout.py promote --version A@... --actor <you>
python scripts/model_rollout.py rollback --kind head_a --actor <you>          # instant
```

Promotion is refused without an evaluation run, if the fairness gate failed, and, for a
commercial deployment, for research-lineage models (I7). For infrastructure-level blue/green,
scale `vg-inference-green`, wait for Ready, then flip the `vg-inference` Service selector
(`deploy/k8s/overlays/prod`).

After a promotion, watch for 30 minutes: latency, abstain ratio, alert rate, and `vg_model_info`.

## Gateway down

*Alert `VGGatewayDown`.*

1. `docker compose ... logs gateway --tail 200` / `kubectl -n voiceguard logs deploy/vg-gateway`.
2. Common causes:
   - A missing secret (`VG_KEK_B64`, `VG_VAULT_KEY_<TENANT>`).
   - A registry checksum mismatch (the model file was replaced). The log line is
     `registry_checksum_mismatch`. Restore the file or roll back.
   - The data volume is full.
3. While it is down, calls are **not** being screened. Tell the contact-centre lead so agents
   apply manual call-back rules for high-value requests.

## Privacy incidents

- **Suspected raw audio at rest:** stop the affected service. Run `pytest -m compliance`
  against the running configuration. Inform the DPO within the hour (DPDP breach timelines).
  Never copy the suspected files off the node.
- **DSAR or erasure request:** `python -m services.privacy.main dsar-export|dsar-erase ...`.
  Every action is written to the audit log.
- **Key compromise:** `python -m services.privacy.main rotate-key --tenant <id>` re-encrypts the
  vaults and crypto-shreds the old key.
