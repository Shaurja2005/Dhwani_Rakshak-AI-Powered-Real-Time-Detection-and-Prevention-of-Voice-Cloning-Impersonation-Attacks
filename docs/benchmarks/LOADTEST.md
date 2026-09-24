# Load test summary (B14-T05)

These are measured numbers on named hardware, not targets. Per-run detail, including the exact commands, is in
[`loadtest/`](loadtest/). Rerun with `scripts/loadtest.py`.

**Hardware:** laptop, 13th Gen Intel Core i7-13620H (6P + 4E cores, 16 logical CPUs), 16 GB RAM, CPU only,
torch 2.6.0+cpu (10 intra-op threads), Windows 11. Windows EcoQoS power throttling was disabled for the
benchmark process only (see the caveats below).

**What runs per window** is the full gateway path (`SessionPipeline`): PCM16 decode → 3 s / 1 s windowing →
VAD + quality gate → heads A, B, C, F → fusion (B9) → policy on state change (B11). Calls are streamed in
1 s chunks, **paced in real time**. Head A goes through the shared cross-call batcher
(`services/inference/batching.py`, max batch 16, max wait 15 ms).

## Head A model used: a latency proxy, not the trained student

No trained Head A exists yet (B4 needs the dataset setup). Head A therefore runs a **random-weight model with the
exact tensor shapes of the planned distilled student**: an XLS-R-shaped front-end (512-channel conv encoder,
1024-d, 16 heads, FFN 4096) truncated to **6 transformer layers**, plus a Nes2Net back-end, 82 M parameters.
Compute cost depends on shapes, not on weight values. The head still abstains `untrained`, but the inference
runs on every window. Re-measure with the real distilled student (`--backend env`) once it exists.

## Results: CPU tier (target: p95 < 700 ms per window)

| Config | Max real-time calls with p95 < 700 ms | p50 / p95 at that load | Single call p50 / p95 |
|---|---:|---:|---:|
| FP32 proxy, 6 layers | **6** | 403 / 543 ms | 252 / 307 ms |
| Dynamic INT8 proxy, 6 layers | **8** | 402 / 579 ms | 222 / 242 ms |

Past the knee, latency climbs above the 1 s hop: at 10 calls p95 is 1269 ms (FP32) and 1112 ms (INT8), so
calls fall behind real time. Batching kicks in as expected under load (mean Head A batch 1.0 → 5.5 at 12
calls). On CPU, however, per-item cost grows almost linearly with batch size, so batching buys little
throughput here. It pays off on GPU.

INT8 dynamic quantization only covers Linear layers. The conv feature encoder stays FP32, which caps the
gain at about 12% latency or +2 calls. **The INT8 accuracy cost is not measured yet.** A random model has no
accuracy to lose, so the EER delta must come from `ml.export.quantize.compare` on the labelled dev set with
the trained student (SETUP_PENDING B14).

## Overload and graceful degradation (B14-T06)

At 16 and 24 simultaneous calls, which is 2–3× the CPU capacity above:

| Calls | Load shedding | p50 | p95 | Windows with Head A shed (fast path) | Head A over budget |
|---:|---|---:|---:|---:|---:|
| 16 | off | 1697 ms | 1843 ms | 0 | 0 |
| 16 | **on** | **46 ms** | 1400 ms | 238 / 288 | 0 |
| 24 | off | 2088 ms | 2177 ms | 0 | 411 / 432 |
| 24 | **on** | **99 ms** | 2197 ms | 351 / 432 | 42 |

With shedding on, most windows are scored immediately by heads B, C and F. Those scores carry
`degraded=true` and reduced confidence; Head A abstains with `timeout` and `load_shed`. Nothing is queued
behind live audio. **The p95 tail is not fixed yet.** Every `min_degraded_s` (5 s) the controller
re-admits Head A as a probe, and that burst is slow at this load. Shaping the probe (admit a fraction of
windows instead of all of them) is follow-up tuning. The numbers above are published as measured.

## Cost per 1,000 call-minutes

`cost = hourly price / (max real-time calls × 60) × 1000`. This laptop has no hourly price, so no dollar
figure is given. For example, a cloud VM of comparable CPU class at price *P* $/h serving 8 calls (INT8) costs
*P* × 2.08 $ per 1,000 call-minutes. Pass `--price-per-hour` for a real quote.

## Caveats (read before quoting any number)

1. **Laptop CPU, not the plan's T4/L4.** The GPU numbers (target p95 < 300 ms) need Triton on a GPU host.
   Open question Q2 (target hardware) is still unanswered.
2. **Proxy weights.** The latency is representative; the scores are not. Detection accuracy lives in
   `docs/benchmarks/REPORT.md` (B15), and nothing here is an accuracy claim.
3. **Synthetic calls.** The audio is speech-like (glottal pulses through moving formants) so that it passes
   the quality gate and every head runs. Real calls have silences that the VAD skips, which would make
   them cheaper.
4. **Windows EcoQoS.** Started from a background shell, the process is moved to efficiency cores after
   about 1 s and per-window latency rises about 5× (a 150 ms Head A call becomes ~750 ms). The load test
   opts its own process out (`SetProcessInformation(ProcessPowerThrottling)`). Linux servers are not
   affected.
5. **Two pipeline defects were found and fixed while producing these numbers** (open question Q8):
   - the B2 quality gate's tone test was mis-scaled by about N, so it flagged most real audio;
   - its Goertzel loop cost about 1 s per window.

   Earlier runs made before these fixes were discarded.
