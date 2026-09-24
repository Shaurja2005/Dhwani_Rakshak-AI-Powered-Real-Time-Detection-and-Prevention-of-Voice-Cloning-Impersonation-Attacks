# Load test — A@xlsr-proxy-L6-nes2net-v0.0.0 via torch

- Date: 2026-09-23 20:07 UTC
- Hardware: 13th Gen Intel(R) Core(TM) i7-13620H, 16 logical CPUs, torch 2.6.0+cpu (threads=10), Windows 10
- Heads: A, B, C, F (Head A batched: max_batch=16, max_wait=15 ms)
- Load shedding (B14-T06): off
- Windows power throttling (EcoQoS) disabled for this process: True
- Head A budget 2000 ms in this test; a window over budget counts as an A timeout
- Calls: synthetic speech-like PCM16 @16 kHz, 20 s each, 1 s chunks paced in real time; 3 s windows, 1 s hop
- Command: `python scripts/loadtest.py --backend proxy --sessions 16,24 --seconds 20 --out docs/benchmarks/loadtest/cpu_i7-13620H_overload_no_shedding.md`

| Concurrent calls | Windows | p50 ms | p95 ms | p99 ms | max ms | Mean A batch | Degraded windows | A timeouts | Keeps up (p95 < 700 ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 16 | 288 | 1697.2 | 1842.8 | 1909.6 | 2026.7 | 7.62 | 0 | 0 | no |
| 24 | 432 | 2088.3 | 2177.0 | 2242.1 | 2280.4 | 12.60 | 0 | 411 | no |

**Max concurrent real-time calls meeting the target: 0**

Cost per 1,000 call-minutes = price_per_hour / (max_calls × 60) × 1000.
First score arrives after 3.0 s of audio by construction (window length); median time-to-first-score adds p50 = 1697 ms at 16 concurrent call(s).
