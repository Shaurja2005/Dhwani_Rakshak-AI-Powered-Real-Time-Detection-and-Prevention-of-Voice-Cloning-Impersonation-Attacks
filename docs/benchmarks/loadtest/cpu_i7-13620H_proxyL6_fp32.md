# Load test — A@xlsr-proxy-L6-nes2net-v0.0.0 via torch

- Date: 2026-09-23 20:04 UTC
- Hardware: 13th Gen Intel(R) Core(TM) i7-13620H, 16 logical CPUs, torch 2.6.0+cpu (threads=10), Windows 10
- Heads: A, B, C, F (Head A batched: max_batch=16, max_wait=15 ms)
- Load shedding (B14-T06): off
- Windows power throttling (EcoQoS) disabled for this process: True
- Head A budget 2000 ms in this test; a window over budget counts as an A timeout
- Calls: synthetic speech-like PCM16 @16 kHz, 15 s each, 1 s chunks paced in real time; 3 s windows, 1 s hop
- Command: `python scripts/loadtest.py --backend proxy --sessions 1,2,4,6,8,10,12 --seconds 15 --out docs/benchmarks/loadtest/cpu_i7-13620H_proxyL6_fp32.md`

| Concurrent calls | Windows | p50 ms | p95 ms | p99 ms | max ms | Mean A batch | Degraded windows | A timeouts | Keeps up (p95 < 700 ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 13 | 251.6 | 307.4 | 318.2 | 320.9 | 1.00 | 0 | 0 | yes |
| 2 | 26 | 292.9 | 464.5 | 655.0 | 711.6 | 1.00 | 0 | 0 | yes |
| 4 | 52 | 353.2 | 459.4 | 476.7 | 486.5 | 1.33 | 0 | 0 | yes |
| 6 | 78 | 402.9 | 543.4 | 612.7 | 640.9 | 1.60 | 0 | 0 | yes |
| 8 | 104 | 638.0 | 887.4 | 977.1 | 1003.8 | 2.90 | 0 | 0 | no |
| 10 | 130 | 1130.9 | 1268.6 | 1324.9 | 1336.0 | 4.27 | 0 | 0 | no |
| 12 | 156 | 1353.6 | 1468.4 | 1569.0 | 1625.9 | 5.45 | 0 | 0 | no |

**Max concurrent real-time calls meeting the target: 6**

Cost per 1,000 call-minutes = price_per_hour / (max_calls × 60) × 1000.
First score arrives after 3.0 s of audio by construction (window length); median time-to-first-score adds p50 = 252 ms at 1 concurrent call(s).
