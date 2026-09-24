# Load test — A@xlsr-proxy-L6-nes2net-v0.0.0-int8 via torch-int8

- Date: 2026-09-23 20:06 UTC
- Hardware: 13th Gen Intel(R) Core(TM) i7-13620H, 16 logical CPUs, torch 2.6.0+cpu (threads=10), Windows 10
- Heads: A, B, C, F (Head A batched: max_batch=16, max_wait=15 ms)
- Load shedding (B14-T06): off
- Windows power throttling (EcoQoS) disabled for this process: True
- Head A budget 2000 ms in this test; a window over budget counts as an A timeout
- Calls: synthetic speech-like PCM16 @16 kHz, 15 s each, 1 s chunks paced in real time; 3 s windows, 1 s hop
- Command: `python scripts/loadtest.py --backend proxy-int8 --sessions 1,2,4,6,8,10,12 --seconds 15 --out docs/benchmarks/loadtest/cpu_i7-13620H_proxyL6_int8.md`

| Concurrent calls | Windows | p50 ms | p95 ms | p99 ms | max ms | Mean A batch | Degraded windows | A timeouts | Keeps up (p95 < 700 ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 13 | 222.4 | 242.4 | 260.7 | 265.3 | 1.00 | 0 | 0 | yes |
| 2 | 26 | 259.1 | 316.8 | 327.0 | 329.2 | 1.00 | 0 | 0 | yes |
| 4 | 52 | 304.6 | 417.2 | 445.5 | 449.0 | 1.33 | 0 | 0 | yes |
| 6 | 78 | 334.5 | 457.4 | 475.9 | 479.7 | 1.49 | 0 | 0 | yes |
| 8 | 104 | 402.3 | 579.3 | 603.4 | 606.0 | 1.95 | 0 | 0 | yes |
| 10 | 130 | 1014.2 | 1111.6 | 1136.3 | 1154.1 | 4.27 | 0 | 0 | no |
| 12 | 156 | 1176.0 | 1256.4 | 1276.1 | 1295.5 | 5.45 | 0 | 0 | no |

**Max concurrent real-time calls meeting the target: 8**

Cost per 1,000 call-minutes = price_per_hour / (max_calls × 60) × 1000.
First score arrives after 3.0 s of audio by construction (window length); median time-to-first-score adds p50 = 222 ms at 1 concurrent call(s).
