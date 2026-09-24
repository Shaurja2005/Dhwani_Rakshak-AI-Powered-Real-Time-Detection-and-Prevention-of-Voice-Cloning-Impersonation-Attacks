"""Load test: concurrent real-time sessions through the full gateway pipeline (B14-T05).

Each simulated call streams 16 kHz PCM16 in 1 s chunks **paced in real time**
through ``SessionPipeline`` (decode → windowing → VAD/quality → heads → fusion →
policy). Head A runs through the shared ``BatchingInferenceServer`` so windows
from all calls are batched, as in production.

Per-window latency = wall time of a ``push_chunk`` that produced exactly one
window (the full per-window path). A concurrency level "keeps up" when p95 is
under the target (default 700 ms, the CPU-only target in IMPLEMENTATION_PLAN B14).

Cost per 1,000 call-minutes = hourly machine price / (max real-time sessions × 60)
× 1,000. The price is an input (``--price-per-hour``); it is not measured.

    python scripts/loadtest.py --backend proxy --sessions 1,4,8,16,32 --seconds 12
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.vg_core.models import CallMetadata  # noqa: E402
from services.api_gateway.pipeline import Services, SessionPipeline  # noqa: E402
from services.inference.backends import (
    InferenceBackend,
    TorchBackend,
    backend_from_env,
    xlsr_shaped_proxy,
)  # noqa: E402
from services.inference.batching import BatchingInferenceServer  # noqa: E402
from services.inference.degrade import DegradeConfig, LoadController  # noqa: E402
from services.inference.served_head import BatchedHeadA  # noqa: E402

SR = 16000


def disable_power_throttling() -> bool:
    """Opt this process out of Windows EcoQoS power throttling (per-process, not a system
    setting). Without it, a benchmark started from a background shell is moved to
    efficiency cores / low clocks after ~1 s and per-window latency rises ~5x."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("v", wintypes.ULONG), ("control", wintypes.ULONG), ("state", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    st = _State(1, 0x1 | 0x4, 0)  # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION -> never throttle
    return bool(
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
    )


def speechlike(seconds: float, seed: int) -> bytes:
    """Voiced-speech surrogate: glottal pulses (jittered f0) through moving formants,
    ~4 syllables/s with short gaps, plus a little breath noise. Passes the quality gate
    (no tone/music flags), so every head actually runs."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        syl = int(rng.uniform(0.15, 0.3) * SR)
        f0 = rng.uniform(95, 230)
        src = np.zeros(syl)
        t = 0.0
        while t < syl:
            src[int(t)] = 1.0
            t += SR / (f0 * (1 + 0.03 * rng.standard_normal()))
        src += 0.02 * rng.standard_normal(syl)
        y = src
        for fc, bw in [(rng.uniform(300, 800), 90), (rng.uniform(900, 2200), 120), (2600, 200)]:
            r = np.exp(-np.pi * bw / SR)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], y)
        seg = y * np.hanning(syl)
        out[pos : pos + syl] = seg[: max(0, min(syl, n - pos))]
        pos += syl + int(rng.uniform(0.03, 0.12) * SR)
    out = 0.3 * out / (np.max(np.abs(out)) + 1e-9) + 0.002 * rng.standard_normal(n)
    return (np.clip(out, -1, 1) * 32767).astype("<i2").tobytes()


@dataclass
class LevelResult:
    sessions: int
    windows: int = 0
    lat_ms: list[float] = field(default_factory=list)
    shed_windows: int = 0
    a_timeouts: int = 0
    wall_s: float = 0.0
    batcher_mean_batch: float = 0.0
    batcher_rejected: int = 0
    degraded_transitions: int = 0

    def pct(self, q: float) -> float:
        return float(np.percentile(self.lat_ms, q)) if self.lat_ms else float("nan")


def heads_factory(server: BatchingInferenceServer, heads: list[str]):  # type: ignore[no-untyped-def]  # noqa: ANN201
    def make():  # type: ignore[no-untyped-def]  # noqa: ANN202
        out = []
        for h in heads:
            if h == "A":
                out.append(BatchedHeadA(server, budget_ms=2000))
            elif h == "B":
                from packages.vg_models.heads.head_b_dsp.head import HeadB

                out.append(HeadB())
            elif h == "C":
                from packages.vg_models.heads.head_c_prosody.head import HeadC

                out.append(HeadC())
            elif h == "F":
                from packages.vg_models.heads.head_f_watermark.head import HeadF

                out.append(HeadF())
        return out

    return make


def run_level(
    n: int,
    seconds: float,
    backend: InferenceBackend,
    heads: list[str],
    shedding: bool,
    max_batch: int,
) -> LevelResult:
    server = BatchingInferenceServer(
        backend, max_batch=max_batch, max_wait_ms=15, max_queue=256, default_timeout_ms=2000
    )
    load = LoadController(DegradeConfig()) if shedding else None
    svc = Services(heads_factory=heads_factory(server, heads), load=load)
    res = LevelResult(sessions=n)
    lock = threading.Lock()
    start = threading.Barrier(n + 1)

    def call(i: int) -> None:
        try:
            _call(i)
        except BaseException:
            start.abort()  # never leave the other calls waiting on the barrier
            raise

    def _call(i: int) -> None:
        meta = CallMetadata(
            session_id=str(uuid.uuid4()),
            tenant_id="loadtest",
            direction="inbound",
            started_at=dt.datetime.now(dt.UTC),
            channel="voip",
            codec_hint="pcm",
            source_sample_rate=SR,
            consent_basis="legitimate_use",
        )
        pipe = SessionPipeline(meta, svc, with_context=False)
        audio = speechlike(seconds, seed=i)
        start.wait()
        time.sleep((i % 10) / 10)  # calls don't arrive in lock-step
        t0 = time.perf_counter()
        step = SR * 2  # 1 s of PCM16
        for k, off in enumerate(range(0, len(audio), step)):
            t = time.perf_counter()
            events = pipe.push_chunk(audio[off : off + step], "pcm_s16le", SR)
            ms = (time.perf_counter() - t) * 1000
            ws = [e for e in events if e.type == "window_score"]
            if len(ws) == 1:
                with lock:
                    res.lat_ms.append(ms)
                    res.windows += 1
                    res.shed_windows += int(bool(ws[0].extra.get("degraded")))
            sleep = t0 + (k + 1) - time.perf_counter()  # real-time pacing
            if sleep > 0:
                time.sleep(sleep)
        pipe.close()
        a_to = sum(
            1
            for s in pipe.head_scores
            if s.head_id.value == "A"
            and s.abstain_reason is not None
            and s.abstain_reason.value == "timeout"
            and not s.evidence.get("load_shed")
        )
        with lock:
            res.a_timeouts += a_to

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(n)]
    for th in threads:
        th.start()
    for h in svc.heads_factory():  # warm the shared model once, outside the timed region
        h.warmup()
    server.infer(np.zeros(3 * SR, np.float32))
    t_all = time.perf_counter()
    start.wait()
    for th in threads:
        th.join()
    res.wall_s = time.perf_counter() - t_all
    res.batcher_mean_batch = server.stats.mean_batch
    res.batcher_rejected = server.stats.rejected
    res.degraded_transitions = len(load.transitions) if load else 0
    server.close()
    return res


def hardware() -> str:
    cpu = platform.processor()
    try:
        import subprocess

        ps = shutil.which("powershell") if sys.platform == "win32" else None
        if ps:
            query = "(Get-CimInstance Win32_Processor).Name"
            out = subprocess.run(  # noqa: S603 - fixed read-only query
                [ps, "-NoProfile", "-Command", query], capture_output=True, text=True, timeout=20
            )
            cpu = out.stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            cpu = next(
                ln.split(":", 1)[1].strip()
                for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")
            )
    except Exception:  # noqa: BLE001, S110 - best-effort description
        pass
    import torch

    return (
        f"{cpu}, {os.cpu_count()} logical CPUs, torch {torch.__version__} "
        f"(threads={torch.get_num_threads()}), {platform.system()} {platform.release()}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--backend", default="proxy", help="proxy | proxy-int8 | env (VG_INFERENCE_BACKEND)"
    )
    ap.add_argument("--proxy-layers", type=int, default=6)
    ap.add_argument("--sessions", default="1,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--heads", default="A,B,C,F")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--target-p95-ms", type=float, default=700.0)
    ap.add_argument("--load-shedding", action="store_true")
    ap.add_argument(
        "--price-per-hour", type=float, default=None, help="USD/hour of this machine class"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--allow-power-throttling", action="store_true", help="keep Windows EcoQoS (not advised)"
    )
    args = ap.parse_args(argv)
    unthrottled = False if args.allow_power_throttling else disable_power_throttling()

    if args.backend.startswith("proxy"):
        backend: InferenceBackend = TorchBackend(
            xlsr_shaped_proxy(args.proxy_layers), trained=False, int8=args.backend.endswith("int8")
        )
    else:
        backend = backend_from_env()
    heads = [h.strip().upper() for h in args.heads.split(",") if h.strip()]
    results = []
    for n in [int(x) for x in args.sessions.split(",")]:
        r = run_level(n, args.seconds, backend, heads, args.load_shedding, args.max_batch)
        results.append(r)
        print(
            f"sessions={n:3d} windows={r.windows:4d} p50={r.pct(50):7.1f} p95={r.pct(95):7.1f} "
            f"p99={r.pct(99):7.1f} ms  batch={r.batcher_mean_batch:.2f} "
            f"degraded_windows={r.shed_windows} a_timeouts={r.a_timeouts}",
            flush=True,
        )

    cli_args = argv if argv is not None else sys.argv[1:]
    ok = [
        r
        for r in results
        if r.pct(95) < args.target_p95_ms and r.shed_windows == 0 and r.a_timeouts == 0
    ]
    max_rt = max((r.sessions for r in ok), default=0)
    lines = [
        f"# Load test — {backend.model_version} via {backend.name}",
        "",
        f"- Date: {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Hardware: {hardware()}",
        f"- Heads: {', '.join(heads)} "
        f"(Head A batched: max_batch={args.max_batch}, max_wait=15 ms)",
        f"- Load shedding (B14-T06): {'on' if args.load_shedding else 'off'}",
        f"- Windows power throttling (EcoQoS) disabled for this process: {unthrottled}",
        "- Head A budget 2000 ms in this test; a window over budget counts as an A timeout",
        f"- Calls: synthetic speech-like PCM16 @16 kHz, {args.seconds:.0f} s each, "
        "1 s chunks paced in real time; 3 s windows, 1 s hop",
        f"- Command: `python scripts/loadtest.py {' '.join(cli_args)}`",
        "",
        "| Concurrent calls | Windows | p50 ms | p95 ms | p99 ms | max ms | Mean A batch "
        f"| Degraded windows | A timeouts | Keeps up (p95 < {args.target_p95_ms:.0f} ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.sessions} | {r.windows} | {r.pct(50):.1f} | {r.pct(95):.1f} | {r.pct(99):.1f} | "
            f"{max(r.lat_ms, default=float('nan')):.1f} | {r.batcher_mean_batch:.2f} | "
            f"{r.shed_windows} | {r.a_timeouts} | {'yes' if r in ok else 'no'} |"
        )
    lines += ["", f"**Max concurrent real-time calls meeting the target: {max_rt}**", ""]
    if args.price_per_hour and max_rt:
        cost = args.price_per_hour / (max_rt * 60) * 1000
        lines.append(
            f"Cost per 1,000 call-minutes at an assumed ${args.price_per_hour:.3f}/h: "
            f"**${cost:.3f}** (price is an input, not a measurement)."
        )
    else:
        lines.append("Cost per 1,000 call-minutes = price_per_hour / (max_calls × 60) × 1000.")
    lines.append(
        f"First score arrives after 3.0 s of audio by construction (window length); "
        f"median time-to-first-score adds p50 = {results[0].pct(50):.0f} ms "
        f"at {results[0].sessions} concurrent call(s)."
    )
    report = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
