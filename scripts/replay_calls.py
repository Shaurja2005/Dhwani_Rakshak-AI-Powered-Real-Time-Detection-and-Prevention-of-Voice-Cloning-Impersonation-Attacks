"""Replay a call set through a running gateway, in real time (B17 DoD: live dashboard traffic).

    python scripts/replay_calls.py --gateway http://localhost:8080 --tenant demo \
        --api-key $VG_BOOTSTRAP_KEY --dir path/to/wavs --calls 40 --concurrency 4

Each call streams 16 kHz PCM16 in 200 ms chunks paced at real time through the
Python SDK (session REST API). If ``--dir`` has no WAVs, speech-like synthetic calls
are generated (clearly labelled "synthetic" in the caller number). A fraction of calls
also send a scripted social-engineering transcript so the intent layer and the fraud
dashboard have something to show. Consent basis is ``legitimate_use`` (fraud detection).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdks.python.voiceguard import VoiceGuardClient  # noqa: E402

SR = 16000
SCRIPT = [
    {
        "start_ms": 1000,
        "end_ms": 5000,
        "text": "This is your bank's fraud team, your account is blocked.",
    },
    {"start_ms": 5000, "end_ms": 9000, "text": "Tell me the OTP right now and do not tell anyone."},
]


def synth_call(seconds: float, seed: int) -> np.ndarray:
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n, pos, out = int(seconds * SR), 0, np.zeros(int(seconds * SR))
    while pos < n:
        syl = int(rng.uniform(0.15, 0.3) * SR)
        src = np.zeros(syl)
        t, f0 = 0.0, rng.uniform(95, 230)
        while t < syl:
            src[int(t)] = 1.0
            t += SR / (f0 * (1 + 0.03 * rng.standard_normal()))
        y = src + 0.02 * rng.standard_normal(syl)
        for fc, bw in [(rng.uniform(300, 800), 90), (rng.uniform(900, 2200), 120), (2600, 200)]:
            r = np.exp(-np.pi * bw / SR)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], y)
        seg = (y * np.hanning(syl))[: max(0, min(syl, n - pos))]
        out[pos : pos + len(seg)] = seg
        pos += syl + int(rng.uniform(0.03, 0.4) * SR)
    return (0.3 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def load_calls(directory: Path | None, n: int, seconds: float) -> list[tuple[str, np.ndarray]]:
    calls: list[tuple[str, np.ndarray]] = []
    if directory and directory.exists():
        import soundfile as sf
        from scipy.signal import resample_poly

        for p in sorted(directory.glob("*.wav"))[:n]:
            x, sr = sf.read(p, dtype="float32", always_2d=False)
            x = x.mean(axis=1) if x.ndim > 1 else x
            if sr != SR:
                x = resample_poly(x, SR, sr).astype(np.float32)
            calls.append((p.stem, x))
    while len(calls) < n:
        i = len(calls)
        calls.append((f"synthetic-{i:03d}", synth_call(seconds, i)))
    return calls


def run_call(
    client: VoiceGuardClient,
    tenant: str,
    name: str,
    pcm: np.ndarray,
    scripted: bool,
    chunk_s: float = 0.2,
) -> str:
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(chunk_s * SR) * 2
    state = "?"
    with client.stream(
        tenant_id=tenant, caller_number=f"+91-replay-{name}", consent_basis="legitimate_use"
    ) as s:
        t0 = time.monotonic()
        for k, off in enumerate(range(0, len(pcm16), step)):
            for ev in s.send_audio(pcm16[off : off + step]):
                if ev["type"] == "session_risk":
                    state = ev["data"]["state"]
            if scripted and k == int(3 / chunk_s):
                s.send_transcript(SCRIPT)
            delay = t0 + (k + 1) * chunk_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    return f"{name}: final state {state}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--api-key", default=os.getenv("VG_API_KEY"))
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=20.0, help="length of synthetic calls")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--scripted-fraction", type=float, default=0.25)
    ap.add_argument("--loop", action="store_true", help="repeat forever (dashboards demo)")
    args = ap.parse_args(argv)
    if not args.api_key:
        print("--api-key or VG_API_KEY is required", file=sys.stderr)
        return 2
    client = VoiceGuardClient(args.gateway, api_key=args.api_key, timeout=60)
    calls = load_calls(args.dir, args.calls, args.seconds)
    rnd = random.Random(0)  # noqa: S311 - which calls get a script, not crypto
    while True:
        with cf.ThreadPoolExecutor(args.concurrency) as pool:
            futs = [
                pool.submit(
                    run_call, client, args.tenant, n, x, rnd.random() < args.scripted_fraction
                )
                for n, x in calls
            ]
            for f in cf.as_completed(futs):
                print(f.result(), flush=True)
        if not args.loop:
            return 0


if __name__ == "__main__":
    sys.exit(main())
