"""scripts/replay.py — Dev entrypoint: replay a WAV through the full pipeline.

This is how the team works without a PBX.  It reads a WAV file, splits it
into 3-second windows, runs all registered heads (stubs by default), applies
naive fusion, and streams risk events to stdout as JSON.

Usage::

    python scripts/replay.py --wav tests/fixtures/sample.wav
    make replay WAV=tests/fixtures/sample.wav

For the Walking Skeleton phase (B0), all heads are StubHeads.
As real heads are implemented (B4–B8) they can be loaded via:

    python scripts/replay.py --wav sample.wav --heads A  (load real head A)

Keep this script working. It is the canonical development feedback loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path so we can import packages without installing
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from packages.vg_core.config import settings
from packages.vg_core.head_api import HeadRegistry
from packages.vg_core.logging import configure_logging, get_logger
from packages.vg_core.models import (
    AnalysisWindow,
    CallMetadata,
    Channel,
    ConsentBasis,
    SessionContext,
)
from packages.vg_core.sample_store import put_samples
from packages.vg_core.stub_head import StubHead
from services.fusion.engine import RiskEngine

log = get_logger(__name__)


def _make_session_id() -> str:
    """Generate a UUIDv7-style session ID (time-ordered)."""
    import uuid

    return str(uuid.uuid4())


def _load_wav_windows(wav_path: Path, window_s: float = 3.0, hop_s: float = 1.0):
    """Yield (window_id, samples_float32, voiced_ms) from a WAV file."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print(
            "soundfile and numpy are required for WAV replay.  " "pip install soundfile numpy",
            file=sys.stderr,
        )
        sys.exit(1)

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # to mono
    if sr != 16000:
        try:
            import scipy.signal as ss

            data = ss.resample_poly(data, 16000, sr).astype("float32")
            sr = 16000
        except ImportError:
            print(
                f"Warning: soundfile loaded {sr} Hz but scipy not available for resampling. "
                "Using raw samples — results may be inaccurate.",
                file=sys.stderr,
            )

    win_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)
    window_id = 0

    for start in range(0, len(data) - win_samples + 1, hop_samples):
        chunk = data[start : start + win_samples]
        # Simple energy-based voiced fraction
        energy = float((chunk**2).mean())
        voiced_ratio = min(1.0, energy * 100)
        voiced_ms = int(voiced_ratio * window_s * 1000)

        yield (
            window_id,
            start / sr,
            (start + win_samples) / sr,
            chunk,
            voiced_ms,
        )
        window_id += 1


def _emit(obj: object) -> None:
    """Write a JSON object to stdout."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    else:
        data = obj
    print(json.dumps(data, default=str))
    sys.stdout.flush()


def run_replay(
    wav_path: Path, head_ids: list[str] | None = None, real_heads: list[str] | None = None
) -> None:
    session_id = _make_session_id()
    now = datetime.now(tz=UTC)

    # Register stub heads (real heads can be loaded by head_ids parameter)
    registry = HeadRegistry.get_instance()
    configured_ids = head_ids or list("ABCDEF")
    real = set(real_heads or [])
    for hid in configured_ids:
        if hid == "A" and "A" in real:
            from packages.vg_models.heads.head_a_ssl.head import HeadA

            registry.register(HeadA())  # VG_HEAD_A_CHECKPOINT selects trained weights
        elif hid == "B" and "B" in real:
            from packages.vg_models.heads.head_b_dsp.head import HeadB

            registry.register(HeadB())  # VG_HEAD_B_MODEL selects a trained GBDT
        elif hid == "C" and "C" in real:
            from packages.vg_models.heads.head_c_prosody.head import HeadC

            registry.register(HeadC())  # VG_HEAD_C_MODEL selects a trained prosody model
        else:
            registry.register(StubHead(head_id=hid))
    registry.warmup_all()

    metadata = CallMetadata(
        session_id=session_id,
        tenant_id=settings.tenant_default,
        direction="inbound",
        started_at=now,
        channel=Channel.FILE,
        codec_hint="pcm",
        source_sample_rate=16000,
        consent_basis=ConsentBasis.LEGITIMATE_USE,
        shadow_mode=settings.shadow_mode,
    )
    _emit(metadata)

    ctx = SessionContext(
        session_id=session_id,
        tenant_id=settings.tenant_default,
        shadow_mode=settings.shadow_mode,
        call_metadata=metadata,
    )

    engine = RiskEngine(session_id)  # B9: calibration -> fusion -> temporal -> states
    session_risk = None

    for window_id, start_s, end_s, chunk, voiced_ms in _load_wav_windows(wav_path):
        samples_ref = put_samples(f"shm://replay/{session_id}/{window_id}", chunk)
        window = AnalysisWindow(
            session_id=session_id,
            window_id=window_id,
            start_ms=int(start_s * 1000),
            end_ms=int(end_s * 1000),
            samples_ref=samples_ref,
            voiced_ms=voiced_ms,
            snr_db=18.0,  # replay: assume clean audio
            clipping_ratio=0.0,
            quality_ok=(voiced_ms > int(settings.min_voiced_s * 1000)),
            original_sample_rate=16000,
        )

        head_scores = [registry.get(hid).score(window, ctx) for hid in configured_ids]
        fused, session_risk = engine.update(window_id, head_scores)
        _emit(fused)

        # Simulate real-time by sleeping hop duration (1 s)
        time.sleep(0.1)  # 10× speed for dev; use time.sleep(1.0) for real-time

    _emit(session_risk if session_risk is not None else engine.session_risk())


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a WAV through the VoiceGuard pipeline")
    parser.add_argument("--wav", required=True, type=Path, help="Path to the WAV file")
    parser.add_argument(
        "--heads",
        nargs="*",
        default=None,
        help="Head IDs to enable (default: all stub heads A-F)",
    )
    parser.add_argument(
        "--real-heads",
        nargs="*",
        default=None,
        help="Head IDs to run with the real implementation instead of a stub (e.g. A)",
    )
    parser.add_argument("--log-level", default="WARNING", help="Log level (default: WARNING)")
    args = parser.parse_args()

    configure_logging(level=args.log_level, json_output=False)

    if not args.wav.exists():
        print(f"Error: WAV file not found: {args.wav}", file=sys.stderr)
        sys.exit(1)

    log.info("replay_start", wav=str(args.wav))
    run_replay(args.wav, head_ids=args.heads, real_heads=args.real_heads)


if __name__ == "__main__":
    main()
