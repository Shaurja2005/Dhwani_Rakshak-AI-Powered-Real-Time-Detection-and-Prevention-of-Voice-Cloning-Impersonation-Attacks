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
from datetime import datetime, timezone
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
    FusedWindowScore,
    HeadScore,
    RiskState,
    SessionContext,
    SessionRisk,
    RiskDriver,
    WindowSummary,
)
from packages.vg_core.sample_store import put_samples
from packages.vg_core.stub_head import StubHead
from packages.vg_core.versioning import STUB_MODEL_VERSION

log = get_logger(__name__)


def _make_session_id() -> str:
    """Generate a UUIDv7-style session ID (time-ordered)."""
    import uuid
    return str(uuid.uuid4())


def _load_wav_windows(wav_path: Path, window_s: float = 3.0, hop_s: float = 1.0):
    """Yield (window_id, samples_float32, voiced_ms) from a WAV file."""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        print(
            "soundfile and numpy are required for WAV replay.  "
            "pip install soundfile numpy",
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
        energy = float((chunk ** 2).mean())
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


def _naive_fusion(scores: list[HeadScore], window_id: int, session_id: str) -> FusedWindowScore:
    """Simple mean fusion over non-abstaining heads.  Replace with real B9 later."""
    active = [s for s in scores if not s.abstain and s.p_spoof is not None]
    abstained = [s.head_id.value for s in scores if s.abstain]

    if not active:
        return FusedWindowScore(
            session_id=session_id,
            window_id=window_id,
            p_spoof=0.0,
            state=RiskState.ABSTAIN,
            contributions={},
            heads_abstained=abstained,
            fusion_version="stub-mean-v0.1",
        )

    p_mean = sum(s.p_spoof for s in active) / len(active)  # type: ignore[misc]
    n = len(active)
    contributions = {s.head_id.value: 1.0 / n for s in active}

    if p_mean < 0.4:
        state = RiskState.LOW
    elif p_mean < 0.7:
        state = RiskState.ELEVATED
    else:
        state = RiskState.HIGH

    return FusedWindowScore(
        session_id=session_id,
        window_id=window_id,
        p_spoof=p_mean,
        state=state,
        contributions=contributions,
        heads_abstained=abstained,
        fusion_version="stub-mean-v0.1",
    )


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
    now = datetime.now(tz=timezone.utc)

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

    window_scores: list[FusedWindowScore] = []
    p_max = 0.0
    p_sum = 0.0
    abstain_count = 0

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
        fused = _naive_fusion(head_scores, window_id, session_id)
        window_scores.append(fused)

        if fused.state == RiskState.ABSTAIN:
            abstain_count += 1
        else:
            p_max = max(p_max, fused.p_spoof)
            p_sum += fused.p_spoof

        _emit(fused)

        # Simulate real-time by sleeping hop duration (1 s)
        time.sleep(0.1)  # 10× speed for dev; use time.sleep(1.0) for real-time

    total_windows = len(window_scores)
    active_windows = total_windows - abstain_count
    p_mean = p_sum / active_windows if active_windows else 0.0
    abstain_ratio = abstain_count / total_windows if total_windows else 0.0

    if p_max < 0.4:
        final_state = RiskState.LOW
    elif p_max < 0.7:
        final_state = RiskState.ELEVATED
    else:
        final_state = RiskState.HIGH

    risk_score = int(p_max * 100)

    session_risk = SessionRisk(
        session_id=session_id,
        updated_at=datetime.now(tz=timezone.utc),
        risk_score=risk_score,
        state=final_state,
        p_spoof_session_max=p_max,
        p_spoof_session_mean=p_mean,
        drivers=[
            RiskDriver(
                factor="voice_authenticity",
                weight=0.8,
                detail=f"max p_spoof={p_max:.2f} over {total_windows} windows",
            )
        ],
        timeline=[
            WindowSummary(window_id=w.window_id, p_spoof=w.p_spoof, state=w.state)
            for w in window_scores
        ],
        model_versions={hid: registry.get(hid).model_version for hid in configured_ids},
        abstain_ratio=abstain_ratio,
    )
    _emit(session_risk)


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
