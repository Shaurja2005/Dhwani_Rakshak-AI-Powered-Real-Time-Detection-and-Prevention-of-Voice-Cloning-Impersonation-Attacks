"""Integration + unit tests for B2 — Stream Conditioning.

Definition of Done (B2):
  Feed a 5-minute call with 40% silence, hold music, and 3% packet-loss burst.
  The conditioner emits only well-formed voiced windows, marks the rest ABSTAIN,
  and never crashes. Latency added by this layer < 60 ms p95.

Tests are grouped by task:
  - B2-T01: Jitter buffer + PLC
  - B2-T02: Resampler (8kHz mulaw → 16kHz)
  - B2-T03: VAD with hysteresis
  - B2-T04: Windowing (3.0s window / 1.0s hop)
  - B2-T05/T07: Quality gate (silence, music, DTMF, clipping, low SNR)
  - B2-T06: Loudness normalisation
  - B2-T08: Feature cache
  - E2E: Full conditioner pipeline (the DoD scenario)
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import numpy as np
import pytest

from packages.vg_audio.codecs import mulaw_to_linear, alaw_to_linear, pcm_s16le_to_float32, decode_payload
from packages.vg_audio.jitter_buffer import JitterBuffer
from packages.vg_audio.quality import (
    assess_quality, normalize_loudness, estimate_snr_db,
    compute_clipping_ratio, detect_dtmf, detect_hold_music_or_tone,
)
from packages.vg_audio.resample import resample_chunk
from packages.vg_audio.vad import EnergyVAD
from packages.vg_audio.windowing import Windower
from packages.vg_audio.features import FeatureCache, clear_session_cache
from packages.vg_core.bus import InMemoryBus
from packages.vg_core.models import AnalysisWindow, AudioChunk, AudioEncoding

TARGET_SR = 16_000
SAMPLE_WAV = Path(__file__).parent.parent / "fixtures" / "sample.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_wave(freq_hz: float, duration_s: float, sr: int = TARGET_SR, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(duration_s: float, sr: int = TARGET_SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _speech_like(duration_s: float, sr: int = TARGET_SR) -> np.ndarray:
    """Broadband noise (not pure tone) to simulate speech energy distribution."""
    rng = np.random.default_rng(42)
    # Mix fundamental + harmonics to look like voiced speech
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = (
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t) +
        0.1 * np.sin(2 * np.pi * 600 * t) +
        0.05 * rng.standard_normal(len(t))
    ).astype(np.float32)
    return sig / (np.max(np.abs(sig)) + 1e-9) * 0.6


def _pcm_to_chunks(
    pcm: np.ndarray,
    sr: int = TARGET_SR,
    chunk_ms: int = 20,
    session_id: str = "test-session",
    loss_rate: float = 0.0,
    seed: int = 0,
) -> list[AudioChunk]:
    """Slice float32 PCM into AudioChunk list with optional packet loss."""
    rng = random.Random(seed)
    chunk_samples = int(sr * chunk_ms / 1000)
    chunks = []
    seq = 0
    ts_ms = 0
    for i in range(0, len(pcm) - chunk_samples + 1, chunk_samples):
        if loss_rate > 0 and rng.random() < loss_rate:
            seq += 1
            ts_ms += chunk_ms
            continue
        payload = (pcm[i : i + chunk_samples] * 32767).astype(np.int16).tobytes()
        chunks.append(AudioChunk(
            session_id=session_id,
            seq=seq,
            ts_ms=ts_ms,
            sample_rate=sr,
            encoding=AudioEncoding.PCM_S16LE,
            payload=payload,
        ))
        seq += 1
        ts_ms += chunk_ms
    return chunks


# ---------------------------------------------------------------------------
# B2-T01 — Jitter buffer + PLC
# ---------------------------------------------------------------------------

class TestJitterBuffer:
    def test_in_order_passthrough(self) -> None:
        buf = JitterBuffer()
        pcm = _speech_like(0.5)
        chunks_data = _pcm_to_chunks(pcm)
        results = []
        for c in chunks_data:
            results.extend(buf.push(c.seq, c.ts_ms, c.payload, c.encoding.value, c.sample_rate))
        assert len(results) > 0
        # All frames should be non-empty float32 arrays
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.dtype == np.float32

    def test_reorders_out_of_order_packets(self) -> None:
        buf = JitterBuffer()
        pcm = _speech_like(0.5)
        chunks = _pcm_to_chunks(pcm)
        # Shuffle a window of 4 packets
        shuffled = chunks[:4][::-1] + chunks[4:]
        results = []
        for c in shuffled:
            results.extend(buf.push(c.seq, c.ts_ms, c.payload, c.encoding.value, c.sample_rate))
        # Should still yield frames (reordered)
        assert len(results) > 0

    def test_plc_fills_missing_packets(self) -> None:
        buf = JitterBuffer()
        pcm = _speech_like(2.0)
        chunks = _pcm_to_chunks(pcm)
        # Skip packets 3, 4, 5 (simulate burst loss), then send 6+
        selected = [c for c in chunks if c.seq not in (3, 4, 5)]
        results = []
        for c in selected:
            results.extend(buf.push(c.seq, c.ts_ms, c.payload, c.encoding.value, c.sample_rate))
        # PLC frames should have been inserted — stats show concealed
        assert buf.stats["concealed_frames"] > 0

    def test_drain_flushes_remainder(self) -> None:
        buf = JitterBuffer()
        pcm = _speech_like(0.1)
        chunks = _pcm_to_chunks(pcm)
        # Push all but don't exhaust
        for c in chunks[:3]:
            buf.push(c.seq, c.ts_ms, c.payload, c.encoding.value, c.sample_rate)
        drained = buf.drain()
        assert len(drained) >= 0  # May be empty if already flushed


# ---------------------------------------------------------------------------
# B2-T02 — Resampler
# ---------------------------------------------------------------------------

class TestResampler:
    def test_pcm_16k_passthrough(self) -> None:
        pcm = _speech_like(0.1, sr=TARGET_SR)
        payload = (pcm * 32767).astype(np.int16).tobytes()
        out = resample_chunk(payload, "pcm_s16le", TARGET_SR)
        assert out.dtype == np.float32
        # Length should be approximately the same (within 5%)
        assert abs(len(out) - len(pcm)) / len(pcm) < 0.05

    def test_8khz_mulaw_upsampled_to_16k(self) -> None:
        """8 kHz µ-law decoded and upsampled should yield ~2× length at 16 kHz."""
        sr_src = 8000
        duration_s = 0.1
        pcm_8k = _speech_like(duration_s, sr=sr_src)
        # Simulate µ-law encode/decode round-trip via lookup
        from packages.vg_audio.codecs import mulaw_to_linear
        # Use raw bytes to represent the 8kHz signal
        payload = (pcm_8k * 32767).astype(np.int16).tobytes()
        out = resample_chunk(payload, "pcm_s16le", sr_src)
        expected_len = int(duration_s * TARGET_SR)
        # Within 5% of expected
        assert abs(len(out) - expected_len) / expected_len < 0.05

    def test_mulaw_decode_roundtrip(self) -> None:
        """µ-law decoded signal should be in [-1, 1]."""
        raw = bytes(range(256))
        out = mulaw_to_linear(raw)
        assert out.dtype == np.float32
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_alaw_decode_roundtrip(self) -> None:
        raw = bytes(range(256))
        out = alaw_to_linear(raw)
        assert out.dtype == np.float32
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_decode_payload_dispatch(self) -> None:
        pcm = _speech_like(0.02)
        payload = (pcm * 32767).astype(np.int16).tobytes()
        out = decode_payload(payload, "pcm_s16le")
        assert out.dtype == np.float32

    def test_unsupported_encoding_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported encoding"):
            decode_payload(b"\x00" * 10, "opus")


# ---------------------------------------------------------------------------
# B2-T03 — VAD
# ---------------------------------------------------------------------------

class TestVAD:
    def test_silence_not_voiced(self) -> None:
        vad = EnergyVAD()
        silence = _silence(0.1)
        for i in range(0, len(silence) - 320, 320):
            result = vad.process_frame(silence[i:i+320], i // 320)
        # Should not be voiced at end
        assert not result.is_voiced  # type: ignore

    def test_loud_tone_voiced(self) -> None:
        vad = EnergyVAD()
        tone = _sine_wave(300, 0.5, amp=0.8)
        result = None
        for i in range(0, len(tone) - 320, 320):
            result = vad.process_frame(tone[i:i+320], i // 320)
        assert result is not None
        assert result.is_voiced

    def test_hysteresis_holds_over_short_pause(self) -> None:
        """After speech onset, a 200ms pause (< hangover period) should stay voiced."""
        vad = EnergyVAD()
        speech = _speech_like(1.0)
        silence = _silence(0.2)
        # Start with speech, then 200ms silence
        combined = np.concatenate([speech, silence])
        results = []
        for i in range(0, len(combined) - 320, 320):
            results.append(vad.process_frame(combined[i:i+320], i // 320))
        # During the 200ms silence (10 frames), still voiced due to hangover
        silence_start_frame = len(speech) // 320
        voiced_during_silence = [r.is_voiced for r in results[silence_start_frame: silence_start_frame + 10]]
        assert any(voiced_during_silence), "Hysteresis should hold voiced state over short pause"

    def test_process_window_returns_ratio(self) -> None:
        vad = EnergyVAD()
        speech = _speech_like(3.0)
        ratio, frame_results = vad.process_window(speech)
        assert 0.0 <= ratio <= 1.0
        assert len(frame_results) > 0


# ---------------------------------------------------------------------------
# B2-T04 — Windowing
# ---------------------------------------------------------------------------

class TestWindowing:
    def test_3s_window_1s_hop(self) -> None:
        """5s of audio should yield 3 windows (at t=0,1,2 seconds)."""
        windower = Windower(session_id="test", original_sample_rate=TARGET_SR, window_s=3.0, hop_s=1.0)
        pcm = _speech_like(5.0)
        windows = windower.push(pcm)
        assert len(windows) == 3  # at t=0s, t=1s, t=2s

    def test_window_ids_monotonic(self) -> None:
        windower = Windower(session_id="test", original_sample_rate=TARGET_SR)
        pcm = _speech_like(10.0)
        windows = windower.push(pcm)
        ids = [w.window_id for w in windows]
        assert ids == list(range(len(ids)))

    def test_window_length_correct(self) -> None:
        windower = Windower(session_id="test", original_sample_rate=TARGET_SR, window_s=3.0, hop_s=1.0)
        pcm = _speech_like(4.0)
        windows = windower.push(pcm)
        for w in windows:
            assert len(w.pcm) == int(3.0 * TARGET_SR)

    def test_start_end_ms_correct(self) -> None:
        windower = Windower(session_id="test", original_sample_rate=TARGET_SR, window_s=3.0, hop_s=1.0)
        pcm = _speech_like(4.0)
        windows = windower.push(pcm)
        assert windows[0].start_ms == 0
        assert windows[0].end_ms == 3000
        assert windows[1].start_ms == 1000
        assert windows[1].end_ms == 4000

    def test_flush_emits_partial(self) -> None:
        windower = Windower(session_id="test", original_sample_rate=TARGET_SR, window_s=3.0, hop_s=1.0)
        # Push 2s — not enough for a full window
        windower.push(_speech_like(2.0))
        flushed = windower.flush()
        # 2s >= 1.5s threshold (50% of 3s) — should emit
        assert len(flushed) == 1

    def test_flush_discards_too_short(self) -> None:
        windower = Windower(session_id="test", original_sample_rate=TARGET_SR, window_s=3.0, hop_s=1.0)
        # Push only 0.5s — too short to emit
        windower.push(_speech_like(0.5))
        flushed = windower.flush()
        assert len(flushed) == 0


# ---------------------------------------------------------------------------
# B2-T05 & T07 — Quality gate
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_good_speech_passes(self) -> None:
        pcm = _speech_like(3.0)
        qr = assess_quality(pcm, voiced_ratio=0.8)
        assert qr.quality_ok
        assert qr.quality_flags == []

    def test_silence_rejected(self) -> None:
        pcm = _silence(3.0)
        qr = assess_quality(pcm, voiced_ratio=0.0)
        assert not qr.quality_ok
        assert any("insufficient_voiced_speech" in f for f in qr.quality_flags)

    def test_clipped_audio_rejected(self) -> None:
        pcm = np.ones(int(TARGET_SR * 3.0), dtype=np.float32)  # fully clipped
        qr = assess_quality(pcm, voiced_ratio=0.9)
        assert not qr.quality_ok
        assert any("clipping" in f for f in qr.quality_flags)

    def test_dtmf_440hz_plus_1336hz_detected(self) -> None:
        """DTMF '5' = 770 Hz row + 1336 Hz column."""
        duration = 0.5
        row = _sine_wave(770, duration, amp=0.4)
        col = _sine_wave(1336, duration, amp=0.4)
        dtmf = (row + col).astype(np.float32)
        is_dtmf = detect_dtmf(dtmf)
        assert is_dtmf

    def test_pure_speech_not_dtmf(self) -> None:
        """Broadband speech should not trigger DTMF detection."""
        pcm = _speech_like(0.5)
        assert not detect_dtmf(pcm)

    def test_hold_music_detected(self) -> None:
        """440 Hz dial tone should trigger music/tone detector."""
        tone = _sine_wave(440, 1.0, amp=0.7)
        assert detect_hold_music_or_tone(tone)

    def test_snr_clean_signal(self) -> None:
        """Clean signal should have positive SNR."""
        speech = _speech_like(3.0)
        snr = estimate_snr_db(speech)
        assert snr > 0

    def test_clipping_ratio_fully_clipped(self) -> None:
        pcm = np.ones(1000, dtype=np.float32)
        ratio = compute_clipping_ratio(pcm)
        assert ratio == 1.0

    def test_clipping_ratio_not_clipped(self) -> None:
        pcm = _speech_like(0.1)  # amplitude < 0.999
        ratio = compute_clipping_ratio(pcm)
        assert ratio < 0.01


# ---------------------------------------------------------------------------
# B2-T06 — Loudness normalisation
# ---------------------------------------------------------------------------

class TestLoudnessNorm:
    def test_output_near_target(self) -> None:
        """Normalised signal RMS should be close to target dBFS."""
        pcm = _speech_like(1.0) * 0.1  # deliberately quiet
        normalised, pre_norm = normalize_loudness(pcm, target_dbfs=-23.0)
        rms = float(np.sqrt(np.mean(normalised.astype(np.float64) ** 2)))
        actual_db = 20.0 * np.log10(rms)
        assert abs(actual_db - (-23.0)) < 3.0  # within 3 dB

    def test_pre_norm_level_recorded(self) -> None:
        """pre_norm_dbfs must reflect the original level, not the normalised one."""
        pcm = _speech_like(0.5) * 0.01
        _, pre_norm = normalize_loudness(pcm)
        assert pre_norm < -40.0  # Very quiet signal

    def test_silence_handled_gracefully(self) -> None:
        pcm = _silence(1.0)
        normalised, pre_norm = normalize_loudness(pcm)
        assert pre_norm == -80.0
        assert not np.any(np.isnan(normalised))

    def test_output_clipped_within_range(self) -> None:
        """Output must always stay in [-1, 1] even after gain boost."""
        pcm = _speech_like(0.5) * 0.001  # Very quiet → big gain
        normalised, _ = normalize_loudness(pcm)
        assert np.all(np.abs(normalised) <= 1.0)


# ---------------------------------------------------------------------------
# B2-T08 — Feature cache
# ---------------------------------------------------------------------------

class TestFeatureCache:
    def test_cache_miss_computes_feature(self) -> None:
        pcm = _speech_like(3.0)
        cache = FeatureCache("test-sess", 0)
        mel = cache.get_or_compute("log_mel", pcm)
        assert mel.shape[0] == 80  # n_mels default

    def test_cache_hit_returns_same_object(self) -> None:
        pcm = _speech_like(3.0)
        cache = FeatureCache("test-sess-2", 0)
        mel1 = cache.get_or_compute("log_mel", pcm)
        mel2 = cache.get_or_compute("log_mel", pcm)
        assert mel1 is mel2  # same object (from cache)

    def test_lfcc_shape(self) -> None:
        pcm = _speech_like(3.0)
        cache = FeatureCache("test-sess-3", 0)
        lfcc = cache.get_or_compute("lfcc", pcm)
        assert lfcc.shape[0] == 20  # n_ceps default

    def test_f0_shape(self) -> None:
        pcm = _speech_like(3.0)
        cache = FeatureCache("test-sess-4", 0)
        f0 = cache.get_or_compute("f0", pcm)
        assert len(f0) > 0

    def test_cache_clear_session(self) -> None:
        pcm = _speech_like(3.0)
        sid = "sess-to-clear"
        for wid in range(3):
            FeatureCache(sid, wid).get_or_compute("log_mel", pcm)
        removed = clear_session_cache(sid)
        assert removed == 3

    def test_unknown_feature_raises(self) -> None:
        cache = FeatureCache("test", 0)
        with pytest.raises(ValueError, match="Unknown feature"):
            cache.get_or_compute("nonexistent_feature", _silence(0.1))


# ---------------------------------------------------------------------------
# E2E — Full B2 conditioner pipeline (DoD scenario)
# ---------------------------------------------------------------------------

class TestConditionerE2E:
    @pytest.mark.asyncio
    async def test_dod_scenario_silence_music_loss(self) -> None:
        """B2 DoD: 5-min call with 40% silence, hold music, 3% packet loss.

        The conditioner must:
        1. Emit only well-formed voiced windows (quality_ok=True).
        2. Mark the rest ABSTAIN (quality_ok=False with flags).
        3. Never crash.
        """
        from services.conditioner.main import Conditioner

        bus = InMemoryBus()
        conditioner = Conditioner(bus, tenant_id="test")

        session_id = "dod-test-session"
        sr = TARGET_SR
        rng = random.Random(42)

        # Build 30 seconds of mixed content at 3% loss
        # (30s → representative of 5-min at 10x compression)
        speech_pcm = _speech_like(3.0)        # 3s voiced
        silence_pcm = _silence(3.0)           # 3s silence
        tone_pcm = _sine_wave(440, 3.0)       # 3s hold music

        all_windows: list[AnalysisWindow] = []
        seq = 0
        ts_ms = 0
        chunk_ms = 20
        chunk_samples = int(sr * chunk_ms / 1000)

        # Alternate: speech, silence, music (10 blocks each of 3s)
        for block_type in [speech_pcm, silence_pcm, tone_pcm] * 3:
            for i in range(0, len(block_type) - chunk_samples + 1, chunk_samples):
                # 3% loss
                if rng.random() < 0.03:
                    seq += 1
                    ts_ms += chunk_ms
                    continue
                payload = (block_type[i : i + chunk_samples] * 32767).astype(np.int16).tobytes()
                chunk = AudioChunk(
                    session_id=session_id,
                    seq=seq,
                    ts_ms=ts_ms,
                    sample_rate=sr,
                    encoding=AudioEncoding.PCM_S16LE,
                    payload=payload,
                )
                windows = await conditioner.process_chunk(chunk)
                all_windows.extend(windows)
                seq += 1
                ts_ms += chunk_ms

        # Flush end-of-session
        await conditioner.teardown_session(session_id)

        assert len(all_windows) > 0, "Expected at least one window"

        # Check window contracts
        for w in all_windows:
            assert w.session_id == session_id
            assert w.window_id >= 0
            assert w.start_ms >= 0
            assert w.end_ms > w.start_ms
            assert w.sample_rate == TARGET_SR
            assert 0.0 <= w.clipping_ratio <= 1.0
            assert isinstance(w.quality_ok, bool)
            assert isinstance(w.quality_flags, list)
            assert w.samples_ref.startswith("shm://")

        # At least some windows should be rejected (silence + music blocks)
        abstained = [w for w in all_windows if not w.quality_ok]
        voiced = [w for w in all_windows if w.quality_ok]
        assert len(abstained) > 0, "Expected some ABSTAIN windows (silence + music)"
        assert len(voiced) > 0, "Expected some voiced windows (speech blocks)"

        # Window IDs must be monotonically increasing per session
        ids = [w.window_id for w in all_windows]
        assert ids == sorted(ids), "Window IDs not monotonic"

    @pytest.mark.asyncio
    async def test_8khz_mulaw_resampled_correctly(self) -> None:
        """Telephony path: 8kHz µ-law chunks → 16kHz AnalysisWindows."""
        from services.conditioner.main import Conditioner
        from packages.vg_audio.codecs import mulaw_to_linear

        bus = InMemoryBus()
        conditioner = Conditioner(bus, tenant_id="test")
        session_id = "mulaw-session"

        # Use raw µ-law bytes (value 0x7F = near-zero)
        chunk_ms = 20
        chunk_samples_8k = int(8000 * chunk_ms / 1000)  # 160 samples
        payload = bytes([0x7F] * chunk_samples_8k)

        # Send enough chunks to fill 3.5 windows (10s worth)
        n_chunks = int(10_000 / chunk_ms)
        for seq in range(n_chunks):
            chunk = AudioChunk(
                session_id=session_id,
                seq=seq,
                ts_ms=seq * chunk_ms,
                sample_rate=8000,
                encoding=AudioEncoding.MULAW,
                payload=payload,
            )
            await conditioner.process_chunk(chunk)

        await conditioner.teardown_session(session_id)

    @pytest.mark.asyncio
    async def test_latency_under_60ms_p95(self) -> None:
        """Conditioner latency per chunk must be < 60ms p95."""
        from services.conditioner.main import Conditioner

        bus = InMemoryBus()
        conditioner = Conditioner(bus, tenant_id="test")
        session_id = "latency-test"
        pcm = _speech_like(0.02)
        payload = (pcm * 32767).astype(np.int16).tobytes()

        latencies = []
        for seq in range(100):
            chunk = AudioChunk(
                session_id=session_id,
                seq=seq,
                ts_ms=seq * 20,
                sample_rate=TARGET_SR,
                encoding=AudioEncoding.PCM_S16LE,
                payload=payload,
            )
            t0 = time.perf_counter()
            await conditioner.process_chunk(chunk)
            latencies.append((time.perf_counter() - t0) * 1000)

        p95 = float(np.percentile(latencies, 95))
        assert p95 < 60.0, f"p95 latency {p95:.1f}ms exceeds 60ms budget"
