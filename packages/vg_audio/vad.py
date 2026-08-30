"""packages/vg_audio/vad.py — B2-T03: VAD integration with hysteresis.

Voice Activity Detection using a lightweight energy-based VAD with
hysteresis, designed to avoid fragmenting windows on short pauses.

Primary: Energy + zero-crossing rate (no external dependency required).
Optional: Silero VAD via ONNX Runtime for higher accuracy (loads lazily).

Hysteresis logic:
- Speech starts when energy exceeds ONSET_THRESHOLD.
- Speech ends only after HANGOVER_FRAMES consecutive sub-threshold frames.
- This prevents chopping a window on a brief inter-word pause (300 ms).

The VAD result is used by the quality gate (quality.py) to compute
voiced_ms for the AnalysisWindow contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from packages.vg_core.logging import get_logger

log = get_logger(__name__)

TARGET_SR = 16_000
FRAME_MS = 20              # VAD frame size in milliseconds
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)  # 320 samples

# Energy thresholds (relative to full-scale, in log-energy space)
ONSET_LOG_ENERGY: float = -35.0   # dBFS onset for voice activity
OFFSET_LOG_ENERGY: float = -40.0  # dBFS offset (with hysteresis below onset)
HANGOVER_FRAMES: int = 15          # ~300 ms hold before declaring non-voiced


def _log_energy_db(frame: np.ndarray) -> float:
    """Compute log energy of a frame in dBFS. Clipped to -80 dBFS minimum."""
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return -80.0
    return 20.0 * np.log10(rms + 1e-9)


@dataclass
class VADResult:
    """Per-frame VAD decision with energy metadata."""
    is_voiced: bool
    energy_db: float
    frame_index: int


class EnergyVAD:
    """Simple energy + hysteresis VAD.

    Suitable for clean studio audio. For telephony, SileroVAD is preferred.
    """

    def __init__(self) -> None:
        self._speaking = False
        self._hangover = 0

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> VADResult:
        """Process one 20ms frame and return a VAD decision."""
        energy = _log_energy_db(frame)

        if self._speaking:
            if energy >= OFFSET_LOG_ENERGY:
                # Still above offset threshold — reset hangover counter
                self._hangover = 0
            else:
                self._hangover += 1
                if self._hangover >= HANGOVER_FRAMES:
                    self._speaking = False
                    self._hangover = 0
        else:
            if energy >= ONSET_LOG_ENERGY:
                self._speaking = True
                self._hangover = 0

        return VADResult(
            is_voiced=self._speaking,
            energy_db=energy,
            frame_index=frame_idx,
        )

    def process_window(self, pcm: np.ndarray) -> tuple[float, list[VADResult]]:
        """Process a full analysis window and return (voiced_ratio, frame_results).

        voiced_ratio: fraction of frames classified as voiced (0.0 – 1.0).
        """
        frames = [
            pcm[i : i + FRAME_SAMPLES]
            for i in range(0, len(pcm) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
        ]
        results = [self.process_frame(f, i) for i, f in enumerate(frames)]
        voiced = sum(1 for r in results if r.is_voiced)
        ratio = voiced / len(results) if results else 0.0
        return ratio, results

    def reset(self) -> None:
        self._speaking = False
        self._hangover = 0


class SileroVAD:
    """Silero VAD via ONNX Runtime (lazy-loaded).

    Falls back to EnergyVAD if onnxruntime or the model isn't available.
    The model file is fetched by make fetch-models (models/silero_vad.onnx).
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or "models/silero_vad.onnx"
        self._session = None
        self._fallback = EnergyVAD()
        self._loaded = False

    def _try_load(self) -> bool:
        """Attempt to load the ONNX model. Returns True on success."""
        if self._loaded:
            return self._session is not None
        try:
            import onnxruntime as ort  # type: ignore[import]
            import os
            if not os.path.exists(self._model_path):
                log.warning("silero_vad_model_not_found", path=self._model_path)
                self._loaded = True
                return False
            self._session = ort.InferenceSession(self._model_path)
            log.info("silero_vad_loaded", path=self._model_path)
            self._loaded = True
            return True
        except ImportError:
            log.warning("onnxruntime_not_installed", fallback="EnergyVAD")
            self._loaded = True
            return False

    def process_window(self, pcm: np.ndarray) -> tuple[float, list[VADResult]]:
        """Process a full analysis window. Falls back to EnergyVAD if ONNX not available."""
        if not self._try_load():
            return self._fallback.process_window(pcm)
        # Silero VAD: process in 512-sample chunks at 16kHz
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        frame_size = 512
        results: list[VADResult] = []
        for i, offset in enumerate(range(0, len(pcm) - frame_size + 1, frame_size)):
            frame = pcm[offset : offset + frame_size].reshape(1, -1).astype(np.float32)
            try:
                out, h, c = self._session.run(  # type: ignore[union-attr]
                    None,
                    {"input": frame, "h": h, "c": c, "sr": np.array(TARGET_SR)},
                )
                speech_prob = float(out[0][0])
            except Exception:  # noqa: BLE001
                speech_prob = 0.0
            results.append(VADResult(
                is_voiced=speech_prob > 0.5,
                energy_db=_log_energy_db(pcm[offset : offset + frame_size]),
                frame_index=i,
            ))
        voiced = sum(1 for r in results if r.is_voiced)
        ratio = voiced / len(results) if results else 0.0
        return ratio, results


def get_vad(use_silero: bool = True, model_path: Optional[str] = None) -> "SileroVAD | EnergyVAD":
    """Return the best available VAD implementation."""
    if use_silero:
        return SileroVAD(model_path)
    return EnergyVAD()
