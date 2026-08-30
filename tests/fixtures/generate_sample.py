"""Generate a short silence WAV for replay tests.

Run once: python tests/fixtures/generate_sample.py
"""
from __future__ import annotations

import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
DURATION_S = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION_S
OUT_PATH = Path(__file__).parent / "sample.wav"


def main() -> None:
    with wave.open(str(OUT_PATH), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        # 5 seconds of silence (zeros)
        data = struct.pack("<" + "h" * NUM_SAMPLES, *([0] * NUM_SAMPLES))
        wf.writeframes(data)
    print(f"Written: {OUT_PATH}")


if __name__ == "__main__":
    main()
