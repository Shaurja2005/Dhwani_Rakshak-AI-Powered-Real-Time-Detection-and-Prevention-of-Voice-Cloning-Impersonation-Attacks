"""XTTS-v2 clone entrypoint implementing the clone-zoo CLI contract."""

import argparse
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch
from TTS.api import TTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["clone"])
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    wav = tts.tts(text=args.text, speaker_wav=args.ref_audio, language=args.lang)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".part.wav"
    sf.write(tmp, np.asarray(wav, dtype=np.float32), tts.synthesizer.output_sample_rate)
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
