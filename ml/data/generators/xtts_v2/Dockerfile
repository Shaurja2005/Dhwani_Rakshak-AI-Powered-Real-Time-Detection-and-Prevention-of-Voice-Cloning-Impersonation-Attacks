# Generator: xtts_v2. One image per family — their deps conflict.
# Contract: ml/data/generators/README.md
# License: Coqui Public Model License (non-commercial) -> research lineage only.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV COQUI_TOS_AGREED=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "coqui-tts==0.24.2" "soundfile>=0.12"

# Bake weights in so the container needs no network at runtime.
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

COPY clone.py /app/clone.py
RUN useradd -m gen
USER gen
ENTRYPOINT ["python", "/app/clone.py"]
