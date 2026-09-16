# Clone zoo — generator container contract (B3-T05)

One Docker image per generator (`vg-gen-<name>`), because their Python
dependencies conflict. The list of generators, families and licenses lives in
`ml/data/generators.yaml`.

## CLI contract

Every image's entrypoint must accept:

```
clone --ref-audio <path> [--text <str> --lang <iso639-1>] [--source-audio <path>] \
      --out <path.wav> --seed <int>
```

- TTS generators use `--text` and `--lang`; voice-conversion generators use `--source-audio`.
- Write a mono WAV at the model's native sample rate. Do not add any channel degradation.
- Be deterministic for a given `--seed`.
- Exit non-zero on failure; never write a partial file at `--out`.
- No network access at runtime: bake weights into the image or mount them read-only.

## Build

```bash
docker build -t vg-gen-xtts_v2 ml/data/generators/xtts_v2
```

## Status

| Generator | Image implemented |
|---|---|
| xtts_v2 | yes (reference implementation) |
| all others | Dockerfile placeholder only |
