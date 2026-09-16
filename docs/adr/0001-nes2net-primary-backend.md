# ADR 0001 — Nes2Net as the primary anti-spoof back-end

**Status:** Accepted (implementation in B4; to be re-validated by B15 numbers after training)

## Context

Head A pairs a large SSL front-end (XLS-R-300M / WavLM-large, 1024-dim hidden
states) with a light back-end. Most back-ends (AASIST, SLS, RawNet2-style)
first project the 1024-dim features down to a small dimension, which is both a
compute cost and an information bottleneck for the subtle artifacts that
intermediate SSL layers carry. The target hardware is modest (an 8 GB laptop
GPU for development; T4/L4 or CPU in deployment), so back-end compute matters.

## Decision

Use **Nes2Net** (nested Res2Net blocks operating directly on the full SSL
feature dimension, followed by attentive statistics pooling) as the primary
back-end, with a learned weighted sum over SSL layers 5–9 in front of it.
Keep an **AASIST-style graph-attention back-end** as the secondary / ensemble
option. Both are selectable per config (`backend: nes2net | aasist`).

## Alternatives considered

- **AASIST as primary** — strong, widely reproduced baseline, but relies on a
  projection bottleneck and heavier graph layers. Kept as secondary.
- **SLS classifier** — simple and effective with layer-wise features, but less
  expressive over time.
- **Plain linear / MLP head** — cheapest, clearly weaker on unseen generators.

## Consequences

- Our Nes2Net and AASIST implementations (`packages/vg_models/heads/head_a_ssl/backends.py`)
  are re-implementations, not ports. **B4-T01 must verify them against published
  results** before any number is quoted; if parity fails, port the reference code.
- Checkpoints store only back-end + layer weights when the front-end is frozen,
  so they stay small and front-end weights are pinned separately in
  `models/registry.yaml`.
- The ensemble option (Nes2Net + AASIST, or XLS-R + WavLM) is deferred to fusion (B9).

## Blocks affected

B4 (implementation), B9 (ensemble fusion), B14 (export/distillation), B15 (validation).
