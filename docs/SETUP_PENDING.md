# Deferred setup checklist

The project is being built **structure first**: every block's code and tests
land before any data download, model fetch, or training. This file collects
every setup step that was deferred, block by block, so it can all be done in
one pass at the end.

Current data position (2026-09-17): **ASVspoof 5 on hand (~132 GB)**;
**IndicSynth planned** for all 12 languages if the dataset permits. Both are
non-commercial → the primary model is **research lineage**.

## Environment

- [ ] Install a CUDA build of PyTorch. The dev machine has an RTX 4060 Laptop
      (8 GB) but the installed torch is CPU-only (`2.6.0+cpu`).
- [ ] Fix `transformers`: it fails to import because `huggingface-hub==1.29.0`
      is installed and transformers requires `<1.0`. Pin a compatible pair.
- [ ] `huggingface-cli login`.

## B3 — Data & corpus

- [ ] Write an ASVspoof 5 manifest (`data/manifests/asvspoof5.jsonl`) from its
      protocol files; place audio under `data/raw/asvspoof5/`.
- [ ] Request IndicSynth access (gated HF); confirm which of the 12 languages exist.
- [ ] Verify licenses; set `license_checked_by` in `ml/data/registry.yaml`.
- [ ] Pin sha256 for downloaded files.
- [ ] Add consented corpora / speakers to `ml/data/consent_register.yaml`.
- [ ] Build the remaining 12 clone-zoo images; build + test `vg-gen-xtts_v2`.
- [ ] Run `clone_job --execute`, `degrade`, then `make corpus-report` (symmetry must PASS).
- [ ] Generate splits: `python -m ml.data.splits --holdout-families <...>`.

## B4 — Head A

- [ ] Download XLS-R-300M (+ optionally WavLM-large); pin `revision` in `models/registry.yaml`.
- [ ] **B4-T01:** reproduce a published ASVspoof 2019 LA baseline (needs that
      dataset, ~7.6 GB public) — validates our Nes2Net/AASIST re-implementations (ADR 0001).
- [ ] Stage 1: `python -m ml.training.train_head_a --config ml/training/configs/head_a_stage1.yaml`
      (frozen front-end, batch 8 fits 8 GB).
- [ ] Stage 2 with IndicSynth (`head_a_stage2.yaml`), Stage 3 robustness (`head_a_stage3.yaml`).
- [ ] Measure Head A p95 latency on target hardware; set `VG_HEAD_A_BUDGET_MS`.
- [ ] Run with `VG_HEAD_A_CHECKPOINT=runs/<run>/best.pt` and confirm scores appear in replay.

## B5 — Head B (DSP + scene)

- [ ] Build a manifest of real vocoder / neural-codec outputs (HiFi-GAN, BigVGAN,
      EnCodec) and check `vocoder.py` detectors fire on them (B5-T02 validation).
- [ ] Train: `python -m ml.training.train_head_b --config ml/training/configs/head_b.yaml` (CPU).
- [ ] Measure p95 latency per window on target hardware (budget 15 ms).
- [ ] B9 fusion ablation: fused EER with Head B vs Head A alone (block DoD).

## B6 — Head C (prosody)

- [ ] Train: `python -m ml.training.train_head_c --config ml/training/configs/head_c.yaml` (CPU).
      The run fails if the per-language bona fide FPR gap exceeds 5 points; fix
      data balance / normalisation rather than loosening the gate.
- [ ] Per-language evaluation across all 12 languages on a codec-degraded set (B15).
- [ ] Wire ASR tokens from B10 into `analyse(..., tokens=...)` once B10 exists.
- [ ] Set `language_hint` in CallMetadata from the tenant/IVR language, or from B10 language ID.

## B7 — Head D (speaker verification)

- [ ] Install one or more embedders: `pip install speechbrain` (ECAPA, default),
      `onnxruntime` + a WeSpeaker ResNet34 ONNX export, or NVIDIA NeMo (TitaNet-L).
- [ ] Build a speaker benchmark manifest of Indian-accented + Indic bona fide speech
      (several utterances per speaker) and run
      `python -m packages.vg_models.heads.head_d_speaker.benchmark --manifest <m> --embedders ecapa,wespeaker,titanet`;
      pick the embedder with the best per-language EER (B7-T01).
- [ ] Build an AS-norm cohort (~1-5k impostor embeddings from public corpora, not
      enrolled customers) and store it with `VoiceprintVault.set_cohort(<embedder>, ...)`.
- [ ] Fit calibration on target / non-target trials (`scoring.fit_calibration`) and
      pass it to `HeadD(calibration=...)`.
- [ ] Generate per-tenant vault keys: `VG_VAULT_KEY_<TENANT>` = base64 of 32 random bytes
      (dev only; production keys from KMS/HSM, B16-T07).
- [ ] Demo DoD: enrol a consenting volunteer ("CEO"), clone them (consent register!),
      score genuine vs clone windows and report `scoring.gap_with_ci`.
