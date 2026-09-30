# Training the base model: quick start

The short version of the first training run. For every detail (expected output of each step,
troubleshooting, VS Code setup) see **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**.

## What is being trained

**Speech model: [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m)**
(XLS-R, 300M parameters). Meta pretrained it on about 436k hours of speech in 128 languages, so it
already knows Indic languages. That matters when IndicSynth is added later.

The first run is **not a full fine-tune**. XLS-R stays frozen, and a small detector learns on top of
it. Config: [`ml/training/configs/head_a_base.yaml`](ml/training/configs/head_a_base.yaml).

| Part | Size | Trained on ASVspoof 5? |
|---|---|---|
| XLS-R 300M, only layers 5–9 are used (the upper 14 are cut off) | ~300M params | **No, frozen** (`freeze_frontend: true`) |
| Nes2Net back-end + OC-Softmax classifier | a few M params | **Yes** |

How it works: XLS-R runs over every clip **once** and its layer 5–9 outputs are saved to E: (the
"feature cache"). The detector then trains for up to 30 epochs on those saved features. That's
only possible because XLS-R doesn't change during training.

Why frozen first, on an RTX 4060 with 8 GB:
- **Fits in memory.** Fine-tuning XLS-R needs its gradients and optimizer state on the GPU too. On
  8 GB that forces tiny batches and gradient checkpointing.
- **Hours, not days.** Fine-tuning would need a full GPU pass over the audio every epoch.
- **Less overfitting.** A frozen backbone is less likely to latch onto ASVspoof 5's specific attacks.
- **Easier to add languages.** Indic languages can be added one at a time without the model
  forgetting earlier ones.

**Stage 2 (later, optional):** start from the base checkpoint and unfreeze only the top layers used
(8–9, or LoRA adapters) with a low learning rate, training on raw audio. Whether it's worth it is
decided by comparing both runs in `docs/benchmarks/REPORT.md`, not by guessing. Numbers from papers
don't count as evidence for this system.

## The plan

| Step | Command (run in order) | Time (estimate) | Space on E: |
|---|---|---|---|
| 1. Set up | the commands in "Step 1" below | 15–30 min | ~10 GB |
| 2. Unpack | `python scripts/train_base_model.py extract` | 20–40 min | +135 GB |
| 3. Label lists | `python scripts/train_base_model.py manifests` | 5–10 min | tiny |
| 4. GPU pass | `python scripts/train_base_model.py cache` | 1–2 h | +135 GB |
| 5. Train | `python scripts/train_base_model.py train` | 30–60 min | tiny |
| 6. Evaluate | `python scripts/train_base_model.py eval` | 1–2 h | tiny |

That's about **4–6 hours** in total, using about **280 GB of E:** (~750 GB free). C: and D: are not
touched. The times are estimates. The benchmark in step 1 prints your laptop's real speed.

## Step 1: set up (one time)

Open **Anaconda Prompt** (Start menu). Type these **one line at a time**:

```bat
conda create -p E:\envs\voiceguard python=3.11 -y
conda activate E:\envs\voiceguard
conda env config vars set HF_HOME=E:\hf_cache PIP_CACHE_DIR=E:\pip_cache
conda deactivate
conda activate E:\envs\voiceguard
cd /d E:\projects\Dhawni_Rakshak
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/train-gpu.txt
python scripts/setup/check_gpu.py --bench
```

It should end with `RESULT ready` and a line showing your **RTX 4060**. The benchmark prints the best
batch size. If it's 16 rather than 24, set `cache: batch: 16` in `ml/training/configs/head_a_base.yaml`.

## Steps 2–6: train

Each time you open a new Anaconda Prompt, first run:

```bat
conda activate E:\envs\voiceguard
cd /d E:\projects\Dhawni_Rakshak
```

Then run steps 2–6 from the table one at a time, or all of them in one go (for example overnight):

```bat
python scripts/train_base_model.py all
```

To see what's done and what to run next at any time:

```bat
python scripts/train_base_model.py status
```

## Making sure the GPU is used

The PyTorch installed before this guide was the **CPU-only** build, so the GPU was never used. The
`cu124` install in step 1 fixes that. The scripts also:
- use half precision (fp16);
- run only the 10 of 24 XLS-R layers the detector needs (the output is identical);
- decode audio in 6 parallel CPU processes, so the GPU isn't kept waiting;
- halve the batch size automatically if the GPU runs out of memory;
- stop Windows from throttling or sleeping the job **while it runs** (nothing changes permanently).

**To see it working:** the reliable check is `nvidia-smi`. In a second Anaconda Prompt, while
step 4 or 5 runs:

```bat
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 5
```

`utilization.gpu` should be mostly 70–100% and `memory.used` a few GB. Press Ctrl+C to stop it. Every
30 s the training script also prints a line with clips/s, the ETA and GPU memory.

Task Manager usually has **no "Cuda" graph** on Windows 11 ("Hardware-accelerated GPU scheduling" hides
it). That's normal, and there's nothing to change. Instead, watch the **3D** graph and **Dedicated
GPU memory** under Performance → GPU (NVIDIA). Or, in the **Details** tab, add the **GPU** and
**GPU engine** columns (right-click a column header → Select columns) and look at `python.exe`.

If GPU utilization stays under ~50% while the CPU is at 100%, raise `cache: workers` from 6 to 8 in
the config.

## Why not WSL?

WSL2 works with the GPU too, but the dataset is on E:. Reading hundreds of thousands of small files
through `/mnt/e` is several times slower than from Windows directly. Native Windows with conda is the
faster and simpler option here.

## Laptop safety

- Keep the charger in and the laptop on a hard, flat surface (a cooling pad helps).
- 75–85 °C under load is normal. The GPU slows itself down before it overheats.
- Don't close the terminal window while a step runs. Minimising it is fine.
- **Ctrl+C at any time is safe.** Running the same command again continues where it stopped.

## When it finishes

- Trained model: `runs/head_a_base/best.pt`
- Official results: `docs/benchmarks/REPORT.md`. Every number there has a row id, and those are the
  only numbers to quote.
- Optional: register and promote the model (step 7 in [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)).

**Licence note (Q12 in `PROJECT_STATUS.md`):** ASVspoof 5's own licence files say ODC-By 1.0 and
CC BY 4.0, while `ml/data/registry.yaml` still says "non-commercial EULA". A human needs to confirm
before the registry is changed. Training isn't blocked either way.

**Next:** the augmented model v0.2 (phone and app channels, codecs, noise), in
**[TRAINING_AUGMENTED.md](TRAINING_AUGMENTED.md)**. After that come the Indic languages, one at a
time: IndicSynth (streamed, sampled) plus Kathbath for genuine speech, then continual fine-tuning
with a replay buffer, then re-evaluation of every language.
