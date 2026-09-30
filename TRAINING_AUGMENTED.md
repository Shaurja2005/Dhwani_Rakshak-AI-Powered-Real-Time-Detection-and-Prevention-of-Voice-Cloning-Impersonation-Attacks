# Training the augmented model (v0.2): step by step

This trains **v0.2** of the detector: the same model as the base run, but it also learns from
copies of the training audio that went through **simulated phone and app channels**. That targets
the weakness the base model showed in `docs/benchmarks/REPORT.md`: compressed and noisy audio.

Before you start, the base model must be finished. `python scripts/train_base_model.py status`
should show `"cache": true`. The clean features from that run are **reused**, not recomputed.

## What changes compared with the base model

| | Base (v0.1.0) | Augmented (v0.2.0) |
|---|---|---|
| Speech model | XLS-R 300M, frozen, layers 5–9 | same |
| Detector | Nes2Net, trained | same design, trained again from scratch (cheap) |
| Training audio | ~58.8k clean clips | the same ~58.8k clean clips **+ ~30k augmented copies** |
| Dev audio (for early stopping) | ~8k clean | ~8k clean **+ ~4k augmented** |

**What an augmented copy is:** a 4-second clip passed through a random chain, like a real call:

- **Room echo:** 20% of clips.
- **Background noise** at 5–25 dB: 50%.
- **Phone processing** (volume control, noise suppression): 20% each.
- **A codec at a random bitrate:**
  - landline: G.711;
  - mobile: AMR-NB, AMR-WB, GSM;
  - VoIP / WhatsApp: Opus, Speex, G.722;
  - voice notes: MP3, AAC.
- **Dropped packets:** 30% of clips.
- **RawBoost distortions:** 30% of clips.

Example chains from your own data:

```
asv5_T_0000004558: noise:brown+codec:amr_wb@12.65k+loss:0.03
asv5_T_0000031906: ns+resample:8000+codec:amr_nb@5.9k
```

**Fairness rule (I3):** the chain for each clip is chosen from its ID only, never from whether it
is real or fake. So real and fake clips get the same mix of channels, and the model can't learn
"compressed = fake".

## Time and space

| Step | Command | Time (estimate) | Space on E: |
|---|---|---|---|
| 1. Check | `python scripts/train_augmented_model.py check` | ~1 min | none |
| 2. Cache | `python scripts/train_augmented_model.py cache` | 20–45 min | **+~69 GB** |
| 3. Train | `python scripts/train_augmented_model.py train` | 1–3 h | tiny |
| 4. Evaluate | `python scripts/train_augmented_model.py eval` | 1–2 h | tiny |

That's about **3–6 hours** in total. The first time is the same kind of run as the base model.
Space: ~61 GB for the training copies plus ~8 GB for dev.

---

## Step 0: Open the environment (every time)

Open **Anaconda Prompt**, then type these one at a time:

```bat
conda activate E:\envs\voiceguard
cd /d E:\projects\Dhawni_Rakshak
python scripts/train_augmented_model.py status
```

`status` should say `"base_cache": true` and `next step: ... cache`.

## Step 1: Check the codecs (1 minute)

```bat
python scripts/train_augmented_model.py check
```

This augments 40 real clips and prints their chains. It should end with:

```
codecs drawn: {'codec:aac': 1, 'codec:amr_nb': 5, ... 'codec:opus': 10, ...}
augmentation speed: 8.5 clips/s on one CPU process (~68 clips/s with 8 workers)
RESULT augmentation ready
```

If it says `FAIL: ffmpeg cannot encode ...`, ffmpeg is missing or incomplete. Run
`winget install Gyan.FFmpeg`, open a **new** Anaconda Prompt, and try again.

## Step 2: Build the augmented features (20–45 minutes, GPU)

```bat
python scripts/train_augmented_model.py cache
```

This works like step 4 of the base guide, except the CPU distorts each clip before the GPU runs
XLS-R on it. Every 30 s you'll see a progress line:

```
  train_aug1: 9600/30000  52 utt/s  ETA 7 min  batch 24  gpu: NVIDIA GeForce RTX 4060 ..., peak mem 3.1/8.0 GB
```

It does `train_aug1` first, then `dev_aug1`.

**Checking the GPU:** run this in a second Anaconda Prompt:

```bat
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 5
```

This step is **CPU-heavy**, because ffmpeg runs for every clip. Seeing the GPU at 40–80% (not
100%) and the CPU busy is normal here. If the CPU is at 100% and the GPU is under 30%, the CPU is
the limit. That's fine; it just takes a little longer.

## Step 3: Train (1–3 hours, GPU)

```bat
python scripts/train_augmented_model.py train
```

Each epoch prints one line with a new field, for example:

```
{"epoch": 2, "loss": 0.21, "dev_eer": 0.034, "dev_eer_by_set": {"dev": 0.012, "dev_aug1": 0.061}, ...}
```

- `dev_eer` is clean and augmented dev combined, and it decides the early stop. **Lower is better.**
- `dev_eer_by_set` shows the two halves separately. Expect `dev_aug1` to be higher, since
  distorted audio is harder. Over the epochs you want **both** to go down or stay flat.
- Training stops by itself when `dev_eer` stops improving for 6 epochs.
- The model is saved to `runs\head_a_aug\best.pt`. Your base model in `runs\head_a_base\` is
  **not touched**.

## Step 4: Evaluate (1–2 hours)

```bat
python scripts/train_augmented_model.py eval
```

This adds a **second section** to `docs/benchmarks/REPORT.md` (`A@xlsr300m-nes2net-v0.2.0`), next to
the base model's. That gives you a before/after comparison on the same 40k ASVspoof 5 eval clips.

**How to read the comparison honestly:**
- **The rows to trust most** are the per-codec rows (`codec.codec_C01q1` ... `codec_C11q8`) and
  `overall.all`. ASVspoof 5's codecs are its own recordings, independent of our simulator.
- The **codec sweep and SNR sweep** rows use the same kind of encoders we trained with, so they'll
  look better partly for that reason. The report adds a note saying so.
- The worst base-model slices were the AI codecs (Encodec: `codec_C04*`, `codec_C07*`). Our
  simulator has no Encodec, so expect **less** improvement there.
- Clean-audio accuracy (`codec.none`) may get slightly worse. That's a common trade-off. The report
  shows whether it happened.

Or run steps 2–4 in one go, e.g. overnight:

```bat
python scripts/train_augmented_model.py all
```

## Stopping, closing the window, problems

- **Ctrl+C or closing the window is safe.** Run the same command again and it continues. The
  augmented cache saves progress every 30 s, and training resumes from the last finished epoch.
- `The clean base cache is missing or incomplete`: finish the base model first
  (`python scripts/train_base_model.py status`).
- `holds a different selection; delete it to rebuild`: you changed the `augment` or `cache`
  settings after a partial run. Delete the folder it names (e.g. `data\features\xlsr300m_L5-9\train_aug1`)
  and run the step again.
- `not enough disk space`: free space on E:, or lower `max_bona` / `spoof_per_family` under
  `train_aug1` in `ml/training/configs/head_a_aug.yaml`. Each 1,000 clips is ~2 GB.
- `out of GPU memory` in training: set `train: batch_size: 32` in the same config.

## After it finishes

- Send me the new `REPORT.md` section, or just tell me it's done, and we'll compare v0.1 with v0.2.
- **Back up** `runs\head_a_aug\best.pt` like the base model. `runs/` is not in git.
- Next: IndicSynth, one language at a time, starting from whichever model the comparison favours.
  The same augmentation settings are reused for each language.
