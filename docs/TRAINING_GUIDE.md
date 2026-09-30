# Training the base model: a step-by-step guide

This trains **Head A**, the synthetic-voice detector, on **ASVspoof 5**. Follow the steps in order.
Every step can be stopped with **Ctrl+C** and resumed by running the same command again.

**What is trained:** frozen `facebook/wav2vec2-xls-r-300m` (layers 5–9) plus a Nes2Net detector
that learns on top of it. XLS-R itself is not fine-tuned in this first run. The reasons and the
optional stage-2 fine-tune are in [`TRAINING_QUICKSTART.md`](../TRAINING_QUICKSTART.md).

**What you will end up with:** `runs/head_a_base/best.pt` (the trained model) and new rows in
`docs/benchmarks/REPORT.md` that say how good it is.

## How long it takes, and how much space it uses

Times are estimates for an RTX 4060 laptop. Step 1 prints real speed measurements for your GPU.

| Step | What happens | Time (estimate) | Disk on E: |
|---|---|---|---|
| 1. Environment | Install GPU PyTorch; check the GPU | 15–30 min (downloads ~5 GB) | ~10 GB |
| 2. Extract | Unpack the ASVspoof 5 tars | 20–40 min | +135 GB |
| 3. Manifests | Turn the label files into lists the code reads | 5–10 min | tiny |
| 4. Cache | Run XLS-R once over 66k clips on the GPU | 1–2 h | +135 GB |
| 5. Train | Train the small detector on top | 30–60 min | tiny |
| 6. Evaluate | Measure it the official way → REPORT.md | 1–2 h | tiny |

In total that's about **4–6 hours**, mostly unattended, using about **280 GB** of E: (you have ~750 GB free).
C: and D: are not used. Everything goes on E:.

## Before you start (laptop care)

- **Plug the charger in** and put the laptop on a hard, flat surface (a cooling pad helps).
- Choose "Best performance" or your laptop maker's "Performance" mode. This is optional; "Balanced" is quieter but slower.
- Long runs are safe. The GPU slows itself down before it gets too hot, and 75–85 °C under load is normal.
  The scripts also stop Windows from sleeping or slowing the job **while they run**; nothing
  is changed permanently.
- Don't close the terminal window while a step runs. Minimising it is fine.

---

## Step 1: Create the training environment (one time)

Open **Anaconda Prompt** from the Start menu (not plain PowerShell). Type these one at a time and
press Enter after each. `(E:\envs\voiceguard)` should appear at the start of the line after the
`activate` command.

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

What these do:
- The environment lives on **E:** (`E:\envs\voiceguard`), so C: doesn't fill up. The
  `config vars` line keeps the model and pip downloads on E: too.
- `cu124` is the **GPU build** of PyTorch. The PyTorch you have now is the CPU-only build, which is why
  the GPU wasn't used.
- The last command checks everything. It downloads XLS-R once (~1.3 GB) and measures your GPU.

**You want to see:**

```
torch       2.6.0+cu124  cuda build: 12.4
gpu         NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB ...
transformers 4.x OK
bench       batch 24:  ... clips/s ...
RESULT      ready
```

Note the `bench` line with the highest clips/s that did not say "out of memory". If it's batch 16
rather than 24, open `ml/training/configs/head_a_base.yaml` and set `cache: batch: 16`.

### Using VS Code instead of Anaconda Prompt (optional)

VS Code → `Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** →
`E:\envs\voiceguard\python.exe`. New terminals in VS Code then use this environment automatically.
Run the commands below in **Terminal → New Terminal**. Use Jupyter only for looking at data,
not for training.

---

## Step 2: Unpack the dataset

```bat
python scripts/train_base_model.py extract
```

This checks each tar against the checksum in the dataset's own README, then unpacks it into
`data\asvspoof5\extracted\`. You'll see lines like `[3/18] flac_T_ac.tar: md5 ok`, then
file counts. The original `.tar` files are kept. Once everything works, you may delete them to
free 135 GB, but it's not necessary.

## Step 3: Build the manifests

```bat
python scripts/train_base_model.py manifests
```

This should end with three lines like `train: wrote 182357 rows (... bona fide, ... spoof, attacks A01..A08)`.
If it says `audio files MISSING`, step 2 did not finish: run step 2 again.

## Step 4: Compute the XLS-R features (the long GPU step)

```bat
python scripts/train_base_model.py cache
```

Every 30 seconds it prints a progress line:

```
  train: 12480/58400  95 utt/s  ETA 8 min  batch 24  gpu: NVIDIA GeForce RTX 4060 ..., peak mem 3.1/8.0 GB
```

**Checking that the GPU is really working:** in a *second* Anaconda Prompt, run

```bat
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 5
```

`utilization.gpu` should be mostly 70–100%. Press Ctrl+C to stop it. Task Manager on Windows 11
usually has **no "Cuda" graph**, because "Hardware-accelerated GPU scheduling" hides it. That's
normal. Watch the **3D** graph and **Dedicated GPU memory** there instead, or add the **GPU engine**
column in the Details tab and look at `python.exe`.

If utilisation stays low (under ~50%) while the CPU is at 100%, the CPU can't decode audio fast
enough. Raise `cache: workers` from 6 to 8 in the config. If you get "out of GPU memory", the batch size
halves automatically, so there's nothing to do.

## Step 5: Train the detector

```bat
python scripts/train_base_model.py train
```

Each epoch prints one line, for example:

```
{"epoch": 3, "loss": 0.41, "dev_eer": 0.0712, "seconds": 84.2, "gpu": "...", ...}
```

- `dev_eer` is the error rate on the development set. **Lower is better.** It should drop over the
  first epochs, then level off. Training stops by itself when it stops improving ("early stop").
- The best model so far is always saved as `runs\head_a_base\best.pt`. If you stop and re-run, it
  continues from the last finished epoch.
- These dev numbers are **not** the official result. Step 6 produces the official result.

## Step 6: Evaluate the official way

```bat
python scripts/train_base_model.py eval
```

This scores a stratified sample of 40,000 ASVspoof 5 **eval** clips, which the model has never
seen. It then runs the codec and noise tests and the attack tests, and writes the results to
**`docs/benchmarks/REPORT.md`**. Every number there has a row id. Those ids are what you cite in
slides and reports. Don't quote numbers from anywhere else.

Adding ASVspoof 2019 as a second, independent test set (recommended):
1. Download it from Kaggle and unzip it somewhere on **E:**, e.g. `E:\datasets\asvspoof2019`.
2. `python -m ml.data.importers.asvspoof2019 --root E:\datasets\asvspoof2019\LA`
   (use the folder that contains `ASVspoof2019_LA_cm_protocols`).
3. In `ml/eval/configs/base_asvspoof5.yaml`, remove the `#` from the four `asvspoof2019_la_eval` lines.
4. Run step 6 again.

## Step 7: Put the model to use (optional)

```bat
python scripts/model_rollout.py register --version A@xlsr300m-nes2net-v0.1.0 --kind head_a --path runs/head_a_base/best.pt --lineage research
python scripts/model_rollout.py attach-eval --version A@xlsr300m-nes2net-v0.1.0 --run-id <run id from REPORT.md>
python scripts/model_rollout.py promote --version A@xlsr300m-nes2net-v0.1.0 --actor <your name>
```

After this, the gateway (`VG_INFERENCE_BACKEND=registry`) and `scripts/demo_scenarios.py` use the
trained model instead of abstaining with "untrained".

---

## Checking progress at any time

```bat
python scripts/train_base_model.py status
```

This shows which steps are done and the next command to run. `python scripts/train_base_model.py all`
runs steps 2–6 in order and skips finished ones, which is good for starting before bed.

## If something goes wrong

| You see | Do this |
|---|---|
| `gpu NOT VISIBLE` in step 1 | The CPU build of torch is installed. Repeat the `pip install torch ... cu124` line inside the activated environment. |
| `transformers BROKEN` | `pip install -r requirements/train-gpu.txt` inside the activated environment. |
| `md5 MISMATCH` for a tar | That tar is damaged. Re-download that one file; the others are fine. |
| `not enough disk space` | Free space on E:, or lower `spoof_per_family` in the config (each 1000 saves ~16 GB). |
| `feature cache incomplete` | Step 4 was interrupted. Run step 4 again; it continues. |
| `out of GPU memory` during training | Set `train: batch_size: 32` in the config. |
| Everything is very slow | Keep the terminal in front, keep the charger in, and use a Performance power mode. |
| The laptop is very hot or loud | Normal under full load. Pause with Ctrl+C whenever you like; nothing is lost. |

## Why not WSL?

WSL2 (your Ubuntu) can use the GPU too, but the dataset lives on E:. From WSL, reading hundreds of
thousands of files through `/mnt/e` is several times slower than from Windows directly. Native Windows
with the conda environment is the faster and simpler option here. WSL is only worth it later if we
use Linux-only tools.

## What comes next

After the base model: **Indic languages, one at a time.** IndicSynth (streamed, sampled) plus Kathbath for
genuine speech in the same language, then a continual fine-tune that keeps a replay buffer of earlier
data, then re-evaluation of every language so far. A separate guide will follow.
