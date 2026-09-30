# Training on Indic languages, one at a time: step by step

This teaches the detector one Indian language at a time, using **IndicSynth** (fake voices) and
**Kathbath** (real voices of the same people). Each language starts from the previous model and
also rehearses everything learnt before, so it shouldn't forget English or earlier languages.
The eval checks that after every language.

Your settings (decided 2026-09-30):
- **12,000 fake + 12,000 real clips** per language.
- The script keeps a small replay sample, and you delete the big cache by hand.
- **Bengali first.**

All settings live in `ml/training/configs/indic_continual.yaml`.

## How it works (the short version)

For each language:

```
1 import   pick 12k fake (IndicSynth) + 12k real (Kathbath) clips spread over the whole
           language, download only those, save 16 kHz FLAC, split by speaker (70/10/20)
2 cache    XLS-R features on the GPU: clean + an augmented copy of half (codecs, noise)
3 train    fine-tune the previous model on: this language + replay of ASVspoof 5
           + replay of every earlier language
4 replay   keep a small sample (~3.8k clips, ~8 GB) of this language for the future;
           then you can delete the big cache (~60 GB)
5 eval     score the previous model AND the new model on the test sets of every language
           so far + ASVspoof 5, then show before/after per language
```

- **Order and naming:** the first language you run becomes step 01, the next step 02, and so on
  (`runs/indic/ledger.json`). Models are named `A@xlsr300m-nes2net-v0.3.1`, `v0.3.2`, … in that
  order.
- **Test data:** a speaker's real and cloned voices always stay in the same split, so the test
  set only contains **people the model never heard in training**.

## Time and space per language (estimates)

| Step | Time | Space on E: |
|---|---|---|
| 1 import | 20–60 min (depends on your internet; ~8 GB download) | +~3 GB audio (kept) |
| 2 cache | 30–60 min (GPU) | +~60 GB (deleted after step 4) |
| 3 train | 20–40 min (GPU) | tiny |
| 4 replay | ~5 min | +~8 GB (kept) |
| 5 eval | 20–40 min for the first language; grows as languages are added | tiny |

The **first** language also builds the English replay sample (~15 GB, once). Per finished
language you keep about **11 GB** (audio plus replay). All 12 languages ≈ 135 GB.

---

## Before the first language (one time)

**A. Finish and compare v0.2 first.**
- `python scripts/train_augmented_model.py status` should say all done.
- Look at the two models in `REPORT.md` (or ask me to). The first language starts from
  `init_from: runs/head_a_aug/best.pt` (v0.2). If v0.1 turns out better, change that line to
  `runs/head_a_base/best.pt`.

**B. Install the one new package.** In Anaconda Prompt:

```bat
conda activate E:\envs\voiceguard
cd /d E:\projects\Dhawni_Rakshak
pip install -r requirements/train-gpu.txt
```

This adds `pyarrow`, which reads the dataset files piece by piece.

**C. Get access to Kathbath (the real voices).** It's free, but gated:
1. Log in (or sign up) at https://huggingface.co.
2. Open https://huggingface.co/datasets/ai4bharat/Kathbath and click **Agree / Access
   repository**. Access is granted automatically.
3. Create a token at https://huggingface.co/settings/tokens → **Create new token** → type
   **Read** → copy it.
4. In Anaconda Prompt (with the environment active), run:
   ```bat
   huggingface-cli login
   ```
   Paste the token when asked (it stays invisible while you paste) and press Enter.
   Answer **n** to "Add token as git credential?".

   The token is saved in `E:\hf_cache`. **Never paste your token into a chat**, including this one.
5. Check it worked:
   ```bat
   huggingface-cli whoami
   ```
   It should print your Hugging Face username. If import still says
   `ai4bharat/Kathbath is gated`, you're logged in but haven't clicked **Agree** on the Kathbath
   page with *the same account*. Do step 2 again.

IndicSynth itself needs no login.

**D. Don't delete the ASVspoof 5 feature caches yet.** The first language's `cache` step copies
the English replay sample out of them (`data\features\xlsr300m_L5-9\replay\asv5_*`). After that
you only need them to retrain v0.1/v0.2 from scratch.

---

## Each language

Open **Anaconda Prompt** and type these one at a time:

```bat
conda activate E:\envs\voiceguard
cd /d E:\projects\Dhawni_Rakshak
python scripts/train_indic_language.py bengali status
```

### Step 1: import (download a sample)

```bat
python scripts/train_indic_language.py bengali import
```

It first plans the sample (a minute of reading file indexes), then downloads:

```
indicsynth: 690 row groups in Bengali; sampling 121 (~12100 rows, ~5400 MB to download)
kathbath: 480 row groups in bengali; sampling 245 (~12050 rows, ~2600 MB to download)
  25/366 row groups, 850 clips, 14 clips/s, ETA 25 min
```

It ends with a short report. Check these parts:

- `"train"`, `"dev"`, `"eval"`: each should have bona fide **and** spoof clips and a few speakers.
- `"original_sample_rates"`: if fake and real audio came at different rates, a **NOTE** explains
  why that matters. We'll look at it together.
- `"vc_rows_whose_source_speaker_is_in_another_split"`: an honest count of a small possible leak
  in the voice-conversion clips.

If it says **Kathbath is gated**, do step C above.

### Step 2: cache (GPU)

```bat
python scripts/train_indic_language.py bengali cache
```

This works like the augmented model's cache: four parts (`indic_bn/train`, `dev`, `train_aug1`,
`dev_aug1`) with progress lines every 30 s. The first language also builds the English replay
sample at the end (a few minutes, no GPU).

### Step 3: train (GPU)

```bat
python scripts/train_indic_language.py bengali train
```

It prints `fine-tuning from runs\head_a_aug\best.pt`, then one line per epoch:

```
{"epoch": 2, "dev_eer": 0.041, "dev_eer_by_set": {"indic_bn/dev": 0.020, "indic_bn/dev_aug1": 0.055, "replay/asv5_dev": 0.048}, "early_stop": "mean_by_set", ...}
```

`dev_eer` here is the **average** over the sets, so a model that improves Bengali while getting
worse at English is not rewarded. It stops by itself (at most 12 epochs).

### Step 4: replay

```bat
python scripts/train_indic_language.py bengali replay
```

This keeps ~3.8k Bengali clips' features for future languages. It then tells you which folder you
can delete (e.g. `data\features\xlsr300m_L5-9\indic_bn`, ~60 GB) with **Shift+Delete**. Keep:

- `data\features\xlsr300m_L5-9\replay\` (all replay samples);
- `data\indic\bn\` (the audio, needed for every future eval).

### Step 5: eval

```bat
python scripts/train_indic_language.py bengali eval
```

This scores **two** models on the same test sets: the one before this language and the new one.
Both go into `docs/benchmarks/REPORT.md` (sections tagged `indic_01`). It then prints the
comparison:

```
step 01 bengali (A@xlsr300m-nes2net-v0.2.0 -> A@xlsr300m-nes2net-v0.3.1), rows in REPORT.md:
   bn   (new)   xx.xx% ->  yy.yy%  (±d.dd%)
   en           xx.xx% ->  yy.yy%  (±d.dd%)
```

- The `bn (new)` line is the before/after for Bengali.
- The other lines show forgetting. A small rise is normal; a big one means the replay is too
  small. No pass/fail limit is set yet (that's open question Q13 for you).

Or run steps 1–5 in one go:

```bat
python scripts/train_indic_language.py bengali all
```

## The next language

Just use its name. It automatically starts from the previous language's model and includes every
earlier language's replay:

```bat
python scripts/train_indic_language.py odia all
```

The script won't train language 2 until language 1 has finished training **and** its replay is
built. You *can* run `import` for the next language while the current one trains; that's a good
way to save time.

Overview at any time:

```bat
python scripts/train_indic_language.py status
python scripts/train_indic_language.py compare
```

Names you can use: bengali, gujarati, hindi, kannada, malayalam, marathi, odia, punjabi, sanskrit,
tamil, telugu, urdu.

## All 12 languages: run commands

The script is the same for every language; only the name changes. Each line below runs all five
steps for that language (import → cache → train → replay → eval). Run them **one language at a
time, in the order you want the model to learn them**. The first language you run becomes step 01,
the next 02, and so on, and that order is fixed in `runs/indic/ledger.json`.

| # | Language | Command | IndicSynth rows (all) | Download for 12k + 12k (est.) |
|---|---|---|---:|---:|
| 1 | Bengali | `python scripts/train_indic_language.py bengali all` | 83,448 | ~7.5 GB |
| 2 | Hindi | `python scripts/train_indic_language.py hindi all` | 205,938 | ~5.5 GB |
| 3 | Marathi | `python scripts/train_indic_language.py marathi all` | 130,150 | ~6.5 GB |
| 4 | Telugu | `python scripts/train_indic_language.py telugu all` | 169,896 | ~9.5 GB |
| 5 | Tamil | `python scripts/train_indic_language.py tamil all` | 282,312 | ~9 GB |
| 6 | Gujarati | `python scripts/train_indic_language.py gujarati all` | 118,778 | ~7.5 GB |
| 7 | Urdu | `python scripts/train_indic_language.py urdu all` | 94,898 | ~7 GB |
| 8 | Kannada | `python scripts/train_indic_language.py kannada all` | 115,023 | ~10 GB |
| 9 | Odia | `python scripts/train_indic_language.py odia all` | 52,236 | ~7.5 GB |
| 10 | Malayalam | `python scripts/train_indic_language.py malayalam all` | 34,128 | ~13 GB |
| 11 | Punjabi | `python scripts/train_indic_language.py punjabi all` | 248,354 | ~7 GB |
| 12 | Sanskrit | `python scripts/train_indic_language.py sanskrit all` | 377,504 | ~9.5 GB |

- **About the order:** it's only a suggestion, roughly by how many people speak each language on
  phone calls in India. Sanskrit is last because it's rarely spoken on calls. You can pick any
  order.
- **About the download sizes:** these are estimates from the datasets' average clip size. The
  import prints the exact figure before downloading.
- **Time:** every language takes roughly **2–4 hours** end to end (import + cache + train + eval).
  Eval takes a little longer with each language, because it re-tests every language so far.

**Several languages in a row (e.g. overnight)** with `queue`. It runs them in the order given,
stops at the first problem, and skips anything already done when you re-run the same line:

```bat
python scripts/train_indic_language.py queue hindi marathi telugu
python scripts/train_indic_language.py queue tamil gujarati urdu kannada
python scripts/train_indic_language.py queue odia malayalam punjabi sanskrit
```

⚠️ With `queue`, remember the **cleanup after each language** (next section). At ~60 GB of cache
per language, three languages in a row need ~180 GB free until you delete their caches. The
replay step prints the folder to delete.

**Progress at any time:**

```bat
python scripts/train_indic_language.py status
python scripts/train_indic_language.py compare
```

## Freeing disk space

Nothing is deleted automatically. These are the folders you can remove **by hand**, and when.
Commands are for Anaconda Prompt; `del` and `rmdir /s /q` delete **permanently** (no Recycle Bin).

| What | Size | When it's safe to delete | Command |
|---|---:|---|---|
| ASVspoof 5 `.tar` archives | ~142 GB | **Now.** Everything is extracted and checksum-verified | `del /q E:\projects\Dhawni_Rakshak\data\asvspoof5\flac_*.tar` |
| ASVspoof 5 feature caches (`train`, `dev`, `train_aug1`, `dev_aug1`) | ~205 GB | **After Bengali's `cache` step has finished.** That step copies the English replay sample out of them into `replay\asv5_*` | see below |
| Each language's cache `data\features\xlsr300m_L5-9\indic_<code>` | ~60 GB each | After that language's `replay` step (it prints the exact folder) | `rmdir /s /q E:\projects\Dhawni_Rakshak\data\features\xlsr300m_L5-9\indic_bn` (for Bengali) |
| pip download cache | ~3 GB | Any time | `pip cache purge` |

ASVspoof 5 feature caches, once `replay\asv5_train` and `replay\asv5_dev` exist:

```bat
rmdir /s /q E:\projects\Dhawni_Rakshak\data\features\xlsr300m_L5-9\train
rmdir /s /q E:\projects\Dhawni_Rakshak\data\features\xlsr300m_L5-9\dev
rmdir /s /q E:\projects\Dhawni_Rakshak\data\features\xlsr300m_L5-9\train_aug1
rmdir /s /q E:\projects\Dhawni_Rakshak\data\features\xlsr300m_L5-9\dev_aug1
```

After that, retraining v0.1 or v0.2 *from scratch* would need its cache rebuilt (1–2 h GPU). The
trained models themselves (`runs\`) are not affected.

**Keep**, because they're still used:

| Folder | Why |
|---|---|
| `data\asvspoof5\extracted\` (~142 GB) | `flac_E_eval` is re-tested after every language; `flac_T` / `flac_D` are needed to ever rebuild a cache |
| `data\asvspoof5\ASVspoof5_protocols.tar`, `LICENSE.txt`, `README.txt` | label lists, licence evidence, checksums |
| `data\features\xlsr300m_L5-9\replay\` | the memory of every language learnt so far |
| `data\indic\<code>\` (~3 GB each) | each language's audio, needed for every future eval |
| `runs\` | all trained models (back up the `best.pt` files; `runs\` is not in git) |
| `E:\hf_cache` | XLS-R and your Hugging Face login |

## Stopping and problems

- **Ctrl+C or closing the window is safe in every step.** Re-run the same command.
  - Import resumes per downloaded piece.
  - Cache resumes every 30 s.
  - Train resumes from the last finished epoch.
- `pyarrow is missing`: step B.
- `... is gated`: step C.
- `was made with other targets/seed`: you changed `import: spoof/bona` after starting. Delete
  `data\indic\<code>\plan.json` and `data\indic\<code>\parts`, then import again.
- `The ASVspoof 5 feature cache ... is gone`: the English replay could not be built, because the
  base caches were deleted before the first language's cache step. Tell me, and we'll rebuild a
  small English cache.
- Internet drops during import: it retries each piece 4 times. If it still fails, just run the
  same command again.

## Things to know (honest limits)

- **Few speakers per language.** Kathbath has tens to about a hundred speakers per language, and
  IndicSynth clones a subset. The test sets therefore have a handful of speakers, so per-language
  numbers come with wide confidence intervals (shown in REPORT.md).
- **Two different sources.** All fakes come from IndicSynth and all real clips from Kathbath. If
  the two differ in recording quality or bandwidth, a model could learn "which dataset" instead of
  "fake or real". The import report shows the original sample rates, and the augmented copies put
  both through the same codecs. We'll watch for suspiciously perfect scores.
- **Licence:** IndicSynth is CC-BY-NC-4.0 (non-commercial), so every model from here on is
  **research lineage** (invariant I7). Kathbath is CC-BY-4.0.
