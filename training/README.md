# MLAI Training Guide — From Zero to Trained Models

> **Who this is for:** you have never trained a machine learning model before. You don't know what an "epoch" is. You just want two working `.tflite` files in your `models/agro/` folder so the engine can stop running in mock mode. Read this top to bottom and execute every command in order.
>
> **What you need on this PC:** Linux, Python 3.12 (already installed on most modern distros), about **8 GB of free disk space**, and an internet connection. **No GPU required** — every step in this guide finishes on a regular laptop CPU.
>
> **Time budget:** plan for an afternoon. About 30 min of setup, then mostly waiting while training runs in the background.

---

## 0. The big picture (read this once)

You are going to produce two model files:

| File | What it does | How we make it |
|---|---|---|
| `models/agro/fruit_detector.tflite` | Finds apples, bananas, and oranges in a frame and draws boxes | **Download a pretrained model** (no training!) |
| `models/agro/fruit_quality.tflite` | Labels a single fruit crop as `good` or `defective` (defective = anything not good: bruised, damaged, *or* rotten) | **Train a small classifier** on a Kaggle dataset |

### What is machine learning, in 60 seconds

Imagine teaching a child what an apple looks like by showing them many apples until they "got it". Machine learning is exactly that: you feed a program thousands of labelled images and it builds an internal *feel* for each label. The result is called a **model** — a file containing learned numbers (called *weights*).

* **Training** = the slow process of teaching the model from examples.
* **Inference** = the fast process of using the trained model to label new images.
* **TFLite** (`.tflite`) = a compact, optimised model file that runs fast on a Raspberry Pi.

AGRO uses **object detection** (show labelled fruits so the model learns to find and name them) and **image classification** (show labelled fruit crops so the model learns good vs defective).

The detector needs no training because **someone has already trained a perfectly good fruit detector** on the COCO dataset (1.5 million labelled images). We'd be wasting your laptop's CPU repeating their work.

---

## 1. Set up your training environment (≈ 15 min)

We will create an isolated Python environment so nothing we install pollutes the rest of your system. We use a plain `venv` rather than conda — it's simpler and the project's deps don't need conda's heavier machinery.

### 1.1 Make a virtual environment

Open a terminal and run:

```bash
cd ~/Desktop/MLAI-Machine-Learning
python3 -m venv .venv-train
source .venv-train/bin/activate
```

After the third command your terminal prompt should change to start with `(.venv-train)`. This means you are now "inside" the environment. You will need to repeat that `source` command every time you open a new terminal.

> **What just happened?** A virtual environment is a self-contained folder with its own copy of Python and its own set of installed packages. It lets you experiment without breaking your main system Python.

### 1.2 Install the libraries

```bash
pip install --upgrade pip
pip install -r training/requirements.txt
```

This installs TensorFlow (the machine learning library) plus small helpers for
image loading and dataset downloading, and the engine/API deps that the §4
verification tests need. It downloads roughly 600 MB and takes 3–5 minutes.

> **Why a separate file?** `training/requirements.txt` lists the PC-side
> training deps; the `requirements.txt` at the repo root lists the lighter
> Raspberry Pi runtime deps. Two files, two environments, no overlap.

### 1.3 Verify it works

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'is ready')"
```

You should see something like `TensorFlow 2.18.0 is ready`. If you instead see an error, scroll to **§7 Troubleshooting** at the bottom.

---

## 2. AGRO fruit detector — download a pretrained model (≈ 5 min)

### 2.1 Why we are not training this

Object detection (drawing a box around a fruit) needs a dataset of **labelled bounding boxes**. The Fruits-360 dataset has no boxes — it's just centred photos of single fruits on white backgrounds. Building a real bounding-box dataset would mean either annotating thousands of images by hand, or downloading a different dataset like Open Images Fruit Subset (~3 GB) and training for several hours.

There's a much easier path: Google has already trained an SSD MobileNet V1 detector on **COCO**, a dataset with 90 everyday object classes. Three of those classes are `apple`, `banana`, and `orange` — exactly the three fruits in the quality classifier dataset we use in §3. We get fruit detection for free, no training required.

> **What about tomato?** It's not in COCO. The original project plan included tomato as a target, but since the easy free model doesn't have it and we'd rather avoid training a custom detector on day one, we've dropped tomato from this guide. You can add it back later by training a custom detector if you collect a tomato dataset.

### 2.2 Download

```bash
cd ~/Desktop/MLAI-Machine-Learning/models/agro
curl -L -o coco_ssd_mobilenet_v1.zip \
  https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1.zip
mv detect.tflite fruit_detector.tflite
mv labelmap.txt fruit_detector.labels.txt
rm coco_ssd_mobilenet_v1.zip
ls -lh fruit_detector.tflite
```

You should end up with a `fruit_detector.tflite` of about **4 MB**. This file is the official TFLite team's reference object-detection model, downloaded directly from Google's CDN.

### 2.3 Update the AGRO config

The model we just downloaded has two differences from what `config/agro/config.yaml` currently expects:
- It uses **300×300** input (the config says 320×320).
- It detects 90 COCO classes; we only care about apple, banana, and orange.

Open `config/agro/config.yaml` in a text editor and change these lines:

```yaml
agro:
  detector_model: models/agro/fruit_detector.tflite
  quality_model: models/agro/fruit_quality.tflite

  detector_input_size: [300, 300]   # was [320, 320]
  quality_input_size: [224, 224]

  detection_threshold: 0.5
  nms_iou: 0.45
  max_detections: 20

  # Order matches the alphabetical order of our COCO indices below.
  fruit_classes:
    - apple
    - banana
    - orange
```

Save the file.

> **Note on class indices.** The COCO model returns class numbers (0..89). Looking at `models/agro/fruit_detector.labels.txt` (line 1 is `???` for background, so the file is effectively 0-indexed via `line_number - 1`):
>
> | Fruit | Index | Line in labelmap |
> |---|---|---|
> | banana | **52** | 53 |
> | apple | **53** | 54 |
> | orange | **55** | 56 |
>
> The engine filters detections to these three COCO indices in `engine/agro/detector.py`. If you want to change or extend the set, edit the filter list there.

---

## 3. AGRO fruit quality classifier (≈ 90 min)

### 3.1 What you're training

A small image classifier that takes a fruit crop and answers: **good or defective?** "Defective" is a catch-all here — it covers physical damage (bruises, cuts, dents) *and* biological decay (mold, rot). The model can't tell the two apart on its own; if you want that distinction, see §3.5 for the upgrade path. This is the only AGRO model we actually train ourselves, because it's both useful and easy.

We use a technique called **transfer learning**, which is the trick that makes serious ML feasible on a regular laptop. The next few paragraphs explain it properly — they're the most important conceptual paragraphs in this whole guide.

#### Transfer learning, in plain English

Imagine you want to teach someone to spot rotten apples. You have two options:

**Option A — Start from a baby.** First teach them to see. Then teach them about edges and colors. Then about shapes. Then about objects. Then about fruit. Then *finally* about apples specifically. Then about good vs rotten apples. **Years of learning, millions of examples.**

**Option B — Start from an experienced art critic.** They already know how to see, recognize colors, understand shapes, parse objects. You just say *"a rotten apple has brown spots and soft skin — here are 200 examples"*. **An afternoon.**

Transfer learning is Option B. **MobileNet V2 is the art critic.**

#### What a CNN actually learns

When you train a deep convolutional network on millions of images, something remarkable happens: **different layers learn different levels of abstraction**, automatically, with no one telling them to. Researchers visualized this and found a consistent pattern across nearly every CNN ever trained:

```
INPUT IMAGE
    │
    ▼
┌──────────────────────────────────────────┐
│ Layer 1: edges, colours, gradients       │  ← "I see a vertical edge here"
├──────────────────────────────────────────┤
│ Layer 2: corners, simple textures        │  ← "smooth area meets rough area"
├──────────────────────────────────────────┤
│ Layer 3: patterns (curves, stripes)      │  ← "this is a curved boundary"
├──────────────────────────────────────────┤
│ Layer 4: simple shapes                   │  ← "this region is roughly circular"
├──────────────────────────────────────────┤
│ Layer 5: object parts                    │  ← "stem, leaf, fruit body"
├──────────────────────────────────────────┤
│ Layer 6: whole-object concepts           │  ← "this is a fruit-like object"
├──────────────────────────────────────────┤
│ Final layer: WHICH object specifically   │  ← "Granny Smith apple, 1 of 1000"
└──────────────────────────────────────────┘
```

The crucial insight: **layers 1–6 are general**. They detect features that are useful for *any* image of *any* object — edges and textures and shapes are the same whether you're looking at a fruit, a car, or a face. **Only the final layer is task-specific** — it's the layer that says "given the features I just extracted, which of *my specific* classes is this?"

This is consistent across nearly every CNN ever trained. It's a deep property of how deep learning learns to see.

#### What MobileNet V2 actually is

MobileNet V2 is a small, fast CNN that Google trained in 2018 on **ImageNet** — a dataset of 1.4 million labelled images covering 1,000 everyday object categories (everything from "Granny Smith apple" to "loggerhead sea turtle" to "trombone"). Training took days on a cluster of GPUs and produced a ~14 MB file containing ~3.4 million learned weights.

**Google released those weights for free.** Anyone can download them in one line of Python:

```python
base = tf.keras.applications.MobileNetV2(weights="imagenet")
```

The expensive "learning to see" work is **already done**. We get the result for free.

#### Keep the seeing, replace the labeling

For our fresh-vs-rotten task we don't care about MobileNet V2's 1,000 ImageNet classes. We don't need it to know what a trombone looks like. We only want its **feature extractor** — layers 1 through 6 in the diagram above. So we do exactly two things:

1. **Cut off the final layer.** Throw away the 1000-class ImageNet classifier head.
2. **Bolt on a new tiny head** with just our 2 classes (good, defective).

Then we **freeze** the base — we tell TensorFlow *"don't change MobileNet V2's weights during training; only adjust the new tiny head we just added"*. That's the trick. We're only training **~2,500 new parameters** instead of all ~3.4 million.

In `train_quality.py`, three lines do the entire heavy lifting:

```python
include_top=False        # ← cut off the original 1000-class layer
weights="imagenet"       # ← download Google's pre-trained weights
base.trainable = False   # ← freeze the base; only the new head learns
```

#### Why this changes everything

| | Train MobileNet V2 from scratch | Transfer learning (us) |
|---|---|---|
| Parameters that learn | 3.4 million | **~2,500** |
| Training images needed | ~1 million | **~1,000 is enough** |
| Training time | Days on a GPU cluster | **~60 min on a laptop CPU** |
| Hardware needed | Multi-GPU server | **Your laptop** |
| Cost | $$$$ in cloud GPU time | **Free** |

Transfer learning is the reason "doing ML" no longer requires being a research lab. Almost every small-team ML app you see in 2026 — fruit graders, pet identifiers, plant disease detectors, defect classifiers — uses this exact pattern. **The art critic does the seeing; you teach the labeling.**

#### "But MobileNet V2 is from 2018 — isn't there something better now?"

**Short answer:** yes, there are several. But for a 2-class fruit classifier with ~1,000 images on a Pi, the difference is small enough that it's not worth complicating the project. The realistic upgrade options are:

- **`MobileNetV3Large`** — drop-in replacement, ~1–2% better accuracy
- **`EfficientNetV2B0`** — best accuracy if you don't mind a slightly bigger model
- **`EfficientNet-Lite0`** (via TF Hub) — Google's officially recommended TFLite backbone in 2026

To swap, change literally one line in `train_quality.py`:

```python
# Instead of:
base = tf.keras.applications.MobileNetV2(...)

# Use:
base = tf.keras.applications.MobileNetV3Large(...)
# or:
base = tf.keras.applications.EfficientNetV2B0(...)
```

**Why we're sticking with MobileNet V2 for this guide anyway:**

1. **The bottleneck is data, not model.** With only ~1,000–2,000 training images, no model architecture will dramatically outperform another.
2. **Battle-tested wins.** MobileNet V2 has 8 years of bug reports, tutorials, and known-working configurations.
3. **Marginal improvement at best.** Going to MobileNet V3 might give you ~1–2% better accuracy. Not worth the risk of breaking the pipeline.
4. **You can always upgrade later.** Once your end-to-end system works with MobileNet V2, swapping the backbone is a one-line change.

In professional ML, **picking the model is almost never the most important decision.** Data quality and pipeline correctness matter 10× more than backbone choice. Use MobileNet V2 for now.

### 3.2 Get the dataset

Download the Kaggle dataset **"Fruits Fresh and Rotten for Classification"** from <https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification>. It's about 1.7 GB unzipped. The dataset already comes with its own train/test split — six fruit folders nested inside both:

```
training/datasets/fruits/
├── train/
│   ├── freshapples/    rottenapples/
│   ├── freshbanana/    rottenbanana/
│   └── freshoranges/   rottenoranges/
└── test/
    ├── freshapples/    rottenapples/
    ├── freshbanana/    rottenbanana/
    └── freshoranges/   rottenoranges/
```

Unzip it into `training/datasets/fruits/` so the layout above is exact. The script in §3.3 expects this path by default.

> **Sanity check:** this command should print a number around 1700:
> ```bash
> ls training/datasets/fruits/train/freshapples/ | wc -l
> ```

### 3.3 Reorganise into the layout the trainer expects

There's a small mismatch between what the dataset gives you and what `train_quality.py` wants:

- **The Kaggle dataset** has 6 fruit-specific folders (`freshapples/`, `freshbanana/`, ..., `rottenoranges/`).
- **`train_quality.py`** wants 2 quality-specific folders (`good/`, `defective/`).

The `reorganise_quality.py` script bridges that gap. It walks the source dataset, treats every folder starting with `fresh*` as "good" and every folder starting with `rotten*` as "defective", then copies the images into the layout the trainer wants. It also **respects the dataset's existing train/test split**.

The script lives at `training/agro/reorganise_quality.py`. The defaults are already correct, so you just run it from inside `training/` with no arguments:

```bash
cd ~/Desktop/MLAI-Machine-Learning/training
source ../.venv-train/bin/activate    # if not already active
python agro/reorganise_quality.py
```

You should see something like:

```
[1/2] Copying datasets/fruits/train/  ->  datasets/agro_quality/train/
      good=5085   defective=5234
[2/2] Copying datasets/fruits/test/   ->  datasets/agro_quality/val/
      good=1693   defective=1781

Done. Reorganised dataset is at datasets/agro_quality
```

The output layout:

```
training/datasets/agro_quality/
├── train/
│   ├── good/        ← all "fresh*" images from datasets/fruits/train/
│   └── defective/   ← all "rotten*" images from datasets/fruits/train/
└── val/
    ├── good/        ← all "fresh*" images from datasets/fruits/test/
    └── defective/   ← all "rotten*" images from datasets/fruits/test/
```

### 3.4 Train the classifier

Time to actually train the model. The script `training/agro/train_quality.py` does everything in one shot — you just run it and wait.

Here's what the script actually does, in plain English:

1. **Loads your fruit photos** from `datasets/agro_quality/`.
2. **Adds variety on the fly** — random horizontal flips, slight brightness changes, slight contrast tweaks.
3. **Builds the model** using the "art critic" trick from §3.1 — the pre-trained MobileNet V2 backbone is frozen, and a tiny new head is bolted on for our 2 classes.
4. **Trains the new head** for 15 passes through the dataset. Each pass is called an "epoch".
5. **Saves a backup** of the trained model so you don't have to retrain if anything goes wrong later.
6. **Exports a `.tflite` file** that the Pi engine can load.

The defaults already point at the right folders, so just run it with no arguments:

```bash
python agro/train_quality.py
```

You'll see the dataset load, the model layout printed, and then training begins:

```
[1/4] Loading datasets from datasets/agro_quality
Found 10319 files belonging to 2 classes.
Found 3474 files belonging to 2 classes.
      classes detected: ['defective', 'good']
[2/4] Building MobileNet V2 + new classification head
[3/4] Training head for 15 epochs (base is frozen)
Epoch 1/15
645/645 ━━━ 270s 410ms/step - loss: 0.4180 - accuracy: 0.8210 - val_loss: 0.2010 - val_accuracy: 0.9230
...
Epoch 15/15
645/645 ━━━ 263s 408ms/step - loss: 0.0480 - accuracy: 0.9850 - val_loss: 0.0820 - val_accuracy: 0.9710
      saved trained classifier -> ../models/agro/fruit_quality.keras
[4/4] Converting to TFLite -> ../models/agro/fruit_quality.tflite
```

**The number to watch:** look at the `val_accuracy` column on the right. That's the only number that really matters. It tells you how well the model does on fruit photos it has *never seen* during training — the honest "exam score". **Aim for `val_accuracy > 0.90`** by epoch 15.

**How long it takes:** about **45–90 minutes** on a normal laptop CPU.

> **If something breaks during export (resumability).** The script saves a backup file at `models/agro/fruit_quality.keras` the moment training finishes — *before* the export step. So if anything goes wrong while writing the `.tflite`, you don't lose the 60 minutes of training. Just re-run the script and it'll spot the backup and skip straight to the export step (~30 seconds total).

> **If accuracy is too low.** Re-run with `--fine_tune`:
>
> ```bash
> rm ../models/agro/fruit_quality.keras    # force a fresh retrain
> python agro/train_quality.py --fine_tune --epochs 30
> ```
>
> This "unlocks" the frozen base and lets it adjust slightly to your specific fruits. Slower, but usually gains 1–3% accuracy.

When it finishes:

```bash
ls -lh ../models/agro/fruit_quality.tflite
```

You should see a file of around **3–10 MB**. Done.

### 3.5 Later: upgrading to a 3-class model

> **You can skip this on your first pass.** Come back here once §3 is working end-to-end and you want a model that tells *bruised* apart from *rotten*.

The engine can support three quality classes (good / defective / unripe) — `train_quality.py` auto-detects the number of classes from the folder structure, so adding a third class is mostly a **dataset** problem.

**What you'd need to do:**

1. **Find a third dataset** (Roboflow, Mendeley FruitNet, or your own photos).
2. **Add a third folder** to the dataset layout:
   ```
   datasets/agro_quality/
   ├── train/
   │   ├── good/
   │   ├── defective/
   │   └── damaged/
   └── val/
       ├── good/
       ├── defective/
       └── damaged/
   ```
3. **Update `config/agro/config.yaml`** so `quality_classes` lists the three folder names alphabetically.
4. **Re-run training**.

---

## 4. Verify the models exist

```bash
cd ~/Desktop/MLAI-Machine-Learning
ls -lh models/agro/
```

Expected:

```
fruit_detector.tflite        ~4 MB
fruit_detector.labels.txt    ~1 KB
fruit_quality.tflite         ~10 MB
```

Now confirm the tests still pass:

```bash
cd ~/Desktop/MLAI-Machine-Learning
source .venv-train/bin/activate
python3 -m pytest tests/ -q
```

### Optional: smoke-test the models actually load and run

There's a ready-made smoke-test script at `tests/scratch_test.py` that loads each `.tflite` file and runs a single dummy input through it:

```bash
python3 tests/scratch_test.py
```

Expected output once everything is in place:

```
Project root: /home/felipe/Desktop/MLAI-Machine-Learning
  [ok]   models/agro/fruit_detector.tflite  input=[1, 300, 300, 3] uint8  outputs=4
  [ok]   models/agro/fruit_quality.tflite  input=[1, 224, 224, 3] float32  outputs=1

Summary: 2 ok, 0 skipped, 0 failed
```

---

## 5. Publish the models (recommended) or scp them (quick)

You have two options to get the trained `.tflite` files onto the Pi.

### Option A — Publish a GitHub Release (recommended)

This is reproducible: the release is tagged and versioned, any Pi (present or future) just runs `scripts/download_models.sh` and gets exactly those bytes.

Prereqs: the [`gh` CLI](https://cli.github.com/) installed and authenticated (`gh auth login`).

```bash
# From the repo root on your PC:
TAG="v1.0.0"  # bump for each retrain
gh release create "${TAG}" \
  models/agro/fruit_detector.tflite \
  models/agro/fruit_detector.labels.txt \
  models/agro/fruit_quality.tflite \
  models/agro/fruit_quality.labels.txt \
  --title "MLAI models ${TAG}" \
  --notes "Trained models for the AGRO module."
```

Then on the Pi:

```bash
MLAI_MODELS_TAG=v1.0.0 bash scripts/download_models.sh
sudo systemctl restart mlai-api
```

If `DEFAULT_TAG` inside `scripts/download_models.sh` already points at the tag you just published, the env var is optional.

### Option B — One-off scp for experimentation

Faster if you're iterating and don't want to make a release for every try:

```bash
scp models/agro/*.tflite models/agro/*.labels.txt \
    pi@<pi-ip>:~/MLAI-Machine-Learning/models/agro/
ssh pi@<pi-ip> "sudo systemctl restart mlai-api"
```

Replace `<pi-ip>` with your Pi's IP. This bypasses the release mechanism, so a fresh reinstall later will pull whatever `DEFAULT_TAG` points to — not your local scratch build.

The full Pi setup — installing the OS, the systemd services, calibrating the camera — lives in `SetupGuide.md` at the repo root.

---

## 6. What you just accomplished

You went from an empty `models/agro/` folder to two trained `.tflite` files that the engine can load:

- **AGRO detector**: a 4 MB SSD MobileNet V1 from Google's reference TFLite models, detecting apples, bananas, and oranges.
- **AGRO quality**: a transfer-learned MobileNet V2 classifier that distinguishes fresh from rotten fruit.

The next step is to copy these files to the Raspberry Pi, install the systemd services, and run a full end-to-end test with the camera. That's covered in `SetupGuide.md`.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | You forgot to activate the venv | `source .venv-train/bin/activate` |
| `pip install tensorflow` is very slow / freezes | Old pip version | `pip install --upgrade pip` and try again |
| `Killed` during training | Out of RAM | Lower `--batch` to `4` |
| `val_accuracy` stuck at 0.5 | Class folders are empty or imbalanced | Re-run `reorganise_quality.py` and check the printed counts |
| `tflite_runtime` not found (when running engine) | The Pi runtime isn't installed on this PC | That's fine — the engine only uses it on the Pi. |

If you hit something not in this table, copy the **exact error message** and we'll debug it together.

---

## 8. Glossary (in plain English)

| Term | What it means |
|---|---|
| **Model** | A file containing learned numbers (weights). Think of it as a brain saved to disk. |
| **Training** | The slow process of showing examples to the model so it adjusts those numbers. |
| **Inference** | Using a trained model on new data. Fast. |
| **Epoch** | One full pass over your training images. |
| **Batch** | A small group of images processed together (e.g. 8 at a time). |
| **Loss** | A number telling the model how wrong it is. Lower = better. |
| **Accuracy** | Percentage of validation images the model labels correctly. |
| **Validation set** | A held-out chunk of data the model never trains on, used to measure honest performance. |
| **Transfer learning** | Starting from someone else's trained model and only retraining the last layer for your task. |
| **TFLite** | A compact file format for trained models, optimised for slow devices. |
| **COCO** | A famous dataset of 90 everyday object categories with bounding boxes. |
