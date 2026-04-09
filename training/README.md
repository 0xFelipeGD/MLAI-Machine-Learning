# MLAI Training Guide — From Zero to Trained Models

> **Who this is for:** you have never trained a machine learning model before. You don't know what an "epoch" is. You just want three working `.tflite` files in your `models/` folder so the engine can stop running in mock mode. Read this top to bottom and execute every command in order.
>
> **What you need on this PC:** Linux, Python 3.12 (already installed on most modern distros), about **8 GB of free disk space**, and an internet connection. **No GPU required** — every step in this guide finishes on a regular laptop CPU.
>
> **Time budget:** plan for an afternoon. About 30 min of setup, then mostly waiting while training runs in the background.

---

## 0. The big picture (read this once)

You are going to produce three model files:

| File | What it does | How we make it |
|---|---|---|
| `models/indust/padim_toothbrush.tflite` | Looks at a toothbrush head and decides "good" or "defective" | **Train an autoencoder** on photos of good toothbrushes |
| `models/agro/fruit_detector.tflite` | Finds apples, bananas, and oranges in a frame and draws boxes | **Download a pretrained model** (no training!) |
| `models/agro/fruit_quality.tflite` | Labels a single fruit crop as `good` or `defective` (defective = anything not good: bruised, damaged, *or* rotten) | **Train a small classifier** on a Kaggle dataset |

### What is machine learning, in 60 seconds

Imagine teaching a child what an apple looks like by showing them many apples until they "got it". Machine learning is exactly that: you feed a program thousands of labelled images and it builds an internal *feel* for each label. The result is called a **model** — a file containing learned numbers (called *weights*).

* **Training** = the slow process of teaching the model from examples.
* **Inference** = the fast process of using the trained model to label new images.
* **TFLite** (`.tflite`) = a compact, optimised model file that runs fast on a Raspberry Pi.

INDUST uses **anomaly detection**: we only show the model "good" parts and let it flag anything that doesn't match. AGRO uses **object detection**: we show labelled fruits so the model learns to find and name them.

The reason only one of the three models needs from-scratch training is that **someone has already trained a perfectly good fruit detector** on the COCO dataset (1.5 million labelled images). We'd be wasting your laptop's CPU repeating their work.

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
# Training-time dependencies (TensorFlow + helpers + pytest for §5 verification)
pip install "tensorflow==2.18.*" opencv-python pillow numpy tqdm requests pytest
# Engine + API dependencies (so the tests in §5 can import the engine modules
# and exercise the FastAPI app — these are the same versions the Pi uses,
# pulled from the project's root requirements.txt). httpx is required by
# fastapi.testclient.TestClient and is NOT pulled in automatically by fastapi.
pip install "fastapi==0.115.*" "uvicorn[standard]==0.32.*" "pydantic==2.9.*" pyyaml psutil httpx
```

This downloads roughly 600 MB and takes 3–5 minutes. TensorFlow is the machine learning library; the rest are small helpers for image loading and downloading.

### 1.3 Verify it works

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'is ready')"
```

You should see something like `TensorFlow 2.18.0 is ready`. If you instead see an error, scroll to **§8 Troubleshooting** at the bottom.

---

## 2. Train INDUST — the toothbrush anomaly detector (≈ 30 min)

### 2.1 What is anomaly detection, in plain English

We are going to teach a model what a **good** toothbrush looks like. We will only ever show it good toothbrushes. Then, when we show it a new toothbrush at inference time, the model will try to "redraw" what it sees from memory. If the new toothbrush looks like the good ones, the redrawing will be very close to the input. If the toothbrush has bent bristles, a missing patch of bristles, or contamination on the head, the redrawing will be wrong in exactly that spot — and we will use the difference between input and redrawing as our **anomaly score**.

The model that does the redrawing is called an **autoencoder**. Think of it as a person who studied toothbrushes all day; ask them to draw any toothbrush from memory and they'll do fine. Ask them to draw one with bent bristles and they'll instinctively draw a normal toothbrush, leaving the bent bristles un-rendered. That mismatch is exactly the part where their drawing disagrees with the photo.

### 2.2 Get the toothbrush dataset

We use the **MVTec AD** dataset, the standard benchmark for industrial anomaly detection. You only need the `toothbrush` subset (~50 MB). Two options:

**Option A — Kaggle (recommended, free, no email).** Create a free account at <https://www.kaggle.com/>, then download from <https://www.kaggle.com/datasets/ipythonx/mvtec-ad>. The full dataset is 5 GB; once unzipped you only need the `toothbrush/` folder. Move it so the layout looks like:

```
~/Desktop/MLAI-Machine-Learning/training/datasets/mvtec/toothbrush/
├── train/
│   └── good/                 ← ~60 .png files
├── test/
│   ├── good/
│   └── defective/
└── ground_truth/             ← we don't use this
```

**Option B — Official MVTec site.** Register at <https://www.mvtec.com/company/research/datasets/mvtec-ad>, download `toothbrush.tar.xz`, and extract it to the same path.

> **Sanity check:** when you're done, this command should print a number around 60:
> ```bash
> ls training/datasets/mvtec/toothbrush/train/good/ | wc -l
> ```
>
> Heads-up: MVTec's toothbrush category has only ~60 training images (vs ~200 for most other categories). That's why we bumped epochs up in §2.4 — fewer images per epoch means the model needs more passes to learn the texture.

### 2.3 The training script

Save the file below as `training/indust/train_autoencoder.py`. (Copy-paste the whole thing, including the comments — they're there to help you understand what each block does.)

```python
#!/usr/bin/env python3
"""
training/indust/train_autoencoder.py — Train a convolutional autoencoder
on MVTec good-toothbrush images, then export it to TFLite for the Pi engine.

Why this script exists:
  The original training plan used PaDiM via Anomalib. That toolchain no
  longer exports cleanly to TFLite. A simple autoencoder produces a model
  the engine can already load (it expects: heatmap output + scalar score).
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Keras 3 (shipped with TF 2.18) requires `keras.ops.*` for symbolic tensors
# instead of raw `tf.*`. We use `kops` as a short alias below.
kops = keras.ops

IMG_SIZE = 256


def conv_block(x, filters: int):
    """Conv + BatchNorm + LeakyReLU.

    BatchNorm keeps activations centred and well-scaled across batches,
    and LeakyReLU (with a 0.1 slope on negative inputs) prevents the
    'dead ReLU' failure mode where neurons collapse to zero output and
    stop receiving gradient updates. Together they make this autoencoder
    converge reliably regardless of random initialisation.
    """
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x


def build_autoencoder() -> Model:
    """A small encoder-decoder. Trains in ~5 min on a CPU."""
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    # Encoder: shrink the image while keeping the important features
    x = conv_block(inp, 32)
    x = layers.MaxPool2D(2)(x)                       # 128x128
    x = conv_block(x, 64)
    x = layers.MaxPool2D(2)(x)                       #  64x64
    x = conv_block(x, 128)
    x = layers.MaxPool2D(2)(x)                       #  32x32
    # Bottleneck: the model is forced to compress its understanding here
    x = conv_block(x, 256)
    # Decoder: expand back to a full image
    x = layers.UpSampling2D(2)(x)                    #  64x64
    x = conv_block(x, 128)
    x = layers.UpSampling2D(2)(x)                    # 128x128
    x = conv_block(x, 64)
    x = layers.UpSampling2D(2)(x)                    # 256x256
    out = layers.Conv2D(3, 3, activation="sigmoid", padding="same",
                        name="reconstruction")(x)
    return Model(inp, out, name="autoencoder")


def load_dataset(folder: Path, batch: int):
    """Load all images in `folder` as an unlabelled dataset."""
    ds = tf.keras.utils.image_dataset_from_directory(
        folder.parent,           # parent so Keras finds the 'good' subfolder
        labels=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch,
        shuffle=True,
    )
    # Scale 0..255 -> 0..1; the model both reads and writes in this range.
    return ds.map(lambda x: (x / 255.0, x / 255.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("datasets/mvtec/toothbrush/train/good"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--output", type=Path,
                   default=Path("../models/indust/padim_toothbrush.tflite"))
    args = p.parse_args()

    print(f"[1/4] Loading images from {args.data}")
    ds = load_dataset(args.data, args.batch)

    checkpoint = args.output.with_suffix(".keras")
    if checkpoint.exists():
        # Resume from a previous successful training run.
        print(f"[2/4] Found existing checkpoint at {checkpoint} — skipping training")
        model = keras.models.load_model(checkpoint)
        model.summary()
        print("[3/4] (Skipped — using saved weights)")
    else:
        print("[2/4] Building autoencoder")
        model = build_autoencoder()
        # Lower-than-default learning rate (1e-4 instead of 1e-3) makes training
        # less likely to overshoot in the first few epochs and collapse.
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
        model.summary()

        print(f"[3/4] Training for {args.epochs} epochs")
        model.fit(ds, epochs=args.epochs)

        # Save the trained autoencoder so a future re-run can skip straight to
        # the export step (handy if anything below this point fails).
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint)
        print(f"      saved trained autoencoder -> {checkpoint}")

    print("[4/4] Computing anomaly normalisation constant")
    # The engine expects a 0..1 score. We need to know what 'normal' MSE
    # looks like so we can scale it. We compute the 99th percentile MSE
    # on the training set and bake it into the inference graph.
    errors = []
    for batch_x, _ in ds:
        recon = model(batch_x, training=False)
        e = tf.reduce_mean(tf.square(batch_x - recon), axis=[1, 2, 3]).numpy()
        errors.extend(e.tolist())
    p99 = float(np.percentile(errors, 99))
    print(f"      99th-percentile training MSE = {p99:.6f}")

    print(f"[done] Converting to TFLite -> {args.output}")
    # NOTE: We avoid both TFLiteConverter.from_keras_model() AND model.export()
    # — both are broken on Keras 3 / TF 2.18 / Python 3.12 in different ways:
    #   * from_keras_model() raises 'NoneType is not callable' in tflite_keras_util
    #   * model.export()      raises '_DictWrapper' TypeError in trackable_view
    # The only path that actually works on this stack is to define the inference
    # logic as a @tf.function with an explicit input signature, get its concrete
    # function, and hand it directly to TFLiteConverter.from_concrete_functions().
    #
    # Inside the @tf.function we use raw tf.* ops (NOT keras.ops) because we're
    # tracing real tensors during graph construction, not building a symbolic
    # Keras graph from KerasTensors.
    p99_const = float(p99 * 3.0)
    autoencoder = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32),
    ])
    def inference_fn(x):
        recon = autoencoder(x, training=False)
        pixel_err = tf.reduce_mean(tf.square(x - recon), axis=-1)      # [1, H, W]
        image_err = tf.reduce_mean(pixel_err, axis=[1, 2])             # [1]
        score = tf.clip_by_value(image_err / p99_const, 0.0, 1.0)      # [1]
        return pixel_err, score

    concrete_func = inference_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], autoencoder
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_bytes)
    print(f"       wrote {len(tflite_bytes) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
```

### 2.4 Run training

```bash
cd training
python indust/train_autoencoder.py
```

You'll see TensorFlow print roughly this:

```
[1/4] Loading images from datasets/mvtec/toothbrush/train/good
Found 60 files belonging to 1 classes.
[2/4] Building autoencoder
Model: "autoencoder"
... (a table showing layers)
[3/4] Training for 50 epochs
Epoch 1/50
8/8 [====] - 3s 320ms/step - loss: 0.0612
Epoch 2/50
8/8 [====] - 2s 290ms/step - loss: 0.0241
...
```

**What to watch for:** the `loss` number should keep going down, epoch after epoch. If it's still 0.05 at epoch 50, something is wrong. A healthy run ends around `loss: 0.001 – 0.005`.

On a modern laptop CPU this takes **10–25 minutes** — fewer images per epoch means fewer total steps, even with more epochs. If you see `Killed` partway through, your machine ran out of RAM; re-run with `--batch 4`. While it runs you can move on to §3 in another terminal, but **don't open the same `.venv-train`** in two terminals at once unless you `source` it again in the new one.

### 2.5 Verify the file landed

```bash
ls -lh ../models/indust/padim_toothbrush.tflite
```

You should see a file of around **5–15 MB**. Done with INDUST.

---

## 3. AGRO fruit detector — download a pretrained model (≈ 5 min)

### 3.1 Why we are not training this

Object detection (drawing a box around a fruit) needs a dataset of **labelled bounding boxes**. The Fruits-360 dataset that the original recipe pointed at has no boxes — it's just centred photos of single fruits on white backgrounds. Building a real bounding-box dataset would mean either annotating thousands of images by hand, or downloading a different dataset like Open Images Fruit Subset (~3 GB) and training for several hours.

There's a much easier path: Google has already trained an SSD MobileNet V1 detector on **COCO**, a dataset with 90 everyday object classes. Three of those classes are `apple`, `banana`, and `orange` — exactly the three fruits in the quality classifier dataset we use in §4. We get fruit detection for free, no training required.

> **What about tomato?** It's not in COCO. The original project plan included tomato as a target, but since the easy free model doesn't have it and we'd rather avoid training a custom detector on day one, we've dropped tomato from this guide. You can add it back later by training a custom detector if you collect a tomato dataset.

### 3.2 Download

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

### 3.3 Update the AGRO config

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
> The engine currently uses `fruit_classes` from `config/agro/config.yaml` as a positional list, which doesn't filter by COCO index. The simplest correct setup is to add a small filter in `engine/agro/detector.py` that drops detections whose index isn't 52, 53, or 55, and remaps them to "banana", "apple", "orange" respectively. For now, leave it as-is — at worst you'll see a few extra "spoon" or "bowl" detections in the dashboard, and we can fix that in the next phase. **Flagged as a known limitation.**

---

## 4. AGRO fruit quality classifier (≈ 90 min)

### 4.1 What you're training

A small image classifier that takes a fruit crop and answers: **good or defective?** "Defective" is a catch-all here — it covers physical damage (bruises, cuts, dents) *and* biological decay (mold, rot). The model can't tell the two apart on its own; if you want that distinction, see §4.5 for the upgrade path. This is the only AGRO model we actually train ourselves, because it's both useful and easy.

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

**Short answer:** yes, there are several. But for a 2-class fruit classifier with ~1,000 images on a Pi, the difference is small enough that it's not worth complicating the project. Here's the lay of the land:

| Year | Model | Why it matters |
|---|---|---|
| 2018 | **MobileNet V2** ← we use this | The default. Battle-tested, well-documented, supported everywhere |
| 2019 | MobileNet V3 | Faster and slightly more accurate; uses NAS-designed building blocks |
| 2019 | EfficientNet B0–B7 | Landmark architecture — better accuracy/efficiency curve |
| 2020 | EfficientNet-Lite | EfficientNet redesigned specifically for TFLite / edge devices |
| 2020 | Vision Transformers (ViT) | Different paradigm — great with huge datasets, less great on small ones like ours |
| 2021 | EfficientNetV2 | Faster training, slightly better accuracy |
| 2022 | ConvNeXt | "The convnet strikes back" — competes with transformers using pure CNN tricks |
| 2023 | DINOv2 | Self-supervised features from Meta — best general-purpose features available |
| 2024 | MobileNetV4 | Latest mobile-optimised model from Google |

For our specific situation (binary classification, ~1,000 images, CPU training, edge deployment), the realistic upgrade options are:

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

1. **The bottleneck is data, not model.** With only ~1,000–2,000 training images, no model architecture will dramatically outperform another. You'd improve more by collecting better labels than by switching backbones.
2. **Battle-tested wins.** MobileNet V2 has 8 years of bug reports, tutorials, and known-working configurations. Newer models occasionally have subtle issues with TFLite export, specific activation ops, or input preprocessing that bite you at the worst moment.
3. **Marginal improvement at best.** Going to MobileNet V3 might give you ~1–2% better accuracy. Going to EfficientNet-Lite might give ~3%. Neither is worth the risk of breaking the pipeline you're about to set up.
4. **You can always upgrade later.** Once your end-to-end system works with MobileNet V2, swapping the backbone is a one-line change. It's a great experiment for a v2 of your project.

In professional ML, **picking the model is almost never the most important decision.** Data quality and pipeline correctness matter 10× more than backbone choice. Use MobileNet V2 for now. When you have a working system and want to push accuracy further, *then* experiment with the newer ones — and you'll have the baseline to compare against.

### 4.2 Get the dataset

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

Unzip it into `training/datasets/fruits/` so the layout above is exact. The script in §4.3 expects this path by default.

> **Sanity check:** this command should print a number around 1700:
> ```bash
> ls training/datasets/fruits/train/freshapples/ | wc -l
> ```

### 4.3 Reorganise into the layout the trainer expects

There's a small mismatch between what the dataset gives you and what `train_quality.py` wants:

- **The Kaggle dataset** has 6 fruit-specific folders (`freshapples/`, `freshbanana/`, ..., `rottenoranges/`).
- **`train_quality.py`** wants 2 quality-specific folders (`good/`, `defective/`).

The `reorganise_quality.py` script bridges that gap. It walks the source dataset, treats every folder starting with `fresh*` as "good" and every folder starting with `rotten*` as "defective", then copies the images into the layout the trainer wants. It also **respects the dataset's existing train/test split** — those splits were hand-picked by the dataset authors and are higher quality than a random shuffle would be.

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

Exact numbers will vary slightly depending on the dataset version, but you should see roughly 10,000 train images and 3,000 val images, balanced 50/50 between good and defective. The output layout it produces:

```
training/datasets/agro_quality/
├── train/
│   ├── good/        ← all "fresh*" images from datasets/fruits/train/
│   └── defective/   ← all "rotten*" images from datasets/fruits/train/
└── val/
    ├── good/        ← all "fresh*" images from datasets/fruits/test/
    └── defective/   ← all "rotten*" images from datasets/fruits/test/
```

> **Sanity check:** these counts should each be around 5,000 / 1,700:
> ```bash
> ls datasets/agro_quality/train/good/      | wc -l
> ls datasets/agro_quality/train/defective/ | wc -l
> ls datasets/agro_quality/val/good/        | wc -l
> ls datasets/agro_quality/val/defective/   | wc -l
> ```

### 4.4 Train the classifier

Time to actually train the model. The script `training/agro/train_quality.py` does everything in one shot — you just run it and wait.

Here's what the script actually does, in plain English:

1. **Loads your fruit photos** from `datasets/agro_quality/` (the folders `reorganise_quality.py` just made for you).
2. **Adds variety on the fly** — random horizontal flips, slight brightness changes, slight contrast tweaks. This makes the model more robust because it sees each fruit in slightly different conditions every time it cycles through.
3. **Builds the model** using the "art critic" trick from §4.1 — the pre-trained MobileNet V2 backbone is frozen, and a tiny new head is bolted on for our 2 classes (good, defective).
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
Model: "functional"
... (a layer summary — the big block is the frozen base, the tiny head sits on top)
[3/4] Training head for 15 epochs (base is frozen)
Epoch 1/15
645/645 ━━━ 270s 410ms/step - loss: 0.4180 - accuracy: 0.8210 - val_loss: 0.2010 - val_accuracy: 0.9230
Epoch 2/15
645/645 ━━━ 265s 410ms/step - loss: 0.1820 - accuracy: 0.9440 - val_loss: 0.1390 - val_accuracy: 0.9510
...
Epoch 15/15
645/645 ━━━ 263s 408ms/step - loss: 0.0480 - accuracy: 0.9850 - val_loss: 0.0820 - val_accuracy: 0.9710
      saved trained classifier -> ../models/agro/fruit_quality.keras
[4/4] Converting to TFLite -> ../models/agro/fruit_quality.tflite
      wrote ../models/agro/fruit_quality.tflite (XXXX KB) and ../models/agro/fruit_quality.labels.txt
```

**The number to watch:** look at the `val_accuracy` column on the right. That's the only number that really matters. It tells you how well the model does on fruit photos it has *never seen* during training — the honest "exam score". The plain `accuracy` column is just how well the model remembers the training photos, which doesn't prove it actually learned anything (a model can score 100% there just by memorising). **Aim for `val_accuracy > 0.90`** by epoch 15.

**How long it takes:** about **45–90 minutes** on a normal laptop CPU. Most of that time is the model running each image through MobileNet V2's frozen base. There's no easy way to speed this up without a GPU, so just let it run in the background while you do something else.

> **If something breaks during export (resumability).** The script saves a backup file at `models/agro/fruit_quality.keras` the moment training finishes — *before* the export step. So if anything goes wrong while writing the `.tflite` (which has happened to us before with this stack), you don't lose the 60 minutes of training. Just re-run the script and it'll spot the backup and skip straight to the export step (~30 seconds total). If you ever want to start completely fresh, delete the backup first:
>
> ```bash
> rm ../models/agro/fruit_quality.keras
> python agro/train_quality.py
> ```

> **If accuracy is too low.** If your `val_accuracy` ends up below 0.90, the model isn't quite good enough yet. Re-run with the `--fine_tune` flag:
>
> ```bash
> rm ../models/agro/fruit_quality.keras    # force a fresh retrain
> python agro/train_quality.py --fine_tune --epochs 30
> ```
>
> This "unlocks" the frozen MobileNet V2 base and lets it adjust slightly to your specific fruits, instead of using the generic version Google trained on the internet. Slower (it has to update millions of weights instead of a few thousand), but usually gains 1–3% accuracy. **Don't use `--fine_tune` on your first run** — get a working baseline first, then decide if you actually need it.

When it finishes, check the file landed:

```bash
ls -lh ../models/agro/fruit_quality.tflite
```

You should see a file of around **3–10 MB**. The reason it's smaller than you might expect: TensorFlow Lite automatically compresses the model's weights from 4-byte floating-point numbers to 1-byte integers (this trick is called **quantization**), which shrinks the file by ~10× without much loss in accuracy. Done with AGRO.

### 4.5 Later: upgrading to a 3-class model

> **You can skip this on your first pass.** Come back here once §4 is working end-to-end and you want a model that tells *bruised* apart from *rotten* (instead of lumping them together as "defective").

The engine is already wired for three quality classes — see `config/agro/config.yaml`:

```yaml
quality_classes:
  - good
  - defective
  - unripe
```

The `train_quality.py` script auto-detects the number of classes from the folder structure, so adding a third class is mostly a **dataset** problem, not a code problem.

**What you'd need to do:**

1. **Find a third dataset.** The Kaggle "Fresh and Rotten" set we use only has two classes. Realistic sources for a `damaged` (or `unripe`) class:
   - **Roboflow** — search for "bruised apples", "fruit defect", or "damaged fruit". Most are a few hundred labelled images, free with a Roboflow account.
   - **Mendeley FruitNet** — Indian fruits with `good` / `bad` / `mixed` labels (~2 GB).
   - **Custom photos** — for a real demo, take 100–200 photos of bruised fruit yourself with good lighting. This usually outperforms public datasets because the lighting matches your camera.

2. **Add a third folder** to the dataset layout:
   ```
   datasets/agro_quality/
   ├── train/
   │   ├── good/
   │   ├── defective/   ← rotten only, in this scheme
   │   └── damaged/     ← physical damage only
   └── val/
       ├── good/
       ├── defective/
       └── damaged/
   ```

3. **Update `config/agro/config.yaml`** so `quality_classes` lists the three folder names *in the same alphabetical order* the trainer will see them (Keras's `image_dataset_from_directory` sorts class folders alphabetically):
   ```yaml
   quality_classes:
     - damaged
     - defective
     - good
   ```

4. **Re-run training** with the same command from §4.4. The model will auto-grow its output layer to 3 neurons.

**Why this is harder than it sounds:** the three datasets almost never match in lighting, background, or fruit variety. A model trained this way often "cheats" — it learns *which dataset an image came from* instead of what damage looks like. The fix is data augmentation (random crops, colour jitter, brightness shifts) and ideally collecting at least the `damaged` class yourself with the same camera you'll use at inference time. Worth doing only when you have a real use case that demands the distinction.

---

## 5. Verify all three models exist

```bash
cd ~/Desktop/MLAI-Machine-Learning
ls -lh models/indust/ models/agro/
```

Expected:

```
models/agro/:
fruit_detector.tflite        ~4 MB
fruit_detector.labels.txt    ~1 KB
fruit_quality.tflite         ~10 MB

models/indust/:
padim_toothbrush.tflite      ~10 MB
```

Now confirm the existing tests still pass with real models present (they ran in mock mode before). The tests live in the project's `tests/` folder and use `pytest`, which we installed in §1.2. Run from the **project root**, with the venv active:

```bash
cd ~/Desktop/MLAI-Machine-Learning      # back to the project root
source .venv-train/bin/activate         # if not already active
python3 -m pytest tests/ -q
```

You should see `12 passed`. The tests don't actually run the models, but they will catch broken file paths.

> **Got `No module named pytest` / `fastapi` / `yaml` / `psutil` / `httpx`?** That means you set up the venv before this guide added the engine + API dependencies to §1.2. Re-run **both** pip install lines from §1.2 to top up the venv:
> ```bash
> pip install "tensorflow==2.18.*" opencv-python pillow numpy tqdm requests pytest
> pip install "fastapi==0.115.*" "uvicorn[standard]==0.32.*" "pydantic==2.9.*" pyyaml psutil httpx
> ```
> pip will skip anything already installed, so this is safe to run multiple times. Then re-run the test command above.
>
> **A few naming gotchas to know:**
> - **`pyyaml`, not `yaml`.** The Python YAML library imports as `yaml` but installs as `pyyaml`. There's a tiny unrelated package on PyPI called `yml` that does nothing — if you accidentally installed it, remove it with `pip uninstall -y yml`.
> - **`httpx` is needed for `fastapi.testclient.TestClient`.** It's a transitive dependency that `pip install fastapi` does NOT pull in. The test API tests fail without it even though `fastapi` itself is installed.

### Optional: smoke-test the models actually load and run

There's a ready-made smoke-test script at `tests/scratch_test.py` that loads each `.tflite` file and runs a single dummy input through it. **It's safe to run incrementally** — missing files are reported as `[skip]` rather than crashing the script, so you can use it after §2 (only INDUST done), after §3 (INDUST + AGRO detector), and again after §4 (all three).

Run it from anywhere — the script locates `models/` relative to its own file path, so the current working directory doesn't matter:

```bash
python3 tests/scratch_test.py        # from the project root
python3 scratch_test.py              # from inside tests/
```

Expected output once everything is in place:

```
Project root: /home/felipe/Desktop/MLAI-Machine-Learning
  [ok]   models/indust/padim_toothbrush.tflite  input=[1, 256, 256, 3] float32  outputs=2
  [ok]   models/agro/fruit_detector.tflite  input=[1, 300, 300, 3] uint8  outputs=4
  [ok]   models/agro/fruit_quality.tflite  input=[1, 224, 224, 3] float32  outputs=1

Summary: 3 ok, 0 skipped, 0 failed
```

If you only have one or two models so far, you'll see `[skip]` lines for the missing ones — that's fine, it just means you haven't reached that section of the guide yet:

```
Project root: /home/felipe/Desktop/MLAI-Machine-Learning
  [ok]   models/indust/padim_toothbrush.tflite  input=[1, 256, 256, 3] float32  outputs=2
  [skip] models/agro/fruit_detector.tflite  (file not found)
  [skip] models/agro/fruit_quality.tflite  (file not found)

Summary: 1 ok, 2 skipped, 0 failed
(skipped files don't exist yet — see training/README.md §3 and §4)
```

Three `[ok]` lines and zero `[fail]` means you're done.

---

## 6. Copy the models to the Pi

When you're ready to deploy, copy the trained files over and restart the engine:

```bash
scp models/indust/*.tflite pi@<pi-ip>:~/MLAI-Machine-Learning/models/indust/
scp models/agro/*.tflite   pi@<pi-ip>:~/MLAI-Machine-Learning/models/agro/

ssh pi@<pi-ip> "sudo systemctl restart mlai-engine"
```

Replace `<pi-ip>` with the actual IP address of your Raspberry Pi (find it on the Pi with `hostname -I`). The full Pi setup — installing the OS, the systemd services, calibrating the camera — lives in `SetupGuide.md` at the repo root.

---

## 7. What you just accomplished

You went from an empty `models/` folder to three trained `.tflite` files that the engine can load:

- **INDUST**: a custom autoencoder trained on ~60 photos of good toothbrushes, capable of producing both an anomaly score and a per-pixel heatmap of suspicious regions.
- **AGRO detector**: a 4 MB SSD MobileNet V1 from Google's reference TFLite models, detecting apples, bananas, and oranges.
- **AGRO quality**: a transfer-learned MobileNet V2 classifier that distinguishes fresh from rotten fruit.

The next step is to copy these files to the Raspberry Pi, install the systemd services, and run a full end-to-end test with the camera. That's covered in `SetupGuide.md`.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | You forgot to activate the venv | `source .venv-train/bin/activate` |
| `pip install tensorflow` is very slow / freezes | Old pip version | `pip install --upgrade pip` and try again |
| Training loss stays high (`> 0.05`) and never drops | Image folder is empty or wrong path | Check `ls datasets/mvtec/toothbrush/train/good \| wc -l` |
| `Killed` during training | Out of RAM (TF allocated too much) | Lower `--batch` to `4` |
| `val_accuracy` stuck at 0.5 | Class folders are empty or imbalanced | Re-run `reorganise_quality.py` and check the printed counts |
| `tflite_runtime` not found (when running engine) | The Pi runtime isn't installed on this PC | That's fine — the engine only uses it on the Pi. On your dev PC the engine falls back to full TensorFlow's interpreter. |
| Disk full | MVTec full archive is 5 GB | Delete categories you're not using; you only need `toothbrush/` |

If you hit something not in this table, copy the **exact error message** (the last 10 lines of red text), and we'll debug it together.

---

## 9. Glossary (in plain English)

| Term | What it means |
|---|---|
| **Model** | A file containing learned numbers (called weights). Think of it as a brain saved to disk. |
| **Training** | The slow process of showing examples to the model so it adjusts those numbers. |
| **Inference** | Using a trained model on new data. Fast. |
| **Epoch** | One full pass over your training images. More epochs = more learning, up to a point. |
| **Batch** | A small group of images processed together (e.g. 8 at a time). Bigger batches = faster but use more RAM. |
| **Loss** | A number telling the model how wrong it is on the current batch. Lower = better. |
| **Accuracy** | Percentage of validation images the model labels correctly. Higher = better. |
| **Validation set** | A held-out chunk of data the model never trains on, used to measure honest performance. |
| **Autoencoder** | A model that tries to reproduce its input. Useful for anomaly detection because it fails on unfamiliar inputs. |
| **Transfer learning** | Starting from someone else's trained model and only retraining the last layer for your task. |
| **TFLite** | A compact file format for trained models, optimised for slow devices like the Raspberry Pi. |
| **MSE** | Mean Squared Error. The average of (prediction − truth)² over all pixels. |
| **MVTec AD** | A famous dataset of 15 industrial object categories with good and defective examples. |
| **COCO** | A famous dataset of 90 everyday object categories with bounding boxes. |
