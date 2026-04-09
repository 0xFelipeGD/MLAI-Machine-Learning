# MLAI Training Guide — From Zero to Trained Models

> **You don't need any prior machine-learning knowledge to follow this guide.** Read it top to bottom and execute every command in order.

This folder contains everything you need to train the two models that ship with MLAI:

| Folder | Module | What it produces |
|--------|--------|------------------|
| `indust/` | Industrial anomaly detection | A `.tflite` file that scores how "weird" a part looks |
| `agro/`   | Fruit detection + grading    | Two `.tflite` files: one finds fruits, one grades them |

Training runs on your **PC** (a GPU helps but is optional). The trained models are then copied to the Raspberry Pi for inference.

---

## 1. What is machine learning, in 60 seconds

Imagine you wanted to teach a child what an apple looks like. You'd show them many apples until they learned the pattern. Machine learning is the same: we feed a program thousands of labelled images, and it builds an internal "feel" for what each label looks like. The result is called a **model** — a file containing learned numbers (called *weights*).

* **Training** = the slow process of teaching the model from examples.
* **Inference** = the fast process of using the trained model to label new images.
* **TFLite** (.tflite) = a small, optimised model file format that runs fast on a Raspberry Pi.

INDUST uses **anomaly detection**: we only show the model "good" parts and it flags anything that doesn't match. AGRO uses **object detection**: we show labelled fruits so the model learns to find and name them.

---

## 2. Set up your training environment (one-time)

### 2.1 Install Miniconda

Download from <https://docs.conda.io/en/latest/miniconda.html> and run the installer for your OS. Conda gives us isolated Python environments so we don't break your system.

### 2.2 Create the `mlai` environment

```bash
cd training
conda env create -f environment.yml
conda activate mlai
```

This installs TensorFlow, OpenCV, Anomalib, TF Model Maker and a few other things. Activation must be repeated every time you open a new terminal (`conda activate mlai`).

### 2.3 Verify it works

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, 'OK')"
```

If you have an NVIDIA GPU, also run:

```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

A non-empty list means TF will use the GPU automatically — training will be **much** faster.

---

## 3. Train INDUST (anomaly detection)

### 3.1 Download the dataset

We use the **MVTec AD** dataset. It contains 15 categories of industrial parts, each with a "good" subset (training) and various defective subsets (testing).

Anomalib downloads MVTec automatically the first time you train, so you can skip ahead. If you want to grab it manually go to <https://www.mvtec.com/company/research/datasets/mvtec-ad>.

### 3.2 Train PaDiM on the bottle category

```bash
python indust/train_padim.py --category bottle --epochs 1
```

PaDiM is special: it doesn't really "train" in the deep-learning sense. It walks through the good images once, extracts features from a frozen CNN, and computes a Gaussian distribution for each pixel patch. **One epoch is enough.**

### 3.3 Export to TFLite

```bash
python indust/export_tflite.py --category bottle --output ../models/indust/padim_bottle.tflite
```

This converts the trained model to the small format that runs on the Pi.

### 3.4 Repeat for other categories

```bash
python indust/train_padim.py --category metal_nut --epochs 1
python indust/export_tflite.py --category metal_nut --output ../models/indust/padim_metal_nut.tflite

python indust/train_padim.py --category screw --epochs 1
python indust/export_tflite.py --category screw --output ../models/indust/padim_screw.tflite
```

You can train any of the 15 MVTec categories. Edit `config/indust/config.yaml` on the Pi to add new categories to the dropdown.

---

## 4. Train AGRO (fruit detection + quality)

### 4.1 Download datasets

* **Fruits-360** — labelled photos of fruits. Get it from <https://www.kaggle.com/datasets/moltean/fruits>.
* **Fruit Freshness** — fresh vs rotten labels. <https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification>.

Unzip both anywhere on your PC. We'll point the prepare script at them.

### 4.2 Prepare the detection dataset

```bash
python agro/prepare_dataset.py \
  --source ~/Downloads/fruits-360 \
  --output dataset/agro
```

This extracts only `apple`, `orange`, and `tomato`, splits them into train/val, and writes a labels file.

### 4.3 Train the fruit detector

```bash
python agro/train_detector.py \
  --dataset dataset/agro \
  --output ../models/agro/fruit_detector.tflite \
  --epochs 50
```

Under the hood this fine-tunes **SSD MobileNet V2** — a fast, small object detector. It uses transfer learning, meaning we start from a model that already knows generic image features and just teach it the new classes. That makes training fast.

### 4.4 Train the quality classifier

```bash
python agro/train_quality.py \
  --dataset ~/Downloads/fruits-fresh-rotten \
  --output ../models/agro/fruit_quality.tflite \
  --epochs 30
```

This produces a small classifier that takes a fruit crop and outputs `good`, `defective` or `unripe`.

---

## 5. Copy the models to the Pi

```bash
scp ../models/indust/*.tflite pi@<pi-ip>:~/MLAI-Machine-Learning/models/indust/
scp ../models/agro/*.tflite   pi@<pi-ip>:~/MLAI-Machine-Learning/models/agro/
```

Restart the engine on the Pi:

```bash
ssh pi@<pi-ip> "sudo systemctl restart mlai-engine"
```

---

## 6. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `tensorflow not found` | Conda env not activated | `conda activate mlai` |
| `CUDA out of memory` | Batch size too high | Pass `--batch_size 4` |
| Anomalib download fails | Firewall / SSL | Set `ANOMALIB_CACHE_DIR` and download MVTec manually |
| TFLite export error | Model uses op not supported | Pass `--allow_select_tf_ops` to the export script |

For anything else, check that you're inside the `mlai` conda environment and that the dataset paths are correct.

---

## 7. Glossary

| Term | Plain English |
|------|---------------|
| Epoch | One complete pass over the training images |
| Batch | A small group of images processed at once |
| Loss | A number that tells the model how wrong it is — lower is better |
| Transfer learning | Starting from someone else's trained model and tweaking it |
| TFLite | A small, fast file format for trained models |
| Inference | Using a trained model on new data |
