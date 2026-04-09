#!/usr/bin/env python3
"""
training/agro/train_quality.py — MobileNet V2 transfer learning for fruit quality.

Trains a small classifier that takes a fruit image crop and outputs one of:
    good  |  defective

We start from MobileNet V2 (pre-trained on ImageNet, free from Google),
freeze its feature extractor, and bolt on a tiny classification head that
learns to distinguish good from defective. This is **transfer learning**:
because the heavy "learning to see" work was already done on millions of
images, we only need ~1000 labelled examples and ~60 minutes on a CPU.

Pipeline:
    1. Load images from <dataset>/train and <dataset>/val using
       image_dataset_from_directory (auto-detects classes from subfolder names).
    2. Normalise inputs to [0, 1] in the dataset pipeline (matches what the
       engine's classifier.py sends at inference time — see below).
    3. Apply data augmentation as a tf.data .map() — random flips, rotations,
       brightness, zoom — only on the training set.
    4. Build a tiny model: Rescaling([0,1] -> [-1,1]) + frozen MobileNet V2 + head.
    5. Train head only for <epochs> epochs.
    6. Optionally unfreeze the base for a fine-tuning pass (--fine_tune).
    7. Save the trained model as a .keras checkpoint.
    8. Export to TFLite via the from_concrete_functions path (the only export
       route that works under Keras 3 / TF 2.18 / Python 3.12).

WHY THE [0, 1] INPUT RANGE INSTEAD OF [0, 255]:
    The engine (engine/agro/classifier.py) divides the camera frame by 255
    before sending it to the model, so the model receives values in [0, 1].
    To stay consistent we feed [0, 1] during training too. The model's first
    real layer is `Rescaling(scale=2.0, offset=-1.0)` which converts
    [0, 1] -> [-1, 1] (MobileNet V2's expected input range). This is the
    SAME math `keras.applications.mobilenet_v2.preprocess_input` does, just
    applied to [0, 1] instead of [0, 255]. If we used preprocess_input
    directly the model would silently mis-process the engine's [0, 1] inputs
    at inference time and produce garbage predictions, even though training
    would look perfect.

WHY AUGMENTATION IS A .map() AND NOT A LAYER:
    Random* layers as part of the model graph occasionally fail to export
    cleanly through TFLite (the TFLite converter is sometimes confused by
    the training=False branch of those layers). Doing augmentation in the
    data pipeline keeps the exported model graph clean and minimal, with
    only deterministic ops, while still augmenting during training.

Dataset format expected (produced by reorganise_quality.py):
    datasets/agro_quality/
        train/{good,defective}/*.jpg
        val/{good,defective}/*.jpg

The script auto-detects the number of classes from the folder structure,
so adding a third class later (e.g. 'damaged') just means adding a new
folder — no code change needed.

Usage (from inside the training/ folder, with .venv-train activated):
    python agro/train_quality.py                  # uses sane defaults
    python agro/train_quality.py --epochs 30      # train longer
    python agro/train_quality.py --fine_tune      # unfreeze base for extra accuracy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("train_quality")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/agro_quality"),
        help="Folder containing train/{good,defective}/ and val/{good,defective}/",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("../models/agro/fruit_quality.tflite"),
        help="Where to write the .tflite file",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument(
        "--fine_tune",
        action="store_true",
        help="Unfreeze the MobileNet base and run extra epochs at a low learning rate. "
             "Slower but can squeeze out a few extra percent of accuracy.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        logger.error("TensorFlow missing: %s", exc)
        return 2

    img_size = (args.image_size, args.image_size)

    if not (args.dataset / "train").exists() or not (args.dataset / "val").exists():
        logger.error("Dataset folder %s must contain train/ and val/ subdirectories.", args.dataset)
        logger.error("Did you run `python agro/reorganise_quality.py` first?")
        return 3

    logger.info("[1/4] Loading datasets from %s", args.dataset)
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        args.dataset / "train",
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        args.dataset / "val",
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    class_names = train_ds_raw.class_names
    logger.info("      classes detected: %s", class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    # Normalise to [0, 1] — matches what engine/agro/classifier.py sends.
    def to_unit_range(x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    # Augmentation only on the training set; runs in the data pipeline so
    # the exported model graph stays free of Random* layers.
    def augment(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        x = tf.clip_by_value(x, 0.0, 1.0)  # keep in [0, 1] after brightness
        return x, y

    train_ds = (
        train_ds_raw
        .map(to_unit_range, num_parallel_calls=AUTOTUNE)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds_raw
        .map(to_unit_range, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    checkpoint = args.output.with_suffix(".keras")
    if checkpoint.exists():
        # Resume from a previous successful training run.
        logger.info("[2/4] Found existing checkpoint at %s — skipping training", checkpoint)
        model = keras.models.load_model(checkpoint)
        logger.info("[3/4] (Skipped — using saved weights)")
    else:
        logger.info("[2/4] Building MobileNet V2 + new classification head")

        # Pre-trained backbone (the "art critic" from the README §4.1
        # transfer learning section). include_top=False drops the original
        # 1000-class ImageNet classifier; weights="imagenet" downloads
        # Google's free pre-trained weights.
        base = keras.applications.MobileNetV2(
            input_shape=img_size + (3,),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False  # FREEZE — only the new head learns

        # Build the model. Input is in [0, 1] (engine sends this); the first
        # real layer rescales to [-1, 1] which is what MobileNet V2 expects.
        inputs = keras.Input(shape=img_size + (3,))
        x = keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
        x = base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        logger.info("[3/4] Training head for %d epochs (base is frozen)", args.epochs)
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

        if args.fine_tune:
            logger.info("      --fine_tune set: unfreezing base for extra epochs")
            base.trainable = True
            model.compile(
                optimizer=keras.optimizers.Adam(1e-5),  # much lower LR
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(train_ds, validation_data=val_ds, epochs=max(5, args.epochs // 3))

        # Save the trained model so a future re-run can skip straight to export.
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint)
        logger.info("      saved trained classifier -> %s", checkpoint)

    logger.info("[4/4] Converting to TFLite -> %s", args.output)
    # NOTE: Same compatibility wall the autoencoder hit. We avoid both
    # TFLiteConverter.from_keras_model() AND model.export() — both are broken
    # on Keras 3 / TF 2.18 / Python 3.12 in different ways. The only path that
    # actually works is to define the inference logic as a @tf.function with
    # an explicit input signature, get its concrete function, and hand it
    # directly to TFLiteConverter.from_concrete_functions().
    classifier = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, args.image_size, args.image_size, 3], dtype=tf.float32),
    ])
    def inference_fn(x):
        return classifier(x, training=False)

    concrete_func = inference_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], classifier
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_bytes)

    # Write the labels file alongside the .tflite so the engine knows which
    # output index corresponds to which class name.
    labels_path = args.output.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(class_names) + "\n")
    logger.info("      wrote %s (%.1f KB) and %s",
                args.output, len(tflite_bytes) / 1024, labels_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
