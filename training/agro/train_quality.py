#!/usr/bin/env python3
"""
training/agro/train_quality.py — MobileNet V2 transfer learning for fruit quality.

Trains a small classifier that takes a fruit image crop and outputs one of:
    good  |  defective  |  unripe

We use the high-level Keras API. Steps:
    1. Load images from <dataset>/train and <dataset>/val using image_dataset_from_directory.
    2. Apply data augmentation (random flips, crops, brightness).
    3. Stack a frozen MobileNet V2 base + a small classification head.
    4. Train head only for a few epochs, then optionally unfreeze and fine-tune.
    5. Export to TFLite.

Dataset format:
    dataset/
        train/{good,defective,unripe}/*.jpg
        val/{good,defective,unripe}/*.jpg

Usage:
    python train_quality.py --dataset ~/Downloads/fruit-quality --output ../../models/agro/fruit_quality.tflite --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("train_quality")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--fine_tune", action="store_true", help="Unfreeze base and fine-tune for extra epochs")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    try:
        import tensorflow as tf
    except Exception as exc:
        logger.error("TensorFlow missing: %s", exc)
        return 2

    img_size = (args.image_size, args.image_size)

    logger.info("Loading datasets from %s", args.dataset)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset / "train",
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset / "val",
        image_size=img_size,
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    logger.info("Classes: %s", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze the base for the first round

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Training head only for %d epochs", args.epochs)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune:
        logger.info("Unfreezing base for fine-tuning")
        base.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=max(5, args.epochs // 3))

    logger.info("Exporting to TFLite at %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    args.output.write_bytes(tflite_bytes)

    labels_path = args.output.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(class_names) + "\n")
    logger.info("Done. Wrote %s and %s", args.output, labels_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
