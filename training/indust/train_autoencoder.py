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