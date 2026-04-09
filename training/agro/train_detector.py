#!/usr/bin/env python3
"""
training/agro/train_detector.py — Fine-tune SSD MobileNet V2 on our 3 fruits.

This script uses **TensorFlow Lite Model Maker**, which is the easiest way
to do transfer learning for object detection on the edge. You feed it a
folder of labelled images and it produces a tiny `.tflite` file ready to
run on a Raspberry Pi.

Concept of transfer learning:
  Training a model from scratch needs millions of images and lots of GPUs.
  Instead, we start from MobileNet V2 — a model someone else already
  trained on ImageNet (1.4M images) — and only re-train the last few
  layers for our specific classes. This usually works in minutes.

Dataset format expected (produced by prepare_dataset.py):
    dataset/agro/
        labels.txt
        train/{apple,orange,tomato}/*.jpg
        val/{apple,orange,tomato}/*.jpg

Usage:
    python train_detector.py --dataset dataset/agro --output ../../models/agro/fruit_detector.tflite --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("train_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    try:
        # tflite-model-maker is heavy and only works on Python 3.9-3.11.
        from tflite_model_maker import model_spec
        from tflite_model_maker import object_detector
        from tflite_model_maker.config import ExportFormat
    except Exception as exc:
        logger.error("tflite-model-maker not installed: %s", exc)
        logger.error("Try: pip install tflite-model-maker")
        return 2

    train_dir = args.dataset / "train"
    val_dir = args.dataset / "val"
    labels_file = args.dataset / "labels.txt"
    if not labels_file.exists():
        logger.error("labels.txt not found in %s — run prepare_dataset.py first", args.dataset)
        return 3
    label_map = {idx + 1: name.strip() for idx, name in enumerate(labels_file.read_text().splitlines()) if name.strip()}
    logger.info("Label map: %s", label_map)

    # NOTE: TF Model Maker's classification API is folder-based; for object
    # detection it expects PASCAL VOC XML labels. If you only have classification
    # labels, see prepare_dataset.py to generate per-image bbox annotations.
    logger.info("Loading datasets from %s", args.dataset)
    train_data = object_detector.DataLoader.from_pascal_voc(
        images_dir=str(train_dir),
        annotations_dir=str(train_dir),
        label_map=label_map,
    )
    val_data = object_detector.DataLoader.from_pascal_voc(
        images_dir=str(val_dir),
        annotations_dir=str(val_dir),
        label_map=label_map,
    )

    spec = model_spec.get("efficientdet_lite0")  # closest stand-in shipped with Model Maker
    logger.info("Training (%d epochs, batch=%d)", args.epochs, args.batch_size)
    model = object_detector.create(
        train_data,
        model_spec=spec,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_whole_model=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting TFLite to %s", args.output)
    model.export(
        export_dir=str(args.output.parent),
        tflite_filename=args.output.name,
        export_format=[ExportFormat.TFLITE, ExportFormat.LABEL],
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
