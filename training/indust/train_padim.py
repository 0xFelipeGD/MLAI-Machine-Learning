#!/usr/bin/env python3
"""
training/indust/train_padim.py — Train PaDiM via Anomalib on an MVTec category.

This script is intentionally short because Anomalib does most of the work.
We just call its CLI / Python API with a category and a few sane defaults.

What's happening conceptually:
  PaDiM looks at a stack of "good" images of, say, a bottle. It runs each
  through a frozen CNN (a model already pre-trained on millions of natural
  images) and stores the *distribution* of features at each spatial patch.
  At inference time, a new image's feature distance to that distribution
  becomes the anomaly score. Big distance = something looks off.

Why one epoch is fine:
  PaDiM doesn't update CNN weights — it only fits Gaussians. A single pass
  through the training data is sufficient.

Usage:
    python train_padim.py --category bottle --epochs 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("train_padim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", type=str, required=True, help="MVTec category name (e.g. bottle, metal_nut, screw)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--data_root", type=Path, default=Path("./datasets/MVTec"), help="Where MVTec lives (auto-downloaded if missing)")
    p.add_argument("--output", type=Path, default=Path("./results/padim"))
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    try:
        # Anomalib >=1.0 ships a Python API. We import it lazily so the file
        # is still importable on a machine that hasn't installed anomalib yet.
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except Exception as exc:
        logger.error("anomalib is not installed. Run `pip install anomalib` inside the conda env.")
        logger.error("Underlying import error: %s", exc)
        return 2

    logger.info("Preparing MVTec(%s) at %s", args.category, args.data_root)
    datamodule = MVTec(root=args.data_root, category=args.category)

    logger.info("Building PaDiM model")
    model = Padim()

    logger.info("Starting training (%d epoch[s])", args.epochs)
    engine = Engine(max_epochs=args.epochs, default_root_dir=args.output)
    engine.fit(model=model, datamodule=datamodule)

    logger.info("Done. Weights are under %s", args.output)
    logger.info("Next step: run training/indust/export_tflite.py to convert to .tflite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
