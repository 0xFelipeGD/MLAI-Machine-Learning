#!/usr/bin/env python3
"""
training/indust/export_tflite.py — Convert a trained PaDiM model to TFLite.

PaDiM doesn't export to TFLite directly (it's PyTorch-based under Anomalib).
We go through the ONNX intermediate format which works for most layers.

Pipeline:
    Anomalib checkpoint  →  ONNX  →  TensorFlow SavedModel  →  TFLite

This script wraps each step with friendly logging. If a step fails, you
can run the steps manually and pipe the result to the next one.

Usage:
    python export_tflite.py --category bottle --output ../../models/indust/padim_bottle.tflite
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("export_tflite")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=Path, default=Path("./results/padim"))
    p.add_argument("--input_size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from anomalib.deploy import ExportType, OpenVINOInferencer  # noqa: F401
        from anomalib.deploy.export import export_to_torch
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except Exception as exc:
        logger.error("anomalib not installed: %s", exc)
        return 2

    logger.info("Loading checkpoint for category=%s", args.category)
    model = Padim()

    # 1) Export to ONNX (preferred bridge to TFLite)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        onnx_path = tmp_dir / "model.onnx"
        try:
            from anomalib.deploy import ExportType
            engine = Engine(default_root_dir=args.checkpoint_dir)
            engine.export(model=model, export_type=ExportType.ONNX, export_root=tmp_dir)
            # Anomalib writes to <export_root>/weights/onnx/model.onnx
            for candidate in tmp_dir.rglob("*.onnx"):
                onnx_path = candidate
                break
            logger.info("ONNX written to %s", onnx_path)
        except Exception:
            logger.exception("ONNX export failed — try running anomalib export CLI manually")
            return 3

        # 2) ONNX  →  TensorFlow SavedModel
        try:
            import onnx
            from onnx_tf.backend import prepare
        except Exception as exc:
            logger.error("Need onnx and onnx-tf: %s", exc)
            return 4
        tf_dir = tmp_dir / "tf_saved_model"
        logger.info("Converting ONNX → TensorFlow SavedModel")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_dir))

        # 3) SavedModel  →  TFLite
        try:
            import tensorflow as tf
        except Exception as exc:
            logger.error("TensorFlow missing: %s", exc)
            return 5
        logger.info("Converting SavedModel → TFLite")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            tflite_bytes = converter.convert()
        except Exception:
            logger.warning("Default ops failed; retrying with SELECT_TF_OPS")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_bytes = converter.convert()

        args.output.write_bytes(tflite_bytes)
        size_kb = len(tflite_bytes) / 1024
        logger.info("Wrote %s (%.1f KB)", args.output, size_kb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
