"""
engine/indust/detector.py — PaDiM TFLite anomaly detector.

PaDiM (Patch Distribution Modeling) is an anomaly detection method:
training learns a Gaussian distribution per spatial patch from "good"
images. At inference time the Mahalanobis distance to this distribution
gives an anomaly score per pixel — high distance = abnormal.

This wrapper hides the boring TFLite plumbing. If no model file is found
(or tflite_runtime is not installed) it operates in MOCK MODE returning
deterministic-but-fake outputs so the rest of the system can still run
during development.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Prefer the lightweight tflite_runtime; fall back to full TF; allow neither.
_INTERPRETER = None
try:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

    _INTERPRETER = _TFLiteInterpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

        _INTERPRETER = _TFLiteInterpreter
    except Exception:  # pragma: no cover
        _INTERPRETER = None


class PaDiMDetector:
    """Loads a PaDiM .tflite model and produces (anomaly_score, heatmap)."""

    def __init__(self, num_threads: int = 4) -> None:
        self.num_threads = num_threads
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self.input_size: Tuple[int, int] = (256, 256)
        self.model_path: Optional[Path] = None
        self.mock = False

    # ------------------------------------------------------------------ load
    def load(self, model_path: Path, input_size: Tuple[int, int] = (256, 256)) -> None:
        self.input_size = input_size
        self.model_path = Path(model_path)

        if _INTERPRETER is None:
            logger.warning("No TFLite runtime available — PaDiMDetector running in MOCK MODE")
            self.mock = True
            return
        if not self.model_path.exists():
            logger.warning("Model file %s not found — MOCK MODE", self.model_path)
            self.mock = True
            return

        try:
            self._interpreter = _INTERPRETER(model_path=str(self.model_path), num_threads=self.num_threads)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self.mock = False
            logger.info("Loaded PaDiM model %s", self.model_path)
        except Exception:
            logger.exception("Failed to load TFLite model — falling back to MOCK MODE")
            self.mock = True

    # ----------------------------------------------------------------- infer
    def infer(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run inference. Returns (anomaly_score in [0,1], heatmap HxW float32 in [0,1])."""
        h, w = self.input_size[1], self.input_size[0]
        if self.mock:
            return self._mock_infer(frame_bgr)

        # Standard preprocessing for PaDiM exports: BGR -> RGB, resize, /255.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        inp = self._input_details[0]
        # Some exports take CHW; transpose if so.
        if list(inp["shape"])[1] == 3:
            x = np.transpose(x, (0, 3, 1, 2))
        self._interpreter.set_tensor(inp["index"], x.astype(inp["dtype"]))
        self._interpreter.invoke()

        outputs = [self._interpreter.get_tensor(o["index"]) for o in self._output_details]
        # Heuristic: pick the largest 2D-shaped output as the heatmap, scalar as score.
        heatmap = None
        score = None
        for arr in outputs:
            squeezed = np.squeeze(arr)
            if squeezed.ndim >= 2 and heatmap is None:
                heatmap = squeezed.astype(np.float32)
            elif squeezed.ndim == 0 and score is None:
                score = float(squeezed)
        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)
        else:
            heatmap = cv2.resize(heatmap, (w, h))
            mn, mx = float(heatmap.min()), float(heatmap.max())
            if mx > mn:
                heatmap = (heatmap - mn) / (mx - mn)
        if score is None:
            score = float(heatmap.max())
        score = max(0.0, min(1.0, score))
        return score, heatmap

    def _mock_infer(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Generate a deterministic but plausible-looking output for development."""
        h, w = self.input_size[1], self.input_size[0]
        gray = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2GRAY)
        # Use gradient magnitude as a fake "anomaly" signal so heatmaps look interesting.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.GaussianBlur(mag, (15, 15), 0)
        mn, mx = float(mag.min()), float(mag.max())
        if mx > mn:
            mag = (mag - mn) / (mx - mn)
        score = float(np.percentile(mag, 95))
        return score, mag.astype(np.float32)
