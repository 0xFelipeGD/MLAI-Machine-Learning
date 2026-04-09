"""
engine/preprocessor.py — Image preprocessing for ML inference.

Every ML model expects inputs in a very specific shape and value range.
This module provides helpers to:

  * undistort frames using calibration data,
  * resize to model input size,
  * normalize pixel values to [0, 1] or by ImageNet mean/std,
  * convert to a tensor (HWC or CHW layout, batch-prefixed).

A Preprocessor instance loads camera_calibration.json once at startup and
applies undistortion to every incoming frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from engine import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Standard ImageNet normalization — used by MobileNet V2 and most TF models
# trained on ImageNet. Stored here so AGRO classifier can reuse them.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Preprocessor:
    """Stateful image preprocessor (holds calibration matrices)."""

    def __init__(self, calibration_path: Optional[Path] = None) -> None:
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.px_per_mm: float = 1.0  # 1 px = 1 mm fallback (no calibration)

        cal_path = calibration_path or (PROJECT_ROOT / "config" / "camera_calibration.json")
        self._load_calibration(cal_path)

    def _load_calibration(self, path: Path) -> None:
        if not path.exists():
            logger.warning("No calibration file at %s — undistortion disabled", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            self.dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
            self.px_per_mm = float(data.get("px_per_mm", 1.0))
            logger.info("Loaded camera calibration from %s (px/mm=%.3f)", path, self.px_per_mm)
        except Exception:
            logger.exception("Failed to read calibration file %s", path)

    # ------------------------------------------------------------------ ops
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Remove lens distortion. Returns the original if not calibrated."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    @staticmethod
    def resize(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize to (width, height). Bilinear interpolation."""
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize(
        frame: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert uint8 BGR -> float32 RGB in [0,1], optionally mean/std normalized."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if mean is not None and std is not None:
            rgb = (rgb - mean) / std
        return rgb

    @staticmethod
    def to_tensor(frame: np.ndarray, layout: str = "HWC", batch: bool = True) -> np.ndarray:
        """Add batch dim and optionally transpose to CHW."""
        if layout.upper() == "CHW":
            frame = np.transpose(frame, (2, 0, 1))
        if batch:
            frame = np.expand_dims(frame, axis=0)
        return np.ascontiguousarray(frame, dtype=np.float32)

    # ----------------------------------------------------------- composite
    def prepare_for_model(
        self,
        frame: np.ndarray,
        input_size: Tuple[int, int],
        normalize_imagenet: bool = False,
        layout: str = "HWC",
    ) -> np.ndarray:
        """One-shot: undistort -> resize -> normalize -> tensor."""
        f = self.undistort(frame)
        f = self.resize(f, input_size)
        if normalize_imagenet:
            f = self.normalize(f, IMAGENET_MEAN, IMAGENET_STD)
        else:
            f = self.normalize(f)
        return self.to_tensor(f, layout=layout, batch=True)
