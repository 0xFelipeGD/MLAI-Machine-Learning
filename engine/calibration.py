"""
engine/calibration.py — Camera intrinsic calibration via checkerboard.

Why calibrate?
  Real lenses bend light. Straight lines bow inward (barrel) or outward
  (pincushion). To measure objects in millimetres we need to undo this.
  We also recover the scale factor (px per mm) which converts pixel
  measurements to physical units.

How it works:
  1. Print a checkerboard with squares of known size.
  2. Capture 15+ photos showing the board at different angles & positions.
  3. OpenCV finds the corner grid in each image.
  4. cv2.calibrateCamera solves for the intrinsic matrix and distortion.
  5. We store everything in JSON for later use by Preprocessor.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default checkerboard: 9 inner corners horizontally, 6 vertically.
# (A 10x7 squares board has 9x6 inner corners.)
DEFAULT_PATTERN_SIZE: Tuple[int, int] = (9, 6)
DEFAULT_SQUARE_SIZE_MM: float = 25.0


def detect_checkerboard(
    frame: np.ndarray,
    pattern_size: Tuple[int, int] = DEFAULT_PATTERN_SIZE,
) -> Optional[np.ndarray]:
    """Find sub-pixel corner coordinates in a single frame, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined


def _make_object_points(pattern_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    """Create the 3D coordinates of the checkerboard corners (Z=0 plane)."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


def compute_calibration(
    samples: List[np.ndarray],
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
) -> dict:
    """Run OpenCV's calibrateCamera over a list of detected corner sets."""
    if len(samples) < 5:
        raise ValueError(f"Need at least 5 valid samples, got {len(samples)}")

    objp = _make_object_points(pattern_size, square_size_mm)
    object_points = [objp.copy() for _ in samples]

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, samples, image_size, None, None
    )

    # Estimate px/mm from the focal length and a typical working distance.
    # The user can override this empirically with a reference object later.
    fx, fy = K[0, 0], K[1, 1]
    px_per_mm = float((fx + fy) / 2.0 / 250.0)  # heuristic baseline

    return {
        "image_size": list(image_size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "px_per_mm": round(px_per_mm, 4),
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "checkerboard": {
            "pattern_size": list(pattern_size),
            "square_size_mm": square_size_mm,
            "samples_used": len(samples),
        },
        "reprojection_error": round(float(rms), 4),
    }


def save_calibration(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Wrote calibration to %s", path)


def load_calibration(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
