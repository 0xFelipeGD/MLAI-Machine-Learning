"""
engine/measurement.py — Object segmentation and dimensional measurement.

Pipeline:
  1. Threshold the image to a binary mask (Otsu).
  2. Clean the mask with morphology (close small holes, remove noise).
  3. Find the largest contour — assumed to be the object of interest.
  4. Fit a rotated rectangle to it.
  5. Convert pixel widths/heights to millimetres using px_per_mm.

Accuracy depends entirely on calibration. With a good calibration the
absolute error should be ±2 mm at typical bench distances.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def segment_object(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8 0/255) of the foreground object."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _t, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background is bright (which is common with a lightbox) invert.
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_contour(mask: np.ndarray, min_area: int = 500) -> Optional[np.ndarray]:
    """Return the biggest contour in a mask, or None if nothing significant."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    return contour


def bbox_from_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Axis-aligned (x1, y1, x2, y2)."""
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(x + w), int(y + h)


def measure_object(contour: np.ndarray, px_per_mm: float) -> Dict[str, float]:
    """Compute width/height/area/perimeter in mm using a rotated min-area rectangle."""
    if px_per_mm <= 0:
        px_per_mm = 1.0  # avoid div-by-zero; user should calibrate
    rect = cv2.minAreaRect(contour)
    (_cx, _cy), (w_px, h_px), _angle = rect
    # Always report width <= height for consistency
    if w_px > h_px:
        w_px, h_px = h_px, w_px
    area_px = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, True))
    return {
        "width_mm": round(w_px / px_per_mm, 2),
        "height_mm": round(h_px / px_per_mm, 2),
        "area_mm2": round(area_px / (px_per_mm ** 2), 2),
        "perimeter_mm": round(perimeter_px / px_per_mm, 2),
    }


def estimate_diameter_mm(contour: np.ndarray, px_per_mm: float) -> float:
    """For roughly circular objects (fruit) use minEnclosingCircle."""
    if contour is None or len(contour) < 5:
        return 0.0
    (_cx, _cy), radius_px = cv2.minEnclosingCircle(contour)
    if px_per_mm <= 0:
        px_per_mm = 1.0
    return round(2.0 * radius_px / px_per_mm, 2)
