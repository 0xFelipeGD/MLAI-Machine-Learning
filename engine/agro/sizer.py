"""
engine/agro/sizer.py — Estimate fruit diameter from a cropped image.

Strategy:
  1. Segment the foreground inside the crop (Otsu).
  2. Find the largest contour.
  3. Fit a minimum enclosing circle.
  4. Convert pixel diameter to millimetres via px_per_mm.

If segmentation fails we fall back to the bounding-box width.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from engine.measurement import (
    estimate_diameter_mm as _diameter_from_contour,
    largest_contour,
    segment_object,
)

logger = logging.getLogger(__name__)


def estimate_diameter_mm(
    crop_bgr: np.ndarray,
    px_per_mm: float,
    fallback_bbox_px: Optional[int] = None,
) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    try:
        mask = segment_object(crop_bgr)
        contour = largest_contour(mask, min_area=100)
        if contour is not None:
            return _diameter_from_contour(contour, px_per_mm)
    except Exception:
        logger.exception("contour-based diameter estimation failed")

    if fallback_bbox_px is not None and px_per_mm > 0:
        return round(float(fallback_bbox_px) / px_per_mm, 2)
    return 0.0
