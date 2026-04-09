"""tests/test_measurement.py — Measurement helpers should produce sane values."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_segment_finds_white_square():
    import cv2

    from engine.measurement import bbox_from_contour, largest_contour, measure_object, segment_object

    frame = np.full((400, 400, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame, (100, 80), (300, 320), (240, 240, 240), -1)
    mask = segment_object(frame)
    contour = largest_contour(mask, min_area=1000)
    assert contour is not None
    x1, y1, x2, y2 = bbox_from_contour(contour)
    assert x2 - x1 > 150
    assert y2 - y1 > 150
    m = measure_object(contour, px_per_mm=2.0)
    # 200 px wide / 2 px per mm == 100 mm
    assert 90 < m["width_mm"] < 130
    assert 110 < m["height_mm"] < 140


def test_estimate_diameter_circle():
    import cv2

    from engine.measurement import estimate_diameter_mm, largest_contour, segment_object

    img = np.full((400, 400, 3), 20, dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (255, 255, 255), -1)
    mask = segment_object(img)
    contour = largest_contour(mask)
    d = estimate_diameter_mm(contour, px_per_mm=2.0)
    # diameter ≈ 160 px / 2 = 80 mm (allow ±10)
    assert 70 < d < 90
