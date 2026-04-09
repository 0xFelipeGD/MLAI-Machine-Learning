"""tests/test_agro_pipeline.py — AGRO pipeline runs end-to-end in mock mode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_agro_mock_run():
    from engine.agro.pipeline import AgroPipeline

    p = AgroPipeline()
    p.save_frames = False
    p.save_annotated = False
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    result, annotated = p.process(frame)
    assert result.total_detections >= 0
    assert annotated.shape == frame.shape
    if result.detections:
        d = result.detections[0]
        assert d.bbox_x2 > d.bbox_x1
        assert d.bbox_y2 > d.bbox_y1
        assert d.fruit_class
