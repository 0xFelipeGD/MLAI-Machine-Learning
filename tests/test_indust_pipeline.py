"""tests/test_indust_pipeline.py — INDUST pipeline runs end-to-end in mock mode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_indust_mock_run():
    from engine.indust.pipeline import IndustPipeline

    p = IndustPipeline()
    p.save_frames = False
    p.save_heatmaps = False
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    result, overlay = p.process(frame)
    assert result.verdict in ("PASS", "FAIL", "WARN")
    assert 0.0 <= result.anomaly_score <= 1.0
    assert overlay.shape == frame.shape
    assert result.inference_ms >= 0


def test_indust_verdict_logic():
    from engine.indust.reporter import decide_verdict

    assert decide_verdict(0.1, 0.5) == "PASS"
    assert decide_verdict(0.42, 0.5) == "WARN"
    assert decide_verdict(0.6, 0.5) == "FAIL"
