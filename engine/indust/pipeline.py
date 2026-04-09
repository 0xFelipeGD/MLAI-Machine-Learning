"""
engine/indust/pipeline.py — Orchestrates one INDUST inspection cycle.

Sequence per frame:
    preprocess  →  PaDiM inference  →  segment  →  measure  →  result

The pipeline owns its detector and config. Switch categories at runtime
via switch_category() — this reloads the corresponding .tflite weights.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from engine import PROJECT_ROOT
from engine.indust.detector import PaDiMDetector
from engine.indust.heatmap import overlay_heatmap
from engine.indust.reporter import IndustResult, build_result
from engine.measurement import largest_contour, measure_object, segment_object
from engine.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class IndustPipeline:
    def __init__(self, config_path: Optional[Path] = None, num_threads: int = 4) -> None:
        cfg_path = config_path or (PROJECT_ROOT / "config" / "indust" / "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self.cfg = (yaml.safe_load(fh) or {}).get("indust", {})

        self.preprocessor = Preprocessor()
        self.detector = PaDiMDetector(num_threads=num_threads)

        self.active_category: str = self.cfg.get("active_category", "bottle")
        self.threshold: float = float(self.cfg.get("default_threshold", 0.5))
        self.save_frames: bool = bool(self.cfg.get("save_frames", True))
        self.save_heatmaps: bool = bool(self.cfg.get("save_heatmaps", True))
        self.captures_dir = PROJECT_ROOT / "data" / "captures" / "indust"
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        self.switch_category(self.active_category)

    # ----------------------------------------------------------- switching
    def switch_category(self, name: str) -> None:
        cats = self.cfg.get("categories", {}) or {}
        if name not in cats:
            raise ValueError(f"Unknown INDUST category '{name}'. Known: {list(cats)}")
        cat = cats[name]
        model_dir = PROJECT_ROOT / self.cfg.get("model_dir", "models/indust")
        model_path = model_dir / cat["model"]
        input_size = tuple(cat.get("input_size", [256, 256]))
        self.threshold = float(cat.get("threshold", self.cfg.get("default_threshold", 0.5)))
        self.detector.load(model_path, input_size)
        self.active_category = name
        logger.info("INDUST switched to category=%s threshold=%.3f", name, self.threshold)

    def list_categories(self) -> list[str]:
        return list((self.cfg.get("categories") or {}).keys())

    def update_config(self, *, category: Optional[str] = None, threshold: Optional[float] = None) -> None:
        if category is not None and category != self.active_category:
            self.switch_category(category)
        if threshold is not None:
            self.threshold = float(threshold)

    # -------------------------------------------------------------- process
    def process(self, frame_bgr: np.ndarray) -> Tuple[IndustResult, np.ndarray]:
        """Run one inspection. Returns (result, overlay_image_bgr)."""
        t0 = time.perf_counter()
        undistorted = self.preprocessor.undistort(frame_bgr)
        score, heatmap = self.detector.infer(undistorted)
        inference_ms = int((time.perf_counter() - t0) * 1000)

        # Measurement (best-effort segmentation, never fatal).
        width_mm = height_mm = area_mm2 = None
        try:
            mask = segment_object(undistorted)
            mc = self.cfg.get("measurement", {}) or {}
            contour = largest_contour(mask, min_area=int(mc.get("min_contour_area", 500)))
            if contour is not None:
                m = measure_object(contour, self.preprocessor.px_per_mm)
                width_mm = m["width_mm"]
                height_mm = m["height_mm"]
                area_mm2 = m["area_mm2"]
        except Exception:
            logger.exception("measurement failed (non-fatal)")

        overlay = overlay_heatmap(undistorted, heatmap, alpha=0.5)

        # Persist
        frame_path = heatmap_path = None
        if self.save_frames or self.save_heatmaps:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            if self.save_frames:
                frame_path = str(self.captures_dir / f"{ts}_frame.jpg")
                cv2.imwrite(frame_path, undistorted)
            if self.save_heatmaps:
                heatmap_path = str(self.captures_dir / f"{ts}_heatmap.jpg")
                cv2.imwrite(heatmap_path, overlay)

        result = build_result(
            category=self.active_category,
            anomaly_score=score,
            threshold=self.threshold,
            inference_ms=inference_ms,
            width_mm=width_mm,
            height_mm=height_mm,
            area_mm2=area_mm2,
            frame_path=frame_path,
            heatmap_path=heatmap_path,
        )
        return result, overlay
