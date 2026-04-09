"""
engine/agro/pipeline.py — Orchestrates one AGRO inspection cycle.

Sequence per frame:
  preprocess
    → detector.detect()         (find each fruit's bbox)
    → for each detection:
        crop the bbox
        classifier.classify()   (good / defective / unripe)
        sizer.estimate_diameter_mm()
    → aggregate into AgroResult
    → draw annotations and persist
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
from engine.agro.classifier import QualityClassifier
from engine.agro.detector import FruitDetector
from engine.agro.reporter import AgroDetectionResult, AgroResult, build_result
from engine.agro.sizer import estimate_diameter_mm
from engine.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


_QUALITY_COLORS = {
    "good": (0, 230, 118),       # green
    "defective": (23, 23, 255),  # red
    "unripe": (0, 171, 255),     # amber
}


class AgroPipeline:
    def __init__(self, config_path: Optional[Path] = None, num_threads: int = 4) -> None:
        cfg_path = config_path or (PROJECT_ROOT / "config" / "agro" / "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self.cfg = (yaml.safe_load(fh) or {}).get("agro", {})

        self.preprocessor = Preprocessor()
        self.detector = FruitDetector(num_threads=num_threads)
        self.classifier = QualityClassifier(num_threads=num_threads)

        self.fruit_classes = list(self.cfg.get("fruit_classes", ["apple", "orange", "tomato"]))
        self.quality_classes = list(self.cfg.get("quality_classes", ["good", "defective", "unripe"]))
        self.detection_threshold = float(self.cfg.get("detection_threshold", 0.5))

        det_size = tuple(self.cfg.get("detector_input_size", [320, 320]))
        qual_size = tuple(self.cfg.get("quality_input_size", [224, 224]))
        det_model = PROJECT_ROOT / self.cfg.get("detector_model", "models/agro/fruit_detector.tflite")
        qual_model = PROJECT_ROOT / self.cfg.get("quality_model", "models/agro/fruit_quality.tflite")

        self.detector.load(
            det_model,
            det_size,
            self.fruit_classes,
            threshold=self.detection_threshold,
            max_detections=int(self.cfg.get("max_detections", 20)),
        )
        self.classifier.load(qual_model, qual_size, self.quality_classes)

        self.save_frames: bool = bool(self.cfg.get("save_frames", True))
        self.save_annotated: bool = bool(self.cfg.get("save_annotated", True))
        self.captures_dir = PROJECT_ROOT / "data" / "captures" / "agro"
        self.captures_dir.mkdir(parents=True, exist_ok=True)

    def update_config(
        self,
        *,
        threshold: Optional[float] = None,
        fruit_classes: Optional[list] = None,
    ) -> None:
        if threshold is not None:
            self.detection_threshold = float(threshold)
            self.detector.threshold = float(threshold)
        if fruit_classes:
            self.fruit_classes = list(fruit_classes)
            self.detector.classes = list(fruit_classes)

    # ------------------------------------------------------------- process
    def process(self, frame_bgr: np.ndarray) -> Tuple[AgroResult, np.ndarray]:
        t0 = time.perf_counter()
        undistorted = self.preprocessor.undistort(frame_bgr)
        detections = self.detector.detect(undistorted)

        results: list[AgroDetectionResult] = []
        annotated = undistorted.copy()
        px_per_mm = self.preprocessor.px_per_mm

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            crop = undistorted[max(0, y1):y2, max(0, x1):x2]
            quality_label, quality_conf = self.classifier.classify(crop)
            diameter_mm = estimate_diameter_mm(
                crop,
                px_per_mm,
                fallback_bbox_px=max(1, x2 - x1),
            )
            results.append(
                AgroDetectionResult(
                    fruit_class=det.class_name,
                    confidence=round(det.confidence, 4),
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    diameter_mm=diameter_mm,
                    quality=quality_label,
                    quality_confidence=round(quality_conf, 4),
                )
            )
            color = _QUALITY_COLORS.get(quality_label or "good", (0, 230, 118))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.0%} | {quality_label} | {diameter_mm:.0f}mm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 17, 23),
                1,
                cv2.LINE_AA,
            )

        inference_ms = int((time.perf_counter() - t0) * 1000)

        frame_path = annotated_path = None
        if self.save_frames or self.save_annotated:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            if self.save_frames:
                frame_path = str(self.captures_dir / f"{ts}_frame.jpg")
                cv2.imwrite(frame_path, undistorted)
            if self.save_annotated:
                annotated_path = str(self.captures_dir / f"{ts}_annotated.jpg")
                cv2.imwrite(annotated_path, annotated)

        result = build_result(
            results,
            inference_ms=inference_ms,
            frame_path=frame_path,
            annotated_frame_path=annotated_path,
        )
        return result, annotated
