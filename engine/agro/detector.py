"""
engine/agro/detector.py — SSD MobileNet V2 fruit detector wrapper.

Loads a TFLite model exported by TF Model Maker. The standard SSD output
contains four tensors: boxes, classes, scores, num_detections.

Falls back to MOCK MODE (returns one fake apple in the centre) when no
TFLite runtime or model file is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INTERPRETER = None
try:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

    _INTERPRETER = _TFLiteInterpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

        _INTERPRETER = _TFLiteInterpreter
    except Exception:
        _INTERPRETER = None


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image coords


class FruitDetector:
    def __init__(self, num_threads: int = 4) -> None:
        self.num_threads = num_threads
        self._interpreter = None
        self._inp = None
        self._out = None
        self.input_size: Tuple[int, int] = (320, 320)
        self.classes: List[str] = []
        self.threshold: float = 0.5
        self.max_detections: int = 20
        self.mock = False

    # ------------------------------------------------------------------ load
    def load(
        self,
        model_path: Path,
        input_size: Tuple[int, int],
        classes: List[str],
        threshold: float = 0.5,
        max_detections: int = 20,
    ) -> None:
        self.input_size = input_size
        self.classes = list(classes)
        self.threshold = float(threshold)
        self.max_detections = int(max_detections)

        if _INTERPRETER is None or not Path(model_path).exists():
            logger.warning("FruitDetector: model unavailable (%s) — MOCK MODE", model_path)
            self.mock = True
            return
        try:
            self._interpreter = _INTERPRETER(model_path=str(model_path), num_threads=self.num_threads)
            self._interpreter.allocate_tensors()
            self._inp = self._interpreter.get_input_details()
            self._out = self._interpreter.get_output_details()
            self.mock = False
            logger.info("Loaded fruit detector %s", model_path)
        except Exception:
            logger.exception("FruitDetector load failed — MOCK MODE")
            self.mock = True

    # ---------------------------------------------------------------- detect
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        H, W = frame_bgr.shape[:2]
        if self.mock:
            return self._mock_detect(W, H)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        dtype = self._inp[0]["dtype"]
        if dtype == np.uint8:
            x = resized.astype(np.uint8)
        else:
            x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        self._interpreter.set_tensor(self._inp[0]["index"], x)
        self._interpreter.invoke()

        # SSD output order varies; sort by tensor shape.
        outs = {tuple(o["shape"]): self._interpreter.get_tensor(o["index"]) for o in self._out}
        boxes = scores = classes = None
        for shape, arr in outs.items():
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr[0]
            elif arr.ndim == 2:
                if scores is None:
                    scores = arr[0]
                else:
                    classes = arr[0]
        if boxes is None or scores is None or classes is None:
            return []

        detections: List[Detection] = []
        for i in range(min(len(scores), self.max_detections)):
            conf = float(scores[i])
            if conf < self.threshold:
                continue
            cls_idx = int(classes[i])
            cls_name = self.classes[cls_idx] if 0 <= cls_idx < len(self.classes) else f"class_{cls_idx}"
            y1, x1, y2, x2 = boxes[i]
            bbox = (
                int(max(0, x1) * W),
                int(max(0, y1) * H),
                int(min(1, x2) * W),
                int(min(1, y2) * H),
            )
            detections.append(Detection(class_name=cls_name, confidence=conf, bbox=bbox))
        return detections

    def _mock_detect(self, W: int, H: int) -> List[Detection]:
        # Single fake centre detection so the rest of the pipeline has data.
        cx, cy = W // 2, H // 2
        side = min(W, H) // 4
        bbox = (cx - side, cy - side, cx + side, cy + side)
        cls = self.classes[0] if self.classes else "apple"
        return [Detection(class_name=cls, confidence=0.85, bbox=bbox)]
