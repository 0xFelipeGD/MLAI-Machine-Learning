"""
engine/agro/classifier.py — Fruit quality classifier (good/defective/unripe).

Wraps a MobileNet V2 transfer-learning model exported as TFLite. Like the
detector, it falls back to MOCK MODE if no model is available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INTERPRETER = None
try:
    from ai_edge_litert.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

    _INTERPRETER = _TFLiteInterpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

        _INTERPRETER = _TFLiteInterpreter
    except Exception:
        try:
            from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore

            _INTERPRETER = _TFLiteInterpreter
        except Exception:
            _INTERPRETER = None


class QualityClassifier:
    def __init__(self, num_threads: int = 4) -> None:
        self.num_threads = num_threads
        self._interpreter = None
        self._inp = None
        self._out = None
        self.input_size: Tuple[int, int] = (224, 224)
        self.classes: List[str] = ["defective", "good"]
        self.mock = False

    @staticmethod
    def _load_labels(model_path: Path) -> List[str]:
        labels_path = model_path.parent / f"{model_path.stem}.labels.txt"
        if not labels_path.exists():
            return []
        return [ln.strip() for ln in labels_path.read_text().splitlines() if ln.strip()]

    def load(self, model_path: Path, input_size: Tuple[int, int], classes: List[str]) -> None:
        self.input_size = input_size
        # A .labels.txt next to the .tflite is written by training/agro/train_quality.py
        # and reflects the actual class order the model was trained on (alphabetical,
        # from Keras' ImageDataset.class_names). Prefer it over the caller-supplied
        # list, which may be a stale copy living in config/agro/config.yaml.
        self.classes = self._load_labels(Path(model_path)) or list(classes)
        if _INTERPRETER is None or not Path(model_path).exists():
            logger.warning("QualityClassifier: model unavailable (%s) — MOCK MODE", model_path)
            self.mock = True
            return
        try:
            self._interpreter = _INTERPRETER(model_path=str(model_path), num_threads=self.num_threads)
            self._interpreter.allocate_tensors()
            self._inp = self._interpreter.get_input_details()
            self._out = self._interpreter.get_output_details()
            self.mock = False
            logger.info("Loaded quality classifier %s", model_path)
        except Exception:
            logger.exception("QualityClassifier load failed — MOCK MODE")
            self.mock = True

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if crop_bgr.size == 0:
            return self.classes[0], 0.0
        if self.mock:
            mean = float(np.mean(crop_bgr)) / 255.0
            cls = self.classes[0] if mean > 0.5 else self.classes[1]
            return cls, 0.75

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        dtype = self._inp[0]["dtype"]
        if dtype == np.uint8:
            x = (x * 255).astype(np.uint8)
        self._interpreter.set_tensor(self._inp[0]["index"], x)
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._out[0]["index"])[0]
        # Softmax-ish pick.
        idx = int(np.argmax(out))
        conf = float(np.max(out))
        if conf > 1.0:
            conf = conf / 255.0  # quantized model
        cls = self.classes[idx] if 0 <= idx < len(self.classes) else f"class_{idx}"
        return cls, conf
