"""
engine/agro/detector.py — Fruit detector wrapper.

Wraps a TFLite SSD MobileNet object detector exported with the standard
COCO post-processing op (TFLite_Detection_PostProcess), which produces
four output tensors: boxes, classes, scores, num_detections.

The model we ship is Google's pretrained COCO SSD MobileNet V1 (downloaded
in training/README.md §3) — it knows 90 COCO classes, of which we only
care about apple, banana, and orange. The detector reads the COCO labelmap
file alongside the .tflite (e.g. fruit_detector.labels.txt) at load() time
to build a filter that keeps only those three classes and remaps their
COCO indices to the fruit_classes list from config/agro/config.yaml.

Falls back to MOCK MODE (returns one fake apple in the centre) when no
TFLite runtime or model file is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
        self.input_size: Tuple[int, int] = (300, 300)
        self.classes: List[str] = []
        self.threshold: float = 0.5
        self.max_detections: int = 20
        self.mock = False
        # COCO output index -> our fruit class name (e.g. {52: "banana"}).
        # Built at load() time from the labels file alongside the .tflite.
        # Empty when no labels file is present, in which case detect() falls
        # back to positional indexing into self.classes for compatibility.
        self._coco_to_fruit: Dict[int, str] = {}
        # Diagnostic counter — every Nth call to detect() that produces no
        # filtered output, log the model's top raw scores so we can tell
        # whether the model is silent or just below threshold / off-class.
        self._frame_counter: int = 0

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
        self._coco_to_fruit = self._build_coco_filter(Path(model_path), self.classes)

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
            if self._coco_to_fruit:
                logger.info(
                    "Loaded fruit detector %s; filtering to COCO indices %s",
                    model_path,
                    sorted(self._coco_to_fruit.keys()),
                )
            else:
                logger.info("Loaded fruit detector %s (no COCO filter)", model_path)
        except Exception:
            logger.exception("FruitDetector load failed — MOCK MODE")
            self.mock = True

    # COCO V1 confusion aliases. The pretrained SSD MobileNet V1 is small
    # and routinely mislabels visually-similar food objects between
    # neighbouring classes — round fruits especially get labelled as
    # 'sandwich' (table food), 'donut' (round sweet), or 'sports ball'
    # (round textureless object) depending on the scene. When a user
    # target like 'orange' is in fruit_classes, we silently also catch
    # detections at these alias indices and relabel them to the target.
    # Hack-level — the right fix is a custom-trained detector (see
    # training/README.md). Remove this dict once that's deployed.
    COCO_V1_ALIASES: Dict[str, List[str]] = {
        "orange": ["sandwich", "donut", "sports ball"],
    }

    @classmethod
    def _build_coco_filter(cls, model_path: Path, target_classes: List[str]) -> Dict[int, str]:
        """Read fruit_detector.labels.txt alongside the .tflite to build a
        map of COCO indices -> our target fruit names.

        The COCO labelmap (90 lines) starts with `???` at line 1 (the
        background placeholder), so the model's 0-indexed output indices
        align with `enumerate(lines)` directly: line 1 -> idx 0 -> "???",
        line 53 -> idx 52 -> "banana", line 54 -> idx 53 -> "apple",
        line 56 -> idx 55 -> "orange".

        For each target in `target_classes`, ALSO map any COCO_V1_ALIASES
        entries to the same target name — so the model's misclassifications
        of e.g. 'sandwich' for 'orange' still produce 'orange' detections
        downstream (bbox label, DB, dashboard).

        Returns an empty dict if the labels file is missing, in which case
        detect() falls back to positional indexing for backward compatibility.
        """
        labels_path = model_path.parent / f"{model_path.stem}.labels.txt"
        if not labels_path.exists():
            logger.warning(
                "Detector labels file not found (%s); class filtering disabled — "
                "detections will use raw positional indexing into fruit_classes.",
                labels_path,
            )
            return {}

        # Map lower-cased target name -> original-cased name (so we preserve
        # whatever casing the user wrote in config/agro/config.yaml).
        targets = {c.lower(): c for c in target_classes}

        # Expand each target with its COCO V1 confusion aliases, all
        # pointing to the same target name. e.g. {'orange': 'orange',
        # 'sandwich': 'orange', 'donut': 'orange', 'sports ball': 'orange'}.
        # The user only writes 'orange' in fruit_classes; aliases are
        # transparent.
        alias_to_target: Dict[str, str] = {}
        for low, original in targets.items():
            alias_to_target[low] = original
            for alias in cls.COCO_V1_ALIASES.get(low, []):
                alias_to_target.setdefault(alias.lower(), original)

        mapping: Dict[int, str] = {}
        for idx, line in enumerate(labels_path.read_text().splitlines()):
            name = line.strip().lower()
            if name and name in alias_to_target:
                mapping[idx] = alias_to_target[name]
        return mapping

    # ---------------------------------------------------------------- detect
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        H, W = frame_bgr.shape[:2]
        if self.mock:
            return self._mock_detect(W, H)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Center-crop to square BEFORE resizing to the model's input shape.
        # Without this, a 16:9 phone frame gets squashed horizontally when
        # resized to 1:1, turning round fruits into ovals — SSD MobileNet
        # was trained on aspect-preserving inputs and stops recognizing
        # the distorted shapes. Boxes from the model are in [0, 1] relative
        # to the cropped square, so we add (crop_x, crop_y) when mapping
        # back to original-frame pixels (further down).
        crop_size = min(H, W)
        crop_x = (W - crop_size) // 2
        crop_y = (H - crop_size) // 2
        cropped = rgb[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        resized = cv2.resize(cropped, self.input_size)
        dtype = self._inp[0]["dtype"]
        if dtype == np.uint8:
            x = resized.astype(np.uint8)
        else:
            x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        self._interpreter.set_tensor(self._inp[0]["index"], x)
        self._interpreter.invoke()

        # Standard COCO SSD post-processed output is four tensors:
        #   boxes      shape (1, N, 4)  float32  in [0, 1], (y1, x1, y2, x2)
        #   classes    shape (1, N)     float32  integer-valued indices
        #   scores     shape (1, N)     float32  confidences in [0, 1]
        #   num_det    shape (1,)       float32
        #
        # The order of these tensors in get_output_details() varies between
        # exporters. We identify each by its SHAPE and (for the two 2-D
        # tensors) by VALUE RANGE — scores are bounded in [0, 1], classes
        # are integer-valued indices that go up to 89. Using value range
        # for disambiguation is robust regardless of tensor order.
        boxes = scores = classes = None
        for o in self._out:
            arr = self._interpreter.get_tensor(o["index"])
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr[0]
            elif arr.ndim == 2:
                arr_max = float(np.max(arr)) if arr.size else 0.0
                # scores are in [0, 1]; classes go well above 1 (e.g. 52, 53, 55)
                if arr_max <= 1.0001 and scores is None:
                    scores = arr[0]
                else:
                    classes = arr[0]
            # ndim == 1 -> num_detections; we infer count from len(scores) instead.
        if boxes is None or scores is None or classes is None:
            return []

        detections: List[Detection] = []
        for i in range(min(len(scores), self.max_detections)):
            conf = float(scores[i])
            if conf < self.threshold:
                continue
            cls_idx = int(classes[i])

            # Resolve the class name. If we have a COCO -> fruit mapping (from
            # the labels file), use it as both filter and remap. Otherwise fall
            # back to the legacy positional indexing into self.classes.
            if self._coco_to_fruit:
                cls_name = self._coco_to_fruit.get(cls_idx)
                if cls_name is None:
                    continue  # not one of our target fruits — skip
            else:
                cls_name = (
                    self.classes[cls_idx]
                    if 0 <= cls_idx < len(self.classes)
                    else f"class_{cls_idx}"
                )

            # Boxes are [0, 1] in the cropped square's coordinate system.
            # Map back to original-frame pixels: scale by crop_size, then
            # add the crop offset so the bbox lands on the right place
            # inside the (possibly wider) source frame.
            y1, x1, y2, x2 = boxes[i]
            bbox = (
                int(crop_x + max(0, x1) * crop_size),
                int(crop_y + max(0, y1) * crop_size),
                int(crop_x + min(1, x2) * crop_size),
                int(crop_y + min(1, y2) * crop_size),
            )
            detections.append(Detection(class_name=cls_name, confidence=conf, bbox=bbox))

        # Diagnostic: log the model's top 5 raw scores every ~30 frames
        # (~1s at 25 fps). WARNING level so it actually appears under
        # uvicorn's default log config (root logger sits at WARNING for
        # non-uvicorn modules; INFO would be silenced). Fires regardless
        # of whether anything passed our class/threshold filter — that
        # way we can see when the model fires below the threshold or
        # at the wrong class index.
        self._frame_counter += 1
        if self._frame_counter % 30 == 0 and len(scores) > 0:
            n = min(len(scores), 5)
            top = sorted(
                ((int(classes[i]), float(scores[i])) for i in range(min(len(scores), 10))),
                key=lambda x: -x[1],
            )[:n]
            logger.warning(
                "detector raw top-%d (coco_idx, score): %s | kept=%d | filter=%s | thr=%.2f",
                n,
                top,
                len(detections),
                sorted(self._coco_to_fruit.keys()) if self._coco_to_fruit else "positional",
                self.threshold,
            )
        return detections

    def _mock_detect(self, W: int, H: int) -> List[Detection]:
        # Single fake centre detection so the rest of the pipeline has data.
        cx, cy = W // 2, H // 2
        side = min(W, H) // 4
        bbox = (cx - side, cy - side, cx + side, cy + side)
        cls = self.classes[0] if self.classes else "apple"
        return [Detection(class_name=cls, confidence=0.85, bbox=bbox)]
