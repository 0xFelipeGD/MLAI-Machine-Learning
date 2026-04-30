"""
scripts/probe_detector.py — Diagnostic for the fruit detector model.

Runs the detector against a synthetic orange disc on a white background
(textbook 'easy' case) AND against the latest captured frame from the
engine, printing the top-5 raw model outputs with their corresponding
label names. Lets us confirm whether the COCO SSD model uses sparse
1-indexed (matching the labels file) or compact 0-indexed (off-by-one)
output convention — and whether it actually recognises an obvious orange.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "agro" / "fruit_detector.tflite"
LABELS_FILE = ROOT / "models" / "agro" / "fruit_detector.labels.txt"
CAPTURES = ROOT / "data" / "captures" / "agro"

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


def parse_outputs(interpreter):
    """Return (boxes, scores, classes) regardless of output tensor order."""
    boxes = scores = classes = None
    for d in interpreter.get_output_details():
        arr = interpreter.get_tensor(d["index"])
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
        elif arr.ndim == 2:
            amax = float(np.max(arr)) if arr.size else 0.0
            if amax <= 1.0001 and scores is None:
                scores = arr[0]
            else:
                classes = arr[0]
    return boxes, scores, classes


def top5(scores, classes, labels):
    pairs = [(int(classes[i]), float(scores[i])) for i in range(min(10, len(scores)))]
    pairs.sort(key=lambda x: -x[1])
    out = []
    for idx, s in pairs[:5]:
        name = labels[idx] if 0 <= idx < len(labels) else "?"
        out.append(f"  idx={idx:3d} score={s:.3f}  labels[{idx}]={name!r}")
    return "\n".join(out)


def main() -> int:
    labels = LABELS_FILE.read_text().splitlines()
    print(f"Labels file has {len(labels)} entries; orange is at line/idx",
          *((i, n) for i, n in enumerate(labels) if n.lower() == "orange"))

    interp = Interpreter(model_path=str(MODEL))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    H, W = inp["shape"][1], inp["shape"][2]
    print(f"Model expects input {W}x{H} dtype={inp['dtype'].__name__}")
    print()

    # === Test 1: synthetic orange disc on white BG ===
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    cv2.circle(img, (W // 2, H // 2), min(W, H) // 4, (50, 140, 230), -1)  # BGR orange-ish
    interp.set_tensor(inp["index"], np.expand_dims(img, 0).astype(inp["dtype"]))
    interp.invoke()
    _, scores, classes = parse_outputs(interp)
    print("=== TEST 1: synthetic orange disc (BGR=50,140,230) on white BG ===")
    print(top5(scores, classes, labels))
    print()

    # === Test 2: latest real captured frame from engine ===
    if CAPTURES.exists():
        captures = sorted(CAPTURES.glob("*_frame.jpg"), key=lambda p: p.stat().st_mtime)
        if captures:
            latest = captures[-1]
            print(f"=== TEST 2: latest engine capture {latest.name} ===")
            frame = cv2.imread(str(latest))
            if frame is None:
                print("  (could not read file)")
            else:
                # Replicate the engine's preprocessing (BGR→RGB + center-crop + resize)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                cs = min(h, w)
                cx, cy = (w - cs) // 2, (h - cs) // 2
                cropped = rgb[cy:cy + cs, cx:cx + cs]
                resized = cv2.resize(cropped, (W, H))
                interp.set_tensor(inp["index"], np.expand_dims(resized, 0).astype(inp["dtype"]))
                interp.invoke()
                _, scores, classes = parse_outputs(interp)
                print(top5(scores, classes, labels))
        else:
            print("=== TEST 2: no captures yet ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
