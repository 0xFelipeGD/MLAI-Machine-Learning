#!/usr/bin/env python3
"""
scripts/calibrate_camera.py — Interactive checkerboard calibration wizard.

What you need:
  * A printed checkerboard. Default: 10x7 squares (= 9x6 inner corners),
    25 mm per square. You can override with --pattern and --square.
  * A way to hold/move the board in front of the Pi camera.

How it works:
  1. The script opens the camera and shows a live preview window.
  2. Press SPACE to attempt corner detection on the current frame.
     If corners are detected they will be drawn and the frame is kept.
  3. Repeat from different angles until you have at least 15 samples.
  4. Press ENTER to compute and save calibration.
  5. Press ESC to abort.

Output: config/camera_calibration.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from engine.calibration import (  # noqa: E402
    compute_calibration,
    detect_checkerboard,
    save_calibration,
)
from engine.camera import CameraService  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", type=str, default="9x6", help="inner corners, e.g. 9x6")
    p.add_argument("--square", type=float, default=25.0, help="square size in millimetres")
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--output", type=Path, default=Path("config/camera_calibration.json"))
    p.add_argument("--headless", action="store_true", help="auto-grab samples without preview window")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    cols, rows = (int(x) for x in args.pattern.lower().split("x"))
    pattern_size = (cols, rows)

    print("=" * 60)
    print("MLAI Camera Calibration Wizard")
    print("=" * 60)
    print(f"Pattern: {cols}x{rows} inner corners,  Square: {args.square} mm")
    print(f"Need ≥ {args.min_samples} samples.")
    if not args.headless:
        print("Controls: SPACE=capture  ENTER=finish  ESC=quit")
    print()

    cam = CameraService()
    cam.start()
    samples = []
    image_size = None
    try:
        time.sleep(1.0)
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])

            preview = frame.copy()
            corners = detect_checkerboard(frame, pattern_size)
            ok = corners is not None
            if ok:
                cv2.drawChessboardCorners(preview, pattern_size, corners, ok)
            cv2.putText(
                preview,
                f"samples={len(samples)}/{args.min_samples}  detected={'YES' if ok else 'no'}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if ok else (0, 0, 255),
                2,
            )

            key = -1
            if args.headless:
                if ok:
                    samples.append(corners)
                    print(f"  +{len(samples)} sample (auto)")
                    time.sleep(0.5)
                if len(samples) >= args.min_samples:
                    break
            else:
                cv2.imshow("MLAI Calibration", preview)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    print("aborted.")
                    return 1
                if key == 32 and ok:  # SPACE
                    samples.append(corners)
                    print(f"  +1 sample ({len(samples)})")
                if key == 13 and len(samples) >= args.min_samples:  # ENTER
                    break
    finally:
        cam.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\nComputing calibration from {len(samples)} samples ...")
    data = compute_calibration(samples, image_size, pattern_size, args.square)
    print(f"  reprojection error: {data['reprojection_error']:.4f} px")
    print(f"  px/mm:              {data['px_per_mm']:.4f}")

    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    save_calibration(output, data)
    print(f"  written to:         {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
