#!/usr/bin/env python3
"""
scripts/test_camera.py — Verify the camera is working.

Captures 30 frames, prints the achieved FPS, and saves the last frame as
test.jpg in the current directory. Run from the project root:

    python scripts/test_camera.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running this script directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from engine.camera import CameraService  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print("Opening camera...")
    cam = CameraService()
    cam.start()
    try:
        time.sleep(1.0)
        frame = None
        N = 30
        t0 = time.time()
        for i in range(N):
            f = cam.read()
            if f is not None:
                frame = f
                print(f"  frame {i+1:02d}/{N}: {f.shape}  fps={cam.get_fps():.1f}")
            time.sleep(0.05)
        elapsed = time.time() - t0
        if frame is None:
            print("ERROR: never received a frame from the camera.")
            return 2
        out = Path("test.jpg")
        cv2.imwrite(str(out), frame)
        print(f"Saved {out.resolve()}  ({frame.shape[1]}x{frame.shape[0]})")
        print(f"Effective FPS: {N/elapsed:.2f}")
    finally:
        cam.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
