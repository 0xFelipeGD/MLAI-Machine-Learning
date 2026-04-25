"""
engine/camera.py — Camera capture service.

Provides a single CameraService class that abstracts away the difference
between picamera2 (Raspberry Pi) and OpenCV's VideoCapture (development PC).

The Pi 4 uses the libcamera stack via the picamera2 Python library.
On a regular PC we fall back to /dev/video0 through OpenCV.

Usage
-----
    from engine.camera import CameraService

    cam = CameraService()           # picks the best backend automatically
    cam.start()
    frame = cam.read()              # numpy array, BGR, HxWx3
    print("FPS:", cam.get_fps())
    cam.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from engine import PROJECT_ROOT

logger = logging.getLogger(__name__)

# picamera2 only exists on a Raspberry Pi. Wrap the import so this module
# still loads on a development PC where the library is missing.
try:
    from picamera2 import Picamera2  # type: ignore

    _HAS_PICAMERA2 = True
except Exception:  # pragma: no cover - hardware dependent
    Picamera2 = None  # type: ignore
    _HAS_PICAMERA2 = False

try:
    import cv2  # OpenCV is mandatory
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required for engine.camera") from exc


def _load_camera_config() -> dict:
    """Read the [camera] section of system_config.yaml."""
    config_path = PROJECT_ROOT / "config" / "system_config.yaml"
    if not config_path.exists():
        return {"resolution": [640, 480], "fps": 5, "source": "auto"}
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("camera", {})


class CameraService:
    """Thread-safe latest-frame camera grabber.

    A background thread continuously grabs frames and stores the most recent
    one in self._latest. read() returns whatever is current. This way slow
    consumers (the ML pipeline) never block fast producers (the camera).
    """

    def __init__(self, source: str = "auto") -> None:
        cfg = _load_camera_config()
        self.width, self.height = cfg.get("resolution", [640, 480])
        self.target_fps: int = int(cfg.get("fps", 5))
        # source: "auto" | "picamera2" | "opencv"
        self.source = cfg.get("source", source) or "auto"

        # Optional 3x3 color correction matrix applied to every captured RGB
        # frame. NoIR (no-IR-cut-filter) Pi cameras need this to look like
        # normal RGB — without correction, oranges look gray-blue and the
        # COCO-trained detector misses every fruit. See system_config.yaml.
        ccm_raw = cfg.get("color_matrix")
        self._ccm: Optional[np.ndarray] = None
        if ccm_raw and isinstance(ccm_raw, list) and len(ccm_raw) == 3:
            try:
                m = np.array(ccm_raw, dtype=np.float32)
                if m.shape == (3, 3):
                    self._ccm = m

                    logger.info("Camera color matrix enabled:\n%s", m)
            except Exception:
                logger.exception("Invalid camera.color_matrix in config; ignoring")

        self._cap = None  # type: Optional[object]
        self._picam = None  # type: Optional[object]
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # FPS tracking — count frames in a 1 s sliding window.
        self._frame_times: list[float] = []
        self._fps: float = 0.0

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        """Open the camera and start the background grab thread."""
        if self._thread is not None:
            return
        self._open_backend()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()
        logger.info("CameraService started (%dx%d @ %d fps target)", self.width, self.height, self.target_fps)

    def _open_backend(self) -> None:
        backend = self.source
        if backend == "auto":
            backend = "picamera2" if _HAS_PICAMERA2 else "opencv"

        if backend == "picamera2" and _HAS_PICAMERA2:
            self._picam = Picamera2()
            video_cfg = self._picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self._picam.configure(video_cfg)
            self._picam.start()
            time.sleep(0.5)  # warm-up
            return

        # OpenCV fallback (PC dev or Pi without picamera2)
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not self._cap.isOpened():
            raise RuntimeError("OpenCV could not open /dev/video0")

    # ------------------------------------------------------------------- loop
    def _loop(self) -> None:
        period = 1.0 / max(self.target_fps, 1)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self._grab_one()
            except Exception:
                logger.exception("camera grab failed")
                time.sleep(0.2)
                continue
            if frame is not None:
                with self._lock:
                    self._latest = frame
                    now = time.time()
                    self._frame_times.append(now)
                    self._frame_times = [t for t in self._frame_times if now - t <= 1.0]
                    self._fps = float(len(self._frame_times))
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _grab_one(self) -> Optional[np.ndarray]:
        if self._picam is not None:
            # picamera2 returns RGB; convert to BGR for OpenCV consistency.
            rgb = self._picam.capture_array()
            if self._ccm is not None:
                rgb = self._apply_ccm(rgb)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
            if self._ccm is not None:
                # OpenCV gives BGR; CCM is RGB-shaped, so we reorder.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._apply_ccm(rgb)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return frame
        return None

    def _apply_ccm(self, rgb: np.ndarray) -> np.ndarray:
        """Apply the configured 3x3 color matrix to an HxWx3 uint8 RGB frame."""
        f = rgb.astype(np.float32)
        # einsum form is faster than reshape-and-matmul for HxWx3 @ 3x3.
        out = np.einsum("hwc,kc->hwk", f, self._ccm)  # type: ignore[arg-type]
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------- read
    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame (or None if not yet captured)."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def get_fps(self) -> float:
        """Return rolling 1-second FPS."""
        with self._lock:
            return self._fps

    # ------------------------------------------------------------------- stop
    def stop(self) -> None:
        """Stop the grab thread and release the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
            self._picam = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.info("CameraService stopped")

    # context manager helpers
    def __enter__(self) -> "CameraService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    with CameraService() as cam:
        time.sleep(1.0)
        f = cam.read()
        print("Got frame:", None if f is None else f.shape, "fps:", cam.get_fps())
