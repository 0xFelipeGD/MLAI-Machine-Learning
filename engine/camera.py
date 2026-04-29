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
import re
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


_URL_RE = re.compile(r"^(https?|rtsp)://", re.IGNORECASE)


def _resolve_backend(source: str, has_picamera2: bool) -> str:
    """Map a config `source` value to a concrete backend kind.

    Returns one of: "stream" | "picamera2" | "opencv".

    A URL (http://, https://, rtsp://) resolves to "stream".
    "auto" picks picamera2 if available, otherwise opencv.
    "picamera2" and "opencv" are returned as-is.

    Raises ValueError for any other input (typo in config, empty string, None).
    Failing fast here gives a clear error at config-parse time instead of a
    confusing silent fallback in _open_backend later.
    """
    if isinstance(source, str) and _URL_RE.match(source):
        return "stream"
    if source == "auto":
        return "picamera2" if has_picamera2 else "opencv"
    if source in ("picamera2", "opencv"):
        return source
    raise ValueError(
        f"camera.source={source!r} is not a URL and not one of "
        "{'auto', 'picamera2', 'opencv'}; check config/system_config.yaml"
    )


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
        # frame. The right way to fix NoIR colour cast is the tuning_file
        # below — the CCM is a fine-tuning escape hatch. Skipped entirely
        # when the matrix is (close to) identity so we don't spend ~2ms
        # per frame on a no-op multiply.
        ccm_raw = cfg.get("color_matrix")
        self._ccm: Optional[np.ndarray] = None
        if ccm_raw and isinstance(ccm_raw, list) and len(ccm_raw) == 3:
            try:
                m = np.array(ccm_raw, dtype=np.float32)
                if m.shape == (3, 3) and not np.allclose(m, np.eye(3, dtype=np.float32)):
                    self._ccm = m
                    logger.info("Camera color matrix enabled:\n%s", m)
            except Exception:
                logger.exception("Invalid camera.color_matrix in config; ignoring")

        # Optional libcamera tuning file override. By default, libcamera
        # auto-detects a NoIR sensor (imx708_noir) and loads the IR-tuned
        # JSON which is built for night vision and produces gray/blue
        # daytime colours. Forcing the regular imx708.json tuning on a
        # NoIR sensor tells the IPA to behave as an ordinary RGB camera —
        # the AWB algorithm switches to visible-light targets and reds
        # come back. None = let libcamera pick (the default + broken path).
        self._tuning_file: Optional[str] = cfg.get("tuning_file") or None

        # picamera2 controls applied at sensor level. Setting AwbEnable=False +
        # manual ColourGains is needed when the auto-WB still misbehaves;
        # leave awb_mode='auto' if the tuning_file override already gives
        # acceptable colours.
        self._picam_controls: dict = {}
        awb_mode = (cfg.get("awb_mode") or "auto").lower()
        if awb_mode in ("manual", "off", "disabled", "none"):
            gains = cfg.get("colour_gains") or [2.0, 0.7]
            try:
                r_gain, b_gain = float(gains[0]), float(gains[1])
                self._picam_controls["AwbEnable"] = False
                self._picam_controls["ColourGains"] = (r_gain, b_gain)
                logger.info("Manual ColourGains: red=%.2f blue=%.2f (AWB disabled)", r_gain, b_gain)
            except Exception:
                logger.exception("Invalid camera.colour_gains in config; ignoring")

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
        backend = _resolve_backend(self.source, _HAS_PICAMERA2)

        if backend == "picamera2" and _HAS_PICAMERA2:
            picam_kwargs = {}
            if self._tuning_file:
                # Try to load a custom tuning file. Most common use:
                # tuning_file: "imx708.json" on a NoIR sensor to force
                # regular-RGB tuning (fixes the daytime blue cast).
                try:
                    tuning = Picamera2.load_tuning_file(self._tuning_file)  # type: ignore[attr-defined]
                    picam_kwargs["tuning"] = tuning
                    logger.info("Loaded custom tuning: %s", self._tuning_file)
                except Exception:
                    logger.exception(
                        "Failed to load tuning_file '%s'; falling back to default tuning",
                        self._tuning_file,
                    )
            self._picam = Picamera2(**picam_kwargs)
            cfg_kwargs = {
                "main": {"size": (self.width, self.height), "format": "RGB888"},
            }
            if self._picam_controls:
                cfg_kwargs["controls"] = dict(self._picam_controls)
            video_cfg = self._picam.create_video_configuration(**cfg_kwargs)
            self._picam.configure(video_cfg)
            self._picam.start()
            if self._picam_controls:
                try:
                    self._picam.set_controls(self._picam_controls)
                except Exception:
                    logger.exception("Failed to apply picamera2 controls: %s", self._picam_controls)
            time.sleep(0.5)  # warm-up
            return

        if backend == "stream":
            # Network camera (phone, IP webcam, RTSP). The OpenCV grab path
            # is identical to the local /dev/video0 case below — only the
            # VideoCapture constructor argument changes. CAP_FFMPEG makes
            # the demuxer choice deterministic across machines that have
            # both GStreamer and FFmpeg backends compiled in.
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                raise RuntimeError(f"OpenCV could not open stream: {self.source}")
            logger.info("CameraService stream backend opened: %s", self.source)
            return

        # OpenCV local fallback (PC dev or Pi without picamera2)
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

    # --------------------------------------------------------- live tuning
    def update_gains(self, red: float, blue: float, awb_auto: bool = False) -> None:
        """Live update of picamera2 ColourGains (no restart).

        When awb_auto is True, hands control back to the camera's auto-WB
        algorithm (red/blue values are still recorded for the next time
        the user toggles back to manual). When False, locks to the gains.
        """
        red = float(np.clip(red, 0.1, 8.0))
        blue = float(np.clip(blue, 0.1, 8.0))
        if awb_auto:
            self._picam_controls.pop("AwbEnable", None)
            self._picam_controls.pop("ColourGains", None)
            if self._picam is not None:
                try:
                    self._picam.set_controls({"AwbEnable": True})
                    logger.info("AWB switched to AUTO (gains follow the tuning file)")
                except Exception:
                    logger.exception("set_controls failed (auto)")
            # Stash the gains under a non-applied key so the dashboard
            # can show what manual values would be used if toggled back.
            self._picam_controls["_pending_gains"] = (red, blue)
            return

        self._picam_controls["AwbEnable"] = False
        self._picam_controls["ColourGains"] = (red, blue)
        self._picam_controls.pop("_pending_gains", None)
        if self._picam is not None:
            try:
                self._picam.set_controls({"AwbEnable": False, "ColourGains": (red, blue)})
                logger.info("Live update ColourGains: red=%.2f blue=%.2f (AWB disabled)", red, blue)
            except Exception:
                logger.exception("set_controls failed (manual)")

    def update_color_matrix(self, matrix: list) -> None:
        """Live update of the post-capture CCM. matrix is a 3x3 nested list.
        Identity matrices clear the CCM entirely (skipping the per-frame
        multiplication)."""
        try:
            m = np.array(matrix, dtype=np.float32)
            if m.shape != (3, 3):
                raise ValueError(f"shape must be 3x3, got {m.shape}")
            if np.allclose(m, np.eye(3, dtype=np.float32)):
                self._ccm = None
                logger.info("Live update CCM: identity (CCM disabled)")
            else:
                self._ccm = m
                logger.info("Live update CCM:\n%s", m)
        except Exception:
            logger.exception("update_color_matrix failed")

    def get_controls(self) -> dict:
        """Return the current camera controls so the dashboard can show them."""
        gains = (
            self._picam_controls.get("ColourGains")
            or self._picam_controls.get("_pending_gains")
            or (1.0, 1.0)
        )
        # AWB is auto unless we explicitly disabled it.
        awb_auto = self._picam_controls.get("AwbEnable") is not False
        return {
            "awb_auto": awb_auto,
            "red_gain": float(gains[0]),
            "blue_gain": float(gains[1]),
            "color_matrix": self._ccm.tolist() if self._ccm is not None else None,
        }

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
