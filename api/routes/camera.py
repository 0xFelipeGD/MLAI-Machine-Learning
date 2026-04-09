"""
api/routes/camera.py — Camera config, calibration trigger, single-frame capture.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import CalibrationStartResponse, CameraConfigResponse
from engine import PROJECT_ROOT
from engine.preprocessor import Preprocessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/camera", tags=["camera"])


def _camera_cfg() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "system_config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("camera", {})


@router.get("/config", response_model=CameraConfigResponse)
async def get_config() -> CameraConfigResponse:
    cam = _camera_cfg()
    pre = Preprocessor()
    width, height = cam.get("resolution", [640, 480])
    return CameraConfigResponse(
        width=int(width),
        height=int(height),
        fps=int(cam.get("fps", 5)),
        source=str(cam.get("source", "auto")),
        calibrated=pre.camera_matrix is not None,
        px_per_mm=pre.px_per_mm if pre.camera_matrix is not None else None,
    )


@router.post("/calibrate", response_model=CalibrationStartResponse)
async def trigger_calibration() -> CalibrationStartResponse:
    return CalibrationStartResponse(
        started=False,
        message="Calibration must be run interactively. SSH into the Pi and run: "
        "python scripts/calibrate_camera.py",
    )


@router.post("/capture")
async def capture():
    """Grab a single JPEG frame and return it."""
    try:
        import cv2

        from engine.camera import CameraService
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"camera unavailable: {exc}")

    cam = CameraService()
    cam.start()
    try:
        import time

        time.sleep(0.5)
        frame = cam.read()
        if frame is None:
            raise HTTPException(status_code=503, detail="camera produced no frame")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise HTTPException(status_code=500, detail="JPEG encode failed")
        return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/jpeg")
    finally:
        cam.stop()
