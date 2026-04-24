"""
api/schemas.py — Pydantic v2 request/response models.

AGRO-only: fruit detection + quality grading.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["PASS", "FAIL", "WARN"]


# ----------------------------------------------------------------- system
class HealthResponse(BaseModel):
    status: str = "ok"
    cpu_percent: float
    ram_percent: float
    temperature_c: Optional[float] = None
    uptime_seconds: int
    fps: float = 0.0
    version: str = "1.0.0"


class SystemInfo(BaseModel):
    hostname: str
    platform: str
    python_version: str
    project_version: str = "1.0.0"


# ----------------------------------------------------------------- camera
class CameraConfigResponse(BaseModel):
    width: int
    height: int
    fps: int
    source: str
    calibrated: bool
    px_per_mm: Optional[float] = None


class CalibrationStartResponse(BaseModel):
    started: bool
    message: str


# ------------------------------------------------------------------ AGRO
class AgroDetectionModel(BaseModel):
    fruit_class: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    diameter_mm: float = 0.0
    quality: Optional[str] = None
    quality_confidence: Optional[float] = None


class AgroResultModel(BaseModel):
    id: Optional[int] = None
    timestamp: str
    total_detections: int
    avg_diameter_mm: float
    inference_ms: int
    frame_path: Optional[str] = None
    annotated_frame_path: Optional[str] = None
    detections: List[AgroDetectionModel] = []


class AgroStatus(BaseModel):
    running: bool
    fruit_classes: List[str]
    detection_threshold: float
    last_result: Optional[AgroResultModel] = None


class AgroHistoryPage(BaseModel):
    items: List[AgroResultModel]
    total: int
    limit: int
    offset: int


class AgroConfigUpdate(BaseModel):
    detection_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fruit_classes: Optional[List[str]] = None


class SizeHistogramBin(BaseModel):
    range_mm: str
    count: int


class AgroStats(BaseModel):
    total_detections: int
    by_class: dict
    by_quality: dict
    size_histogram: List[SizeHistogramBin]


# ------------------------------------------------------------------ WS
class WSFrameMessage(BaseModel):
    type: Literal["frame"] = "frame"
    frame_b64: str
    fps: float
    timestamp: str


class WSAgroResultMessage(BaseModel):
    type: Literal["agro_result"] = "agro_result"
    detections: List[AgroDetectionModel]
    total_count: int
    inference_ms: int


class WSCommandMessage(BaseModel):
    type: Literal["command"] = "command"
    action: Literal["pause", "resume", "capture"]
