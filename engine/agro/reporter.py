"""
engine/agro/reporter.py — AGRO result dataclasses.

Mirrors the agro_results / agro_detections SQLite schema in §11.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple


@dataclass
class AgroDetectionResult:
    fruit_class: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    diameter_mm: float = 0.0
    quality: Optional[str] = None
    quality_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgroResult:
    timestamp: str
    total_detections: int
    avg_diameter_mm: float
    inference_ms: int
    detections: List[AgroDetectionResult] = field(default_factory=list)
    frame_path: Optional[str] = None
    annotated_frame_path: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detections"] = [det.to_dict() for det in self.detections]
        return d


def build_result(
    detections: List[AgroDetectionResult],
    *,
    inference_ms: int,
    frame_path: Optional[str] = None,
    annotated_frame_path: Optional[str] = None,
) -> AgroResult:
    diameters = [d.diameter_mm for d in detections if d.diameter_mm > 0]
    avg = round(sum(diameters) / len(diameters), 2) if diameters else 0.0
    return AgroResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_detections=len(detections),
        avg_diameter_mm=avg,
        inference_ms=int(inference_ms),
        detections=list(detections),
        frame_path=frame_path,
        annotated_frame_path=annotated_frame_path,
    )
