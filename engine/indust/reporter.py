"""
engine/indust/reporter.py — INDUST result dataclass and verdict logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class IndustResult:
    """Mirrors the indust_results SQLite schema in INSTRUCTIONS.md §11."""

    timestamp: str
    category: str
    anomaly_score: float
    verdict: str  # PASS | FAIL | WARN
    threshold_used: float
    inference_ms: int
    defect_type: Optional[str] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    area_mm2: Optional[float] = None
    frame_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide_verdict(score: float, threshold: float, warn_margin: float = 0.1) -> str:
    """PASS if clearly below threshold, FAIL if above, WARN if within margin below."""
    if score >= threshold:
        return "FAIL"
    if score >= max(0.0, threshold - warn_margin):
        return "WARN"
    return "PASS"


def build_result(
    *,
    category: str,
    anomaly_score: float,
    threshold: float,
    inference_ms: int,
    width_mm: Optional[float] = None,
    height_mm: Optional[float] = None,
    area_mm2: Optional[float] = None,
    defect_type: Optional[str] = None,
    frame_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
) -> IndustResult:
    return IndustResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        category=category,
        anomaly_score=round(float(anomaly_score), 4),
        verdict=decide_verdict(anomaly_score, threshold),
        threshold_used=float(threshold),
        inference_ms=int(inference_ms),
        defect_type=defect_type,
        width_mm=width_mm,
        height_mm=height_mm,
        area_mm2=area_mm2,
        frame_path=frame_path,
        heatmap_path=heatmap_path,
    )
