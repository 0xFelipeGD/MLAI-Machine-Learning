"""
api/routes/agro.py — AGRO-only routes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    AgroConfigUpdate,
    AgroDetectionModel,
    AgroHistoryPage,
    AgroResultModel,
    AgroStats,
    AgroStatus,
    SizeHistogramBin,
)
from engine import PROJECT_ROOT
from engine.db import agro_stats, count_agro, get_agro_by_id, list_agro
from engine.state import STATE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agro", tags=["agro"])


def _load_cfg() -> dict:
    p = PROJECT_ROOT / "config" / "agro" / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("agro", {})


def _save_cfg(cfg: dict) -> None:
    p = PROJECT_ROOT / "config" / "agro" / "config.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"agro": cfg}, fh, sort_keys=False)


def _result_to_model(row: dict) -> AgroResultModel:
    detections = [AgroDetectionModel(**d) for d in row.get("detections", []) or []]
    return AgroResultModel(
        id=row.get("id"),
        timestamp=row.get("timestamp"),
        total_detections=int(row.get("total_detections") or 0),
        avg_diameter_mm=float(row.get("avg_diameter_mm") or 0.0),
        inference_ms=int(row.get("inference_ms") or 0),
        frame_path=row.get("frame_path"),
        annotated_frame_path=row.get("annotated_frame_path"),
        detections=detections,
    )


@router.get("/status", response_model=AgroStatus)
async def status() -> AgroStatus:
    cfg = _load_cfg()
    snap = STATE.snapshot()
    last = snap.get("last_agro")
    return AgroStatus(
        running=not snap.get("paused", False),
        fruit_classes=list(cfg.get("fruit_classes", [])),
        detection_threshold=float(cfg.get("detection_threshold", 0.5)),
        last_result=_result_to_model(last) if last else None,
    )


@router.get("/history", response_model=AgroHistoryPage)
async def history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> AgroHistoryPage:
    rows = list_agro(limit=limit, offset=offset)
    items: list[AgroResultModel] = []
    for r in rows:
        full = get_agro_by_id(int(r["id"]))
        items.append(_result_to_model(full or r))
    return AgroHistoryPage(items=items, total=count_agro(), limit=limit, offset=offset)


@router.get("/history/{item_id}", response_model=AgroResultModel)
async def history_detail(item_id: int) -> AgroResultModel:
    row = get_agro_by_id(item_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return _result_to_model(row)


@router.get("/stats", response_model=AgroStats)
async def stats() -> AgroStats:
    s = agro_stats()
    return AgroStats(
        total_detections=int(s.get("total_detections", 0)),
        by_class=s.get("by_class", {}),
        by_quality=s.get("by_quality", {}),
        size_histogram=[SizeHistogramBin(**b) for b in s.get("size_histogram", [])],
    )


@router.post("/config", response_model=AgroStatus)
async def update_config(req: AgroConfigUpdate) -> AgroStatus:
    cfg = _load_cfg()
    if req.detection_threshold is not None:
        cfg["detection_threshold"] = float(req.detection_threshold)
    if req.fruit_classes is not None:
        cfg["fruit_classes"] = list(req.fruit_classes)
    _save_cfg(cfg)
    return await status()
