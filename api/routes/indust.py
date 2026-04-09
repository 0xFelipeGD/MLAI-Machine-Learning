"""
api/routes/indust.py — INDUST-only routes.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas import (
    IndustCategoryInfo,
    IndustConfigUpdate,
    IndustHistoryPage,
    IndustResultModel,
    IndustStatus,
)
from engine import PROJECT_ROOT
from engine.db import count_indust, get_indust_by_id, list_indust
from engine.state import STATE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/indust", tags=["indust"])


def _load_cfg() -> dict:
    p = PROJECT_ROOT / "config" / "indust" / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("indust", {})


def _save_cfg(cfg: dict) -> None:
    p = PROJECT_ROOT / "config" / "indust" / "config.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"indust": cfg}, fh, sort_keys=False)


@router.get("/status", response_model=IndustStatus)
async def status() -> IndustStatus:
    cfg = _load_cfg()
    last = STATE.snapshot().get("last_indust")
    return IndustStatus(
        running=STATE.active_module == "INDUST",
        active_category=cfg.get("active_category", "bottle"),
        threshold=float(cfg.get("default_threshold", 0.5)),
        last_result=IndustResultModel(**last) if last else None,
    )


@router.get("/history", response_model=IndustHistoryPage)
async def history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    verdict: Optional[str] = None,
    category: Optional[str] = None,
) -> IndustHistoryPage:
    items = list_indust(limit=limit, offset=offset, verdict=verdict, category=category)
    total = count_indust(verdict=verdict, category=category)
    return IndustHistoryPage(
        items=[IndustResultModel(**i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/history/export")
async def history_export(verdict: Optional[str] = None, category: Optional[str] = None):
    rows = list_indust(limit=10000, offset=0, verdict=verdict, category=category)
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=indust_history.csv"},
    )


@router.get("/history/{item_id}", response_model=IndustResultModel)
async def history_detail(item_id: int) -> IndustResultModel:
    row = get_indust_by_id(item_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return IndustResultModel(**row)


@router.get("/categories", response_model=List[IndustCategoryInfo])
async def categories() -> List[IndustCategoryInfo]:
    cfg = _load_cfg()
    cats = cfg.get("categories") or {}
    model_dir = PROJECT_ROOT / cfg.get("model_dir", "models/indust")
    out = []
    for name, c in cats.items():
        model_file = model_dir / c.get("model", "")
        out.append(
            IndustCategoryInfo(
                name=name,
                threshold=float(c.get("threshold", cfg.get("default_threshold", 0.5))),
                description=c.get("description"),
                has_model=model_file.exists(),
            )
        )
    return out


@router.post("/config", response_model=IndustStatus)
async def update_config(req: IndustConfigUpdate) -> IndustStatus:
    cfg = _load_cfg()
    cats = cfg.get("categories") or {}
    if req.category is not None:
        if req.category not in cats:
            raise HTTPException(status_code=400, detail=f"unknown category {req.category}")
        cfg["active_category"] = req.category
    if req.threshold is not None:
        active = cfg.get("active_category")
        if active in cats:
            cats[active]["threshold"] = float(req.threshold)
        cfg["default_threshold"] = float(req.threshold)
    _save_cfg(cfg)
    return await status()
