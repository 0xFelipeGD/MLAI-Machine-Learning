"""
api/routes/system.py — System health and module switching.
"""

from __future__ import annotations

import platform
import socket
import sys
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import HealthResponse, ModuleSwitchRequest, SystemInfo
from engine.state import STATE

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

router = APIRouter(prefix="/api/system", tags=["system"])


def _read_temperature() -> Optional[float]:
    """Best-effort CPU temperature in °C. Works on Pi via /sys."""
    paths = (
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    )
    for p in paths:
        try:
            with open(p, "r") as fh:
                return round(int(fh.read().strip()) / 1000.0, 1)
        except Exception:
            continue
    if psutil is not None:
        try:
            sensors = psutil.sensors_temperatures()
            for entries in sensors.values():
                for e in entries:
                    if e.current:
                        return round(float(e.current), 1)
        except Exception:
            pass
    return None


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cpu = ram = 0.0
    if psutil is not None:
        cpu = float(psutil.cpu_percent(interval=None))
        ram = float(psutil.virtual_memory().percent)
    snap = STATE.snapshot()
    return HealthResponse(
        status="ok",
        cpu_percent=cpu,
        ram_percent=ram,
        temperature_c=_read_temperature(),
        uptime_seconds=int(snap["uptime_seconds"]),
        active_module=snap["active_module"],
        fps=float(snap["fps"]),
    )


@router.post("/module")
async def switch_module(req: ModuleSwitchRequest) -> dict:
    try:
        STATE.set_module(req.module)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"active_module": STATE.active_module}


@router.get("/info", response_model=SystemInfo)
async def info() -> SystemInfo:
    return SystemInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
    )
