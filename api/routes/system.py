"""
api/routes/system.py — System health and info endpoints.
"""

from __future__ import annotations

import platform
import socket
import sys
from typing import Optional

from fastapi import APIRouter

from api.schemas import HealthResponse, PauseState, SystemInfo
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
        fps=float(snap["fps"]),
    )


@router.get("/info", response_model=SystemInfo)
async def info() -> SystemInfo:
    return SystemInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
    )


@router.get("/pause", response_model=PauseState)
async def get_pause_state() -> PauseState:
    return PauseState(paused=STATE.paused)


@router.post("/pause", response_model=PauseState)
async def set_pause(req: PauseState) -> PauseState:
    """Pause / resume frame processing. While paused, the engine still
    captures frames (so the live feed keeps streaming) but skips inference
    and DB writes — useful to stop polluting history during setup/tuning."""
    STATE.paused = bool(req.paused)
    return PauseState(paused=STATE.paused)
