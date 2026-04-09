"""
engine/state.py — In-process shared state.

Both the engine main loop and the API process need to know which module
is currently active and what the latest result was. We keep this in a
small process-global object guarded by a lock. The API uses Unix sockets
or shared globals (when running in the same process for development) to
read this state.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import yaml

from engine import PROJECT_ROOT


class EngineState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        cfg_path = PROJECT_ROOT / "config" / "system_config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        else:
            cfg = {}
        self.active_module: str = (cfg.get("system") or {}).get("default_module", "INDUST")
        self.last_indust_result: Optional[dict] = None
        self.last_agro_result: Optional[dict] = None
        self.last_frame_b64: Optional[str] = None
        self.fps: float = 0.0
        self.paused: bool = False
        self.started_at: float = time.time()

    def set_module(self, module: str) -> None:
        if module not in ("INDUST", "AGRO"):
            raise ValueError("module must be INDUST or AGRO")
        with self._lock:
            self.active_module = module

    def update_indust(self, result: dict) -> None:
        with self._lock:
            self.last_indust_result = result

    def update_agro(self, result: dict) -> None:
        with self._lock:
            self.last_agro_result = result

    def update_frame(self, frame_b64: str, fps: float) -> None:
        with self._lock:
            self.last_frame_b64 = frame_b64
            self.fps = fps

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active_module": self.active_module,
                "fps": self.fps,
                "paused": self.paused,
                "uptime_seconds": int(time.time() - self.started_at),
                "last_indust": self.last_indust_result,
                "last_agro": self.last_agro_result,
            }


# Singleton — both engine and API import from here.
STATE = EngineState()
