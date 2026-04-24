"""
engine/state.py — In-process shared state.

The engine main loop and the API need to share the latest frame, the
latest AGRO result, and runtime flags like `paused`. Keep it in a small
process-global object guarded by a lock.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class EngineState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.last_agro_result: Optional[dict] = None
        self.last_frame_b64: Optional[str] = None
        self.fps: float = 0.0
        self.paused: bool = False
        self.started_at: float = time.time()

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
                "fps": self.fps,
                "paused": self.paused,
                "uptime_seconds": int(time.time() - self.started_at),
                "last_agro": self.last_agro_result,
            }


# Singleton — both engine and API import from here.
STATE = EngineState()
