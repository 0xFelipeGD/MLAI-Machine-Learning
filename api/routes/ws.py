"""
api/routes/ws.py — WebSocket live stream.

ws://<host>:8000/ws/live

Server pushes (every 1/fps seconds):
    {type: "frame",          frame_b64, fps, timestamp}
    {type: "agro_result",    ...}

Client may send:
    {type: "command", action: "pause" | "resume" | "capture"}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.state import STATE

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("WS client connected: %s", ws.client)
    try:
        while True:
            await asyncio.sleep(0.2)  # ~5 fps update cadence
            snap = STATE.snapshot()
            ts = datetime.now(timezone.utc).isoformat()
            if STATE.last_frame_b64:
                await ws.send_json(
                    {
                        "type": "frame",
                        "frame_b64": STATE.last_frame_b64,
                        "fps": snap["fps"],
                        "timestamp": ts,
                    }
                )
            if snap["last_agro"]:
                r = snap["last_agro"]
                await ws.send_json(
                    {
                        "type": "agro_result",
                        "detections": r.get("detections", []),
                        "total_count": r.get("total_detections", 0),
                        "inference_ms": r.get("inference_ms", 0),
                    }
                )

            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                _handle_command(msg)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                raise
            except Exception:
                pass
    except WebSocketDisconnect:
        logger.info("WS client disconnected: %s", ws.client)


def _handle_command(msg: dict) -> None:
    if not isinstance(msg, dict):
        return
    if msg.get("type") != "command":
        return
    action = msg.get("action")
    if action == "pause":
        STATE.paused = True
    elif action == "resume":
        STATE.paused = False
    elif action == "capture":
        pass
