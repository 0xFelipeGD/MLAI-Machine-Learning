"""
api/routes/ws.py — WebSocket live stream.

ws://<host>:8000/ws/live

Server pushes (every 1/fps seconds):
    {type: "frame",          module, frame_b64, fps, timestamp}
    {type: "indust_result",  ...}
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
        # Producer loop runs while the client is connected.
        while True:
            await asyncio.sleep(0.2)  # ~5 fps update cadence
            snap = STATE.snapshot()
            ts = datetime.now(timezone.utc).isoformat()
            if STATE.last_frame_b64:
                await ws.send_json(
                    {
                        "type": "frame",
                        "module": snap["active_module"],
                        "frame_b64": STATE.last_frame_b64,
                        "fps": snap["fps"],
                        "timestamp": ts,
                    }
                )
            if snap["active_module"] == "INDUST" and snap["last_indust"]:
                r = snap["last_indust"]
                await ws.send_json(
                    {
                        "type": "indust_result",
                        "verdict": r["verdict"],
                        "anomaly_score": r["anomaly_score"],
                        "measurements": {
                            "width_mm": r.get("width_mm"),
                            "height_mm": r.get("height_mm"),
                            "area_mm2": r.get("area_mm2"),
                        },
                        "defect_type": r.get("defect_type"),
                        "inference_ms": r.get("inference_ms", 0),
                    }
                )
            elif snap["active_module"] == "AGRO" and snap["last_agro"]:
                r = snap["last_agro"]
                await ws.send_json(
                    {
                        "type": "agro_result",
                        "detections": r.get("detections", []),
                        "total_count": r.get("total_detections", 0),
                        "inference_ms": r.get("inference_ms", 0),
                    }
                )

            # Drain inbound commands without blocking the producer.
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
        # No-op stub: pipelines already auto-save when configured.
        pass
