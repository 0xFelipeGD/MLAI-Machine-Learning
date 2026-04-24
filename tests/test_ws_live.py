"""tests/test_ws_live.py — Contract tests for the /ws/live WebSocket.

What this proves
----------------
The payloads the FastAPI server emits on /ws/live match the shape the
frontend (LiveFeed.tsx + AgroDashboard.tsx) expects to consume. If these
tests pass, the dashboard will at least *receive* data in the correct
format — i.e. it will not silently go blank or skip drawing bboxes
because of a renamed field.

What this does NOT prove
------------------------
- Actual browser render (the <img> element updating, bboxes drawn in the
  DOM). That requires Playwright / headless browser — see the E2E plan.
- End-to-end non-freezing behaviour under sustained load. That is what
  the follow-up "cadence" test is for (pytest spinning uvicorn + a
  synthetic engine producer over a few seconds).

Scope here is cheap and fast (<1s total).
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------- helpers

# Minimal valid JPEG header+footer (SOI/JFIF/EOI). Not a real image, but
# enough for a contract test — we never decode pixels, we only assert the
# round-trip bytes start with the JPEG magic number.
_FAKE_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
)
_FAKE_JPEG_B64 = base64.b64encode(_FAKE_JPEG).decode()


def _reset_state() -> None:
    """Singletons leak between tests — wipe before each."""
    from engine.state import STATE

    STATE.last_frame_b64 = None
    STATE.last_agro_result = None
    STATE.fps = 0.0
    STATE.paused = False


# --------------------------------------------------------------- fixture

@pytest.fixture
def client():
    """Fresh TestClient with a wiped STATE."""
    from fastapi.testclient import TestClient

    from api.main import create_app
    from engine.db import init_db

    _reset_state()
    init_db(ROOT / "data" / "test_mlai.db")
    app = create_app()
    with TestClient(app) as c:
        yield c
    _reset_state()


# ---------------------------------------------------------------- tests

def test_ws_emits_frame_when_state_has_frame(client):
    """With a frame in STATE, WS sends {type: 'frame', frame_b64, fps, timestamp}.

    Fields pulled straight out of the TypeScript union in web/lib/types.ts:
        { type: "frame"; frame_b64: string; fps: number; timestamp: string }
    """
    from engine.state import STATE

    STATE.update_frame(_FAKE_JPEG_B64, fps=5.0)

    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "frame"
    assert msg["frame_b64"] == _FAKE_JPEG_B64
    assert msg["fps"] == 5.0
    assert "timestamp" in msg and isinstance(msg["timestamp"], str)

    # The base64 payload must round-trip to a valid JPEG byte stream —
    # otherwise the <img src="data:image/jpeg;base64,..."> will not render.
    raw = base64.b64decode(msg["frame_b64"])
    assert raw.startswith(b"\xff\xd8\xff"), "payload is not a JPEG — browser <img> will break"


def test_ws_emits_agro_result_with_bbox_and_quality(client):
    """With an AGRO result in STATE, WS sends the detection shape the frontend expects.

    Every field asserted below is consumed by AgroDashboard.tsx / LiveFeed.tsx
    to draw the bbox overlay and the quality label. Losing any of them would
    silently degrade the dashboard (no box, no label, no diameter readout).
    """
    from engine.state import STATE

    STATE.update_agro(
        {
            "total_detections": 1,
            "inference_ms": 42,
            "detections": [
                {
                    "fruit_class": "apple",
                    "confidence": 0.87,
                    "bbox_x1": 120,
                    "bbox_y1": 80,
                    "bbox_x2": 280,
                    "bbox_y2": 260,
                    "diameter_mm": 74.5,
                    "quality": "good",
                    "quality_confidence": 0.91,
                }
            ],
        }
    )

    # No frame populated → only an agro_result should arrive.
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "agro_result"
    assert msg["total_count"] == 1
    assert msg["inference_ms"] == 42
    assert len(msg["detections"]) == 1

    det = msg["detections"][0]
    required = (
        "fruit_class",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "diameter_mm",
        "quality",
    )
    for field in required:
        assert field in det, f"missing '{field}' — overlay would not render correctly"

    assert det["fruit_class"] == "apple"
    assert det["quality"] == "good"
    assert det["bbox_x2"] > det["bbox_x1"], "degenerate bbox (x2 <= x1)"
    assert det["bbox_y2"] > det["bbox_y1"], "degenerate bbox (y2 <= y1)"


def test_ws_accepts_pause_command(client):
    """Frontend can send {type:'command', action:'pause'} and STATE flips to paused=True.

    This is the minimum that proves the control channel is wired — the user
    can toggle the engine from the dashboard without the WS rejecting input.
    """
    from engine.state import STATE

    STATE.update_frame(_FAKE_JPEG_B64, fps=5.0)  # so the loop has something to send
    assert STATE.paused is False

    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "command", "action": "pause"})
        # Drain at least one server frame so the server has had a chance to
        # process our inbound message (the loop checks receive after sending).
        ws.receive_json()
        # Give the server loop one more tick to apply the command.
        ws.receive_json()

    assert STATE.paused is True, "pause command did not propagate to STATE"
