"""tests/test_ws_cadence.py — /ws/live keeps streaming fresh frames without freezing.

What this proves
----------------
A background producer simulates the engine: it mutates STATE.last_frame_b64
every ~100 ms with a uniquely-encoded fake JPEG. We then open the real WS
and count how many frames we receive, and how many of them are *distinct*,
over a short window.

Two regressions this catches that the plain contract test (test_ws_live.py)
does NOT catch:

1. **Stream stall.** If somebody refactors the WS loop and introduces a
   deadlock, busy-loop, or slow path, we receive << expected frames in the
   time window. The contract test would pass because a single message still
   comes through before timing out.

2. **Stale/cached frame.** If somebody caches STATE at connect time instead
   of re-reading it each iteration (common cache bug), we would receive N
   frames but they would all be the same one. `set(received_b64)` catches
   this: if the producer writes 20 different JPEGs but we see only 1
   unique value, the server is frozen on a cached read.

What this does NOT prove
------------------------
- Real camera latency on the Pi.
- Browser-side rendering (that is still the Playwright E2E plan, which we
  decided to skip for now).
- Sustained hours of streaming (this window is ~2–4 s).

Total runtime budget: ≤ 5 s.
"""

from __future__ import annotations

import base64
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------- helpers

def _fake_jpeg_with_counter(counter: int) -> str:
    """Build a tiny but structurally valid JPEG whose bytes depend on `counter`.

    Structure: SOI + JFIF APP0 + COM (comment segment carrying the counter)
    + EOI. The COM segment makes the raw bytes unique per call so downstream
    code can distinguish one frame from the next.
    """
    counter_bytes = str(counter).encode().rjust(8, b"\x00")
    # COM marker is 0xFF 0xFE, length=2+len(payload).
    payload_len = 2 + len(counter_bytes)  # 2 bytes length field + payload
    jpeg = (
        b"\xff\xd8"  # SOI
        b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"  # APP0 (JFIF)
        b"\xff\xfe" + payload_len.to_bytes(2, "big") + counter_bytes  # COM
        + b"\xff\xd9"  # EOI
    )
    return base64.b64encode(jpeg).decode()


def _reset_state() -> None:
    from engine.state import STATE

    STATE.last_frame_b64 = None
    STATE.last_agro_result = None
    STATE.fps = 0.0
    STATE.paused = False


class _FrameProducer:
    """Background thread emulating engine/main's capture loop."""

    def __init__(self, period_s: float = 0.1) -> None:
        self.period = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.count = 0

    def _run(self) -> None:
        from engine.state import STATE

        while not self._stop.is_set():
            self.count += 1
            STATE.update_frame(_fake_jpeg_with_counter(self.count), fps=float(self.count))
            time.sleep(self.period)

    def __enter__(self) -> "_FrameProducer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


# --------------------------------------------------------------- fixture

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from api.main import create_app
    from engine.db import init_db

    _reset_state()
    init_db(ROOT / "data" / "test_mlai.db")
    app = create_app()
    with TestClient(app) as c:
        yield c
    _reset_state()


# ---------------------------------------------------------------- test

def test_ws_streams_changing_frames_without_freeze(client):
    """Over ≤ 5 s the client must receive several distinct frames.

    Sizing:
      - Producer cadence: 100 ms → 10 Hz of state updates.
      - WS loop cadence : 200 ms → 5 Hz of send_json.
      - In 2 s that is ~10 frames emitted; we require at least 5.
      - Producer encodes a unique counter each frame, so with 5 frames we
        require at least 3 distinct payloads. Anything less than 3 is a
        strong smell of a stale/frozen read.
    """
    FRAME_TARGET = 10            # try to collect this many
    MIN_FRAMES = 5               # fail below this (stream stalled)
    MIN_UNIQUE = 3               # fail below this (frames frozen)
    TIMEOUT_S = 5.0              # overall hard cap

    received: list[str] = []
    start = time.time()

    with _FrameProducer(period_s=0.1):
        with client.websocket_connect("/ws/live") as ws:
            while len(received) < FRAME_TARGET and time.time() - start < TIMEOUT_S:
                msg = ws.receive_json()
                if msg.get("type") == "frame":
                    received.append(msg["frame_b64"])

    elapsed = time.time() - start

    assert len(received) >= MIN_FRAMES, (
        f"stream stalled: only {len(received)} frames in {elapsed:.2f}s "
        f"(expected ≥ {MIN_FRAMES})"
    )

    unique = len(set(received))
    assert unique >= MIN_UNIQUE, (
        f"frames appear frozen: {unique} unique out of {len(received)} "
        f"received in {elapsed:.2f}s (expected ≥ {MIN_UNIQUE} distinct)"
    )
