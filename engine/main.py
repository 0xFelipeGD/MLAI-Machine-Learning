"""
engine/main.py — MLAI inference engine entry point.

Runs the camera, dispatches each frame to the AGRO pipeline, persists the
result to SQLite, and updates the shared EngineState that the API serves
to clients.

Run from project root:
    python -m engine.main
"""

from __future__ import annotations

import asyncio
import base64
import logging
import signal
import time
from typing import Optional

import cv2
import yaml

from engine import PROJECT_ROOT
from engine.agro.pipeline import AgroPipeline
from engine.camera import CameraService
from engine.db import init_db, insert_agro_result, prune_old
from engine.state import STATE

logger = logging.getLogger(__name__)


def _encode_jpeg_b64(img) -> str:
    quality = int(STATE.jpeg_quality)
    if quality < 30:
        quality = 30
    elif quality > 95:
        quality = 95
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Engine:
    def __init__(self) -> None:
        cfg_path = PROJECT_ROOT / "config" / "system_config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self.cfg = yaml.safe_load(fh) or {}
        cam_cfg = self.cfg.get("camera") or {}
        self.fps_target = int(cam_cfg.get("fps", 5))
        self.num_threads = int((self.cfg.get("inference") or {}).get("num_threads", 4))
        self.prune_days = int((self.cfg.get("storage") or {}).get("prune_after_days", 30))

        # Seed runtime-mutable knobs from config so the dashboard sliders
        # start where the YAML says. Sliders mutate STATE; "bake favourites"
        # by editing the YAML and restarting.
        STATE.target_fps = self.fps_target
        STATE.jpeg_quality = int(cam_cfg.get("jpeg_quality", 80))

        self.camera = CameraService()
        self.agro: Optional[AgroPipeline] = None
        self._stop = asyncio.Event()
        init_db()

    def start(self) -> None:
        logger.info("Starting MLAI engine (AGRO)")
        # Load TFLite models FIRST. Once the camera thread starts it
        # decodes MJPEG at STREAM_READ_FPS (30 Hz) and easily saturates
        # a Pi 4 core, which can starve the XNNPACK thread pool that
        # AgroPipeline init needs — turning a normally fast init into
        # a several-minute hang. Models loaded → camera started: no
        # contention during the (one-time) load.
        try:
            self.agro = AgroPipeline(num_threads=self.num_threads)
        except Exception:
            logger.exception("Failed to construct AGRO pipeline")
        self.camera.start()
        # Expose the live camera to the API so the dashboard sliders can
        # tune ColourGains / CCM without restarting the service.
        STATE.camera = self.camera

    def stop(self) -> None:
        self.camera.stop()
        STATE.camera = None
        self._stop.set()

    async def run(self) -> None:
        last_prune = time.time()
        while not self._stop.is_set():
            # Re-read each iteration so the dashboard slider takes effect
            # without restarting the engine.
            period = 1.0 / max(int(STATE.target_fps), 1)
            t0 = time.perf_counter()
            frame = self.camera.read()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            try:
                if STATE.paused:
                    # Keep the live preview alive while paused — only inference
                    # and DB writes are skipped, so the dashboard sliders can
                    # still show colour changes without polluting history.
                    STATE.update_frame(_encode_jpeg_b64(frame), self.camera.get_fps())
                elif self.agro is not None:
                    result, annotated = self.agro.process(frame)
                    insert_agro_result(result.to_dict())
                    STATE.update_agro(result.to_dict())
                    STATE.update_frame(_encode_jpeg_b64(annotated), self.camera.get_fps())
            except Exception:
                logger.exception("frame processing failed")

            if time.time() - last_prune > 3600:
                try:
                    n = prune_old(self.prune_days)
                    if n:
                        logger.info("Pruned %d old result rows", n)
                except Exception:
                    logger.exception("prune failed")
                last_prune = time.time()

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                await asyncio.sleep(period - elapsed)


async def _amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = Engine()
    engine.start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, engine.stop)

    try:
        await engine.run()
    finally:
        engine.stop()
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
