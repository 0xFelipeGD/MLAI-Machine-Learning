"""
api/main.py — FastAPI application factory + inference engine supervisor.

The API and the inference engine run in the SAME process: FastAPI's
lifespan starts an Engine instance and schedules its async loop as a
background task. The engine populates engine.state.STATE, which the
REST routes and the /ws/live WebSocket read from directly — no IPC,
no state divergence.

Entry points:

    # production (via systemd / mlai-api.service):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # dev (single command with auto-reload disabled by default):
    python -m api.main

Engine startup is best-effort: if the camera or TFLite models are
unavailable (e.g. running on a dev PC or in CI), the engine is skipped
and the API still serves without live data. Set MLAI_NO_ENGINE=1 to
force-skip engine startup (used by pytest).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from engine import PROJECT_ROOT
from engine.db import init_db

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "system_config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    engine = None
    task: Optional[asyncio.Task] = None

    # Skip the engine under pytest (no camera, no TFLite) or when explicitly
    # disabled. Keep this import INSIDE the function so test collection
    # doesn't pay the cost of importing the engine stack.
    if os.environ.get("MLAI_NO_ENGINE"):
        logger.info("MLAI_NO_ENGINE set — skipping inference engine startup")
    else:
        try:
            from engine.main import Engine

            engine = Engine()
            engine.start()
            task = asyncio.create_task(engine.run(), name="mlai-engine")
            logger.info("Inference engine task started")
        except Exception:
            # Graceful degradation: the API is still useful (health, config
            # editing, history queries) even when the engine can't run.
            logger.exception("Engine failed to start — API running without live data")
            engine = None
            task = None

    logger.info("API startup complete")
    try:
        yield
    finally:
        if engine is not None:
            logger.info("Stopping inference engine")
            engine.stop()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Engine task didn't exit in 5s; cancelling")
                task.cancel()
            except Exception:
                logger.exception("Engine task raised during shutdown")
        logger.info("API shutdown")


def create_app() -> FastAPI:
    cfg = _load_config()
    api_cfg = cfg.get("api", {})
    cors_origins = api_cfg.get("cors_origins", ["http://localhost:3000"])

    app = FastAPI(
        title="MLAI API",
        version="1.0.0",
        description="REST + WebSocket API for the MLAI fruit inspection system.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    from api.routes import system as system_routes
    from api.routes import camera as camera_routes
    from api.routes import agro as agro_routes
    from api.routes import ws as ws_routes

    app.include_router(system_routes.router)
    app.include_router(camera_routes.router)
    app.include_router(agro_routes.router)
    app.include_router(ws_routes.router)

    captures_dir = PROJECT_ROOT / "data" / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static/captures", StaticFiles(directory=str(captures_dir)), name="captures")

    @app.get("/")
    async def root() -> dict:
        return {"name": "MLAI API", "version": "1.0.0", "docs": "/docs"}

    return app


app = create_app()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import uvicorn
    except Exception:
        logger.error("uvicorn is required to run the dev server (pip install uvicorn)")
        return 2
    cfg = _load_config().get("api", {})
    uvicorn.run(
        "api.main:app",
        host=cfg.get("host", "0.0.0.0"),
        port=int(cfg.get("port", 8000)),
        reload=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
