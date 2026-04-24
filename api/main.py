"""
api/main.py — FastAPI application factory and dev server.

For development you can run the API and engine in the same process:

    python -m api.main

For production each service is its own systemd unit; the API still imports
engine.state.STATE, but in production the engine writes its updates via
that singleton because both run inside the same process supervised by
mlai-engine.service.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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
    logger.info("API startup complete")
    yield
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
