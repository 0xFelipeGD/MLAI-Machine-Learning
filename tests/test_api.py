"""tests/test_api.py — Smoke tests for FastAPI endpoints (AGRO-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import create_app
    from engine.db import init_db

    init_db(ROOT / "data" / "test_mlai.db")
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_root(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "MLAI API"


def test_health(client) -> None:
    r = client.get("/api/system/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "fps" in body


def test_agro_history_empty(client) -> None:
    r = client.get("/api/agro/history?limit=10")
    assert r.status_code == 200
    assert "items" in r.json()


def test_agro_stats(client) -> None:
    r = client.get("/api/agro/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_detections" in body
