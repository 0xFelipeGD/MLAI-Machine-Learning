"""
engine/db.py — SQLite persistence layer.

Implements the schema in INSTRUCTIONS.md §11. Uses the standard library
sqlite3 (no ORM dependency). All functions are thread-safe via a single
shared connection guarded by a lock.

Quick start:

    from engine.db import init_db, insert_indust_result
    init_db()                          # creates the file + tables on first call
    insert_indust_result(result_dict)  # save a dict-shaped IndustResult
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from engine import PROJECT_ROOT

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CONN: Optional[sqlite3.Connection] = None
_DB_PATH: Optional[Path] = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS indust_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    verdict TEXT NOT NULL,
    defect_type TEXT,
    width_mm REAL,
    height_mm REAL,
    area_mm2 REAL,
    threshold_used REAL NOT NULL,
    inference_ms INTEGER,
    frame_path TEXT,
    heatmap_path TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_indust_timestamp ON indust_results(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_indust_verdict ON indust_results(verdict);
CREATE INDEX IF NOT EXISTS idx_indust_category ON indust_results(category);

CREATE TABLE IF NOT EXISTS agro_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_path TEXT,
    annotated_frame_path TEXT,
    total_detections INTEGER DEFAULT 0,
    avg_diameter_mm REAL,
    inference_ms INTEGER,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_agro_timestamp ON agro_results(timestamp DESC);

CREATE TABLE IF NOT EXISTS agro_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL REFERENCES agro_results(id) ON DELETE CASCADE,
    fruit_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
    diameter_mm REAL,
    quality TEXT,
    quality_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_agro_det_result ON agro_detections(result_id);
"""


def _resolve_db_path() -> Path:
    cfg_path = PROJECT_ROOT / "config" / "system_config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        rel = data.get("storage", {}).get("db_path", "data/mlai.db")
    else:
        rel = "data/mlai.db"
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def init_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (or reuse) a connection and ensure the schema exists."""
    global _CONN, _DB_PATH
    with _LOCK:
        target = path or _DB_PATH or _resolve_db_path()
        if _CONN is not None and _DB_PATH == target:
            return _CONN
        target.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(target), check_same_thread=False, isolation_level=None)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("PRAGMA foreign_keys=ON")
        _CONN.executescript(SCHEMA)
        _DB_PATH = target
        logger.info("SQLite DB ready at %s", target)
        return _CONN


def get_conn() -> sqlite3.Connection:
    if _CONN is None:
        return init_db()
    return _CONN


@contextmanager
def transaction():
    conn = get_conn()
    with _LOCK:
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


# ----------------------------------------------------------------- INDUST ops
def insert_indust_result(r: dict) -> int:
    sql = """
    INSERT INTO indust_results
        (timestamp, category, anomaly_score, verdict, defect_type,
         width_mm, height_mm, area_mm2, threshold_used, inference_ms,
         frame_path, heatmap_path, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _LOCK:
        cur = get_conn().execute(
            sql,
            (
                r.get("timestamp"),
                r["category"],
                float(r["anomaly_score"]),
                r["verdict"],
                r.get("defect_type"),
                r.get("width_mm"),
                r.get("height_mm"),
                r.get("area_mm2"),
                float(r["threshold_used"]),
                int(r.get("inference_ms") or 0),
                r.get("frame_path"),
                r.get("heatmap_path"),
                r.get("notes"),
            ),
        )
        return int(cur.lastrowid)


def list_indust(
    *,
    limit: int = 50,
    offset: int = 0,
    verdict: Optional[str] = None,
    category: Optional[str] = None,
) -> list[dict]:
    where = []
    params: list[Any] = []
    if verdict:
        where.append("verdict = ?")
        params.append(verdict)
    if category:
        where.append("category = ?")
        params.append(category)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM indust_results {where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    with _LOCK:
        rows = get_conn().execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_indust(verdict: Optional[str] = None, category: Optional[str] = None) -> int:
    where = []
    params: list[Any] = []
    if verdict:
        where.append("verdict = ?")
        params.append(verdict)
    if category:
        where.append("category = ?")
        params.append(category)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    with _LOCK:
        row = get_conn().execute(f"SELECT COUNT(*) FROM indust_results {where_sql}", params).fetchone()
    return int(row[0])


def get_indust_by_id(item_id: int) -> Optional[dict]:
    with _LOCK:
        row = get_conn().execute("SELECT * FROM indust_results WHERE id = ?", (int(item_id),)).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------- AGRO ops
def insert_agro_result(r: dict) -> int:
    sql = """
    INSERT INTO agro_results
        (timestamp, frame_path, annotated_frame_path, total_detections,
         avg_diameter_mm, inference_ms, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with transaction() as conn:
        cur = conn.execute(
            sql,
            (
                r.get("timestamp"),
                r.get("frame_path"),
                r.get("annotated_frame_path"),
                int(r.get("total_detections") or 0),
                float(r.get("avg_diameter_mm") or 0.0),
                int(r.get("inference_ms") or 0),
                r.get("notes"),
            ),
        )
        result_id = int(cur.lastrowid)
        for det in r.get("detections", []) or []:
            conn.execute(
                """
                INSERT INTO agro_detections
                    (result_id, fruit_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     diameter_mm, quality, quality_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    det["fruit_class"],
                    float(det["confidence"]),
                    int(det.get("bbox_x1") or 0),
                    int(det.get("bbox_y1") or 0),
                    int(det.get("bbox_x2") or 0),
                    int(det.get("bbox_y2") or 0),
                    float(det.get("diameter_mm") or 0),
                    det.get("quality"),
                    float(det.get("quality_confidence") or 0),
                ),
            )
    return result_id


def list_agro(*, limit: int = 50, offset: int = 0) -> list[dict]:
    with _LOCK:
        rows = get_conn().execute(
            "SELECT * FROM agro_results ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (int(limit), int(offset)),
        ).fetchall()
    return [dict(r) for r in rows]


def count_agro() -> int:
    with _LOCK:
        return int(get_conn().execute("SELECT COUNT(*) FROM agro_results").fetchone()[0])


def get_agro_by_id(item_id: int) -> Optional[dict]:
    with _LOCK:
        conn = get_conn()
        row = conn.execute("SELECT * FROM agro_results WHERE id = ?", (int(item_id),)).fetchone()
        if not row:
            return None
        d = dict(row)
        det_rows = conn.execute(
            "SELECT * FROM agro_detections WHERE result_id = ?", (int(item_id),)
        ).fetchall()
        d["detections"] = [dict(r) for r in det_rows]
        return d


def agro_stats() -> dict:
    with _LOCK:
        conn = get_conn()
        total = int(conn.execute("SELECT COUNT(*) FROM agro_detections").fetchone()[0])
        by_class = {
            r[0]: int(r[1])
            for r in conn.execute(
                "SELECT fruit_class, COUNT(*) FROM agro_detections GROUP BY fruit_class"
            ).fetchall()
        }
        by_quality = {
            r[0]: int(r[1])
            for r in conn.execute(
                "SELECT COALESCE(quality, 'unknown'), COUNT(*) FROM agro_detections GROUP BY quality"
            ).fetchall()
        }
        sizes = [
            float(r[0])
            for r in conn.execute("SELECT diameter_mm FROM agro_detections WHERE diameter_mm > 0").fetchall()
        ]
    histogram: list[dict] = []
    if sizes:
        bin_edges = list(range(0, 101, 10))
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            histogram.append({"range_mm": f"{lo}-{hi}", "count": sum(1 for s in sizes if lo <= s < hi)})
    return {
        "total_detections": total,
        "by_class": by_class,
        "by_quality": by_quality,
        "size_histogram": histogram,
    }


# ----------------------------------------------------------------- pruning
def prune_old(days: int) -> int:
    if days <= 0:
        return 0
    with _LOCK:
        cur = get_conn().execute(
            "DELETE FROM indust_results WHERE timestamp < datetime('now', ?)", (f"-{int(days)} days",)
        )
        deleted_indust = cur.rowcount
        cur = get_conn().execute(
            "DELETE FROM agro_results WHERE timestamp < datetime('now', ?)", (f"-{int(days)} days",)
        )
        deleted_agro = cur.rowcount
    return int((deleted_indust or 0) + (deleted_agro or 0))


# ----------------------------------------------------------- system_state ops
def set_state(key: str, value: Any) -> None:
    with _LOCK:
        get_conn().execute(
            "INSERT INTO system_state(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            (key, json.dumps(value)),
        )


def get_state(key: str, default: Any = None) -> Any:
    with _LOCK:
        row = get_conn().execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]
