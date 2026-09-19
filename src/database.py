"""SQLite storage for prediction history."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    filename    TEXT,
    model       TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    latency_ms  REAL,
    top_k       TEXT,
    image_path  TEXT,
    source      TEXT    DEFAULT 'upload'
)
"""


@contextmanager
def _connect():
    con = sqlite3.connect(config.DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.execute(_SCHEMA)


def _save_thumbnail(image: Image.Image) -> str | None:
    try:
        thumb = image.copy()
        thumb.thumbnail((384, 384))
        path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
        thumb.convert("RGB").save(path, quality=88)
        return str(path)
    except Exception:
        return None


def add_prediction(image: Image.Image, filename: str, model: str, pred: dict,
                   source: str = "upload") -> int:
    """Store one prediction (and a small thumbnail of the image)."""
    image_path = _save_thumbnail(image)
    row = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        model,
        pred["label"],
        float(pred["confidence"]),
        float(pred["latency_ms"]),
        json.dumps(pred["top_k"]),
        image_path,
        source,
    )
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO predictions (created_at, filename, model, label, confidence, "
            "latency_ms, top_k, image_path, source) VALUES (?,?,?,?,?,?,?,?,?)",
            row,
        )
        return cur.lastrowid


def fetch_history(limit: int | None = None) -> pd.DataFrame:
    """All predictions, newest first, as a DataFrame."""
    query = "SELECT * FROM predictions ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with _connect() as con:
        df = pd.read_sql_query(query, con)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["confidence_pct"] = df["confidence"] * 100
    return df


def _remove_file(path: str | None) -> None:
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def delete_prediction(prediction_id: int) -> None:
    with _connect() as con:
        row = con.execute("SELECT image_path FROM predictions WHERE id = ?", (int(prediction_id),)).fetchone()
        if row:
            _remove_file(row[0])
        con.execute("DELETE FROM predictions WHERE id = ?", (int(prediction_id),))


def clear_history() -> None:
    with _connect() as con:
        for (path,) in con.execute("SELECT image_path FROM predictions").fetchall():
            _remove_file(path)
        con.execute("DELETE FROM predictions")
