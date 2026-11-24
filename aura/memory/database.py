"""SQLite wrapper. Stores structured facts only - never images or audio."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,            -- detector category, e.g. 'cell phone'
    name TEXT NOT NULL,             -- what the user calls it, e.g. 'phone'
    room TEXT NOT NULL,             -- scene/room identity
    position TEXT NOT NULL,         -- relative position text, e.g. 'to your left'
    near_object TEXT,               -- nearby landmark object, e.g. 'sofa'
    confidence REAL NOT NULL,
    source TEXT NOT NULL,           -- 'camera' | 'user'
    user_confirmed INTEGER NOT NULL DEFAULT 0,
    observed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_label ON observations(label, observed_at);
CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(observed_at);

CREATE TABLE IF NOT EXISTS personal_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,             -- 'location' | 'note'
    text TEXT NOT NULL,
    room TEXT,
    created_at REAL NOT NULL
);
"""


class Database:
    """Thread-safe SQLite connection. Use ``":memory:"`` for tests."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
