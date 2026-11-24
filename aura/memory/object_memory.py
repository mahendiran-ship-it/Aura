"""Remembering where objects were last seen - with honest freshness labels.

Freshness states (never present old memory as current):
  CURRENT         seen by the camera right now (set by the caller, not stored)
  RECENT          seen within ``recent_seconds``
  PREVIOUS        seen longer ago, but not yet stale
  STALE           older than ``stale_seconds``; may well have moved
  USER_CONFIRMED  the user told us (still dated; can be outdated)
  UNKNOWN         no record at all
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from aura.memory.database import Database
from aura.utils.helpers import clock_time, humanize_age, now, spoken_name


class Freshness(str, Enum):
    CURRENT = "currently detected"
    RECENT = "recently observed"
    PREVIOUS = "previously observed"
    STALE = "stale"
    USER_CONFIRMED = "user-confirmed"
    UNKNOWN = "unknown"


@dataclass
class MemoryRecord:
    id: int
    label: str
    name: str
    room: str
    position: str
    near_object: Optional[str]
    confidence: float
    source: str
    user_confirmed: bool
    observed_at: float
    freshness: Freshness = Freshness.UNKNOWN


class ObjectMemory:
    """Store and query object sightings."""

    def __init__(self, db: Database, recent_seconds: float = 600, stale_seconds: float = 86400,
                 retention_days: float = 30, write_interval_s: float = 30,
                 clock: Callable[[], float] = now) -> None:
        self.db = db
        self.recent_seconds = recent_seconds
        self.stale_seconds = stale_seconds
        self.retention_days = retention_days
        self.write_interval_s = write_interval_s
        self._clock = clock

    # -- writing -----------------------------------------------------------
    def record(self, label: str, room: str, position: str, confidence: float,
               near_object: Optional[str] = None, source: str = "camera",
               user_confirmed: bool = False, name: Optional[str] = None) -> bool:
        """Save a sighting. Skips near-duplicates written within ``write_interval_s``.

        Returns True if a row was written.
        """
        t = self._clock()
        last = self.db.query_one(
            "SELECT observed_at, position, room, near_object FROM observations "
            "WHERE label=? ORDER BY observed_at DESC LIMIT 1", (label,))
        if (last and not user_confirmed and (t - last["observed_at"]) < self.write_interval_s
                and last["position"] == position and last["room"] == room
                and last["near_object"] == near_object):
            return False
        self.db.execute(
            "INSERT INTO observations(label,name,room,position,near_object,confidence,source,"
            "user_confirmed,observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (label, name or spoken_name(label), room, position, near_object, float(confidence),
             source, int(user_confirmed), t))
        return True

    # -- reading -----------------------------------------------------------
    def freshness_of(self, observed_at: float, user_confirmed: bool = False) -> Freshness:
        age = self._clock() - observed_at
        if age > self.stale_seconds:
            return Freshness.STALE
        if user_confirmed:
            return Freshness.USER_CONFIRMED
        return Freshness.RECENT if age <= self.recent_seconds else Freshness.PREVIOUS

    def _to_record(self, row) -> MemoryRecord:
        rec = MemoryRecord(id=row["id"], label=row["label"], name=row["name"], room=row["room"],
                           position=row["position"], near_object=row["near_object"],
                           confidence=row["confidence"], source=row["source"],
                           user_confirmed=bool(row["user_confirmed"]), observed_at=row["observed_at"])
        rec.freshness = self.freshness_of(rec.observed_at, rec.user_confirmed)
        return rec

    def last_seen(self, labels: List[str]) -> Optional[MemoryRecord]:
        """Most recent record for any of ``labels``."""
        if not labels:
            return None
        marks = ",".join("?" * len(labels))
        row = self.db.query_one(
            f"SELECT * FROM observations WHERE label IN ({marks}) ORDER BY observed_at DESC LIMIT 1",
            tuple(labels))
        return self._to_record(row) if row else None

    def history(self, labels: Optional[List[str]] = None, limit: int = 10) -> List[MemoryRecord]:
        if labels:
            marks = ",".join("?" * len(labels))
            rows = self.db.query(f"SELECT * FROM observations WHERE label IN ({marks}) "
                                 "ORDER BY observed_at DESC LIMIT ?", (*labels, limit))
        else:
            rows = self.db.query("SELECT * FROM observations ORDER BY observed_at DESC LIMIT ?", (limit,))
        return [self._to_record(r) for r in rows]

    def search(self, text: str, limit: int = 10) -> List[MemoryRecord]:
        like = f"%{text.lower()}%"
        rows = self.db.query(
            "SELECT * FROM observations WHERE lower(name) LIKE ? OR lower(label) LIKE ? "
            "OR lower(room) LIKE ? OR lower(coalesce(near_object,'')) LIKE ? "
            "ORDER BY observed_at DESC LIMIT ?", (like, like, like, like, limit))
        return [self._to_record(r) for r in rows]

    def count(self) -> int:
        return int(self.db.query_one("SELECT COUNT(*) AS n FROM observations")["n"])

    # -- deleting ----------------------------------------------------------
    def delete_label(self, labels: List[str]) -> int:
        """Delete every memory of the given labels. Returns rows deleted."""
        if not labels:
            return 0
        marks = ",".join("?" * len(labels))
        return self.db.execute(f"DELETE FROM observations WHERE label IN ({marks})",
                               tuple(labels)).rowcount

    def delete_id(self, record_id: int) -> int:
        return self.db.execute("DELETE FROM observations WHERE id=?", (record_id,)).rowcount

    def clear_all(self) -> int:
        n = self.db.execute("DELETE FROM observations").rowcount
        self.db.vacuum()
        return n

    def purge_expired(self) -> int:
        """Delete observations older than the retention period."""
        cutoff = self._clock() - self.retention_days * 86400
        return self.db.execute("DELETE FROM observations WHERE observed_at < ?", (cutoff,)).rowcount

    # -- speech ------------------------------------------------------------
    def describe(self, rec: Optional[MemoryRecord], display_name: str) -> str:
        """Honest sentence about a memory, including its age and freshness."""
        if rec is None:
            return f"I have no record of seeing your {display_name}."
        where = rec.room if rec.room and rec.room != "unknown room" else "an unknown room"
        place = f"near the {rec.near_object}" if rec.near_object else rec.position
        when = f"at {clock_time(rec.observed_at)}, {humanize_age(self._clock() - rec.observed_at)}"
        base = f"I last observed your {display_name} {place} in {where} {when}."
        if rec.freshness == Freshness.STALE:
            return base + " That was a long time ago, so it has probably moved."
        if rec.freshness == Freshness.PREVIOUS:
            return base + " It may have been moved since then."
        if rec.freshness == Freshness.USER_CONFIRMED:
            return f"You told me your {display_name} was {place} in {where} {when}. It may have moved since."
        return base + " I can't confirm it is still there."
