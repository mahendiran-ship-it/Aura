"""Room/scene identity.

Implemented: *user-named* rooms ("Remember this location" / "I'm in the
kitchen"). Planned: automatic room recognition from visual landmarks (see
docs/ROADMAP.md) - AURA does not guess room names on its own.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from aura.memory.database import Database
from aura.utils.helpers import now


class SpatialMemory:
    """Tracks the current room and saved named locations."""

    def __init__(self, db: Database, default_room: str = "unknown room",
                 clock: Callable[[], float] = now) -> None:
        self.db = db
        self.default_room = default_room
        self.current_room = default_room
        self._clock = clock

    def set_room(self, name: str) -> str:
        """User declares where they are. Returns the normalised name."""
        self.current_room = name.strip().lower() or self.default_room
        return self.current_room

    def remember_location(self, note: Optional[str] = None) -> str:
        """Save the current room as a named place the user can recall."""
        text = note.strip() if note else self.current_room
        self.db.execute("INSERT INTO personal_notes(kind,text,room,created_at) VALUES ('location',?,?,?)",
                        (text, self.current_room, self._clock()))
        return text

    def saved_locations(self, limit: int = 10) -> List[str]:
        rows = self.db.query("SELECT text FROM personal_notes WHERE kind='location' "
                             "ORDER BY created_at DESC LIMIT ?", (limit,))
        return [r["text"] for r in rows]
