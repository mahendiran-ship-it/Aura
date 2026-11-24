"""User-provided notes ("Remember that my keys are in the blue bowl")."""

from __future__ import annotations

from typing import Callable, List

from aura.memory.database import Database
from aura.utils.helpers import now


class PersonalMemory:
    """Free-text notes the user explicitly asked AURA to keep."""

    def __init__(self, db: Database, clock: Callable[[], float] = now) -> None:
        self.db = db
        self._clock = clock

    def add_note(self, text: str, room: str = "") -> None:
        self.db.execute("INSERT INTO personal_notes(kind,text,room,created_at) VALUES ('note',?,?,?)",
                        (text.strip(), room, self._clock()))

    def search(self, text: str, limit: int = 5) -> List[str]:
        rows = self.db.query("SELECT text FROM personal_notes WHERE kind='note' AND lower(text) LIKE ? "
                             "ORDER BY created_at DESC LIMIT ?", (f"%{text.lower()}%", limit))
        return [r["text"] for r in rows]

    def clear(self) -> int:
        return self.db.execute("DELETE FROM personal_notes").rowcount
