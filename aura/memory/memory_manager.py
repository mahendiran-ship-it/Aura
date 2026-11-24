"""Single entry point to all memory, honouring the privacy settings."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from aura.memory.database import Database
from aura.memory.object_memory import Freshness, MemoryRecord, ObjectMemory
from aura.memory.personal_memory import PersonalMemory
from aura.memory.spatial_memory import SpatialMemory
from aura.types import TrackedObject
from aura.utils.helpers import now, spoken_name
from aura.vision.labels import resolve_target
from aura.vision.scene_analyzer import NOT_PATH_OBSTACLES  # noqa: F401  (re-exported for callers)

# Large, mostly-static things that make good "near the ..." landmarks.
LANDMARK_LABELS = {"couch", "dining table", "bed", "chair", "tv", "refrigerator", "sink",
                   "toilet", "oven", "microwave", "bench"}
# Moving things are never remembered as locations.
NEVER_STORE = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "dog", "cat", "bird", "train"}


class MemoryManager:
    """Facade over object, spatial and personal memory."""

    def __init__(self, db_path: str = ":memory:", enabled: bool = True,
                 recent_seconds: float = 600, stale_seconds: float = 86400,
                 retention_days: float = 30, write_interval_s: float = 30,
                 default_room: str = "unknown room", clock: Callable[[], float] = now) -> None:
        self.enabled = enabled
        self.db = Database(db_path)
        self.objects = ObjectMemory(self.db, recent_seconds, stale_seconds, retention_days,
                                    write_interval_s, clock)
        self.spatial = SpatialMemory(self.db, default_room, clock)
        self.personal = PersonalMemory(self.db, clock)
        if enabled:
            self.objects.purge_expired()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], db_path: str) -> "MemoryManager":
        return cls(db_path=db_path, enabled=bool(cfg.get("enabled", True)),
                   recent_seconds=cfg["recent_seconds"], stale_seconds=cfg["stale_seconds"],
                   retention_days=cfg["retention_days"], write_interval_s=cfg["write_interval_s"],
                   default_room=cfg["default_room"])

    @property
    def room(self) -> str:
        return self.spatial.current_room

    def observe(self, visible: List[TrackedObject]) -> int:
        """Store sightings of stable, static objects. Returns rows written."""
        if not self.enabled:
            return 0
        landmarks = [t for t in visible if t.label in LANDMARK_LABELS and t.spatial is not None]
        written = 0
        for t in visible:
            if t.label in NEVER_STORE or t.spatial is None or t.hits < 5:
                continue
            near = None
            if t.label not in LANDMARK_LABELS:
                others = [l for l in landmarks if l.track_id != t.track_id]
                if others:
                    closest = min(others, key=lambda l: abs(l.bbox.cx - t.bbox.cx) + abs(l.bbox.cy - t.bbox.cy))
                    near = spoken_name(closest.label)
            from aura.vision.spatial_estimator import describe_direction
            written += int(self.objects.record(t.label, self.room, describe_direction(t.spatial.horizontal),
                                                t.confidence, near_object=near))
        return written

    def last_seen(self, target: str) -> tuple[Optional[MemoryRecord], Optional[str]]:
        """Return ``(record, problem)`` for a spoken object name."""
        labels, problem = resolve_target(target)
        if problem and not labels:
            # unsupported live, but user-confirmed notes may still exist
            rec = self.objects.last_seen([target.lower()])
            return rec, None
        return self.objects.last_seen(labels), None

    def confirm(self, target: str, position: str) -> str:
        """User tells AURA where something is (source = user, confirmed)."""
        labels, _ = resolve_target(target)
        label = labels[0] if labels else target.lower()
        self.objects.record(label, self.room, position, 1.0, source="user",
                            user_confirmed=True, name=target.lower())
        return label

    def forget(self, target: str) -> int:
        labels, _ = resolve_target(target)
        return self.objects.delete_label(labels or [target.lower()])

    def clear_all(self) -> int:
        return self.objects.clear_all() + self.personal.clear()

    def close(self) -> None:
        self.db.close()
