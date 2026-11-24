"""Shared, thread-safe runtime state (what is AURA doing right now?)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aura.types import TrackedObject


@dataclass
class AuraState:
    """Plain data holder; access it through :class:`StateManager`."""

    running: bool = False
    paused: bool = False  # user said "stop assistance"
    camera_status: str = "not started"
    camera_ok: bool = False
    view_quality_ok: bool = True
    view_quality_issues: List[str] = field(default_factory=list)
    detector_ready: bool = False
    detector_message: str = "not loaded"
    ai_status: str = "disabled"
    cloud_in_use: bool = False
    current_room: str = "unknown room"
    last_spoken: str = ""
    last_command: str = ""
    search_target: Optional[str] = None
    navigating: bool = False
    walking_mode: bool = True  # proactive hazard alerts on
    tracked: List[TrackedObject] = field(default_factory=list)
    frame_size: tuple = (0, 0)
    frames_processed: int = 0
    last_frame_time: float = 0.0


class StateManager:
    """Locked accessor around :class:`AuraState`."""

    def __init__(self) -> None:
        self._state = AuraState()
        self._lock = threading.RLock()

    def get(self, name: str) -> Any:
        with self._lock:
            return getattr(self._state, name)

    def set(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._state, key):
                    raise AttributeError(f"Unknown state field: {key}")
                setattr(self._state, key, value)

    def snapshot(self) -> Dict[str, Any]:
        """A plain-dict copy safe to serialise (no tracked-object internals)."""
        with self._lock:
            s = self._state
            return {
                "running": s.running, "paused": s.paused,
                "camera": s.camera_status, "camera_ok": s.camera_ok,
                "view_quality_ok": s.view_quality_ok, "view_quality_issues": list(s.view_quality_issues),
                "detector": s.detector_message, "detector_ready": s.detector_ready,
                "ai": s.ai_status, "cloud_in_use": s.cloud_in_use,
                "room": s.current_room, "search_target": s.search_target,
                "navigating": s.navigating, "tracked_objects": len(s.tracked),
                "frames_processed": s.frames_processed, "performance_note": "",
            }
