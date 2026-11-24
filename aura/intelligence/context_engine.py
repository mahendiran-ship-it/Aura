"""Context engine: decides WHAT information matters right now.

It merges detections/tracks, the user's command, search state, memory,
navigation and safety events into one small structured ``Context``. The
same object feeds (a) the rule-based responder and (b) the optional LLM,
which therefore never sees raw camera frames and cannot invent positions.

Focus rules (highest first):
  1. safety   - a HIGH/CRITICAL alert is active       -> say only the alert
  2. command  - the user asked something              -> answer that
  3. search   - a find-object task is active          -> only that object
  4. quiet    - nothing meaningful                    -> say nothing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aura.audio.command_parser import Command, Intent
from aura.memory.object_memory import MemoryRecord
from aura.types import Alert, AlertLevel, TrackedObject
from aura.utils.helpers import humanize_age, spoken_name
from aura.vision.labels import resolve_target


@dataclass
class Context:
    focus: str  # safety | command | search | quiet
    intent: Optional[Intent]
    objects: List[Dict[str, Any]] = field(default_factory=list)
    relevant_objects: List[Dict[str, Any]] = field(default_factory=list)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    room: str = "unknown room"
    search_target: Optional[str] = None
    navigation: Optional[str] = None
    view_quality: List[str] = field(default_factory=list)
    detector_ready: bool = True

    def to_llm_payload(self) -> Dict[str, Any]:
        """JSON-safe facts for the language model. No images, no raw frames."""
        return {
            "room": self.room, "visible_objects": self.objects,
            "objects_relevant_to_question": self.relevant_objects,
            "remembered_objects": self.memory, "recent_safety_alerts": self.alerts,
            "active_search": self.search_target, "navigation": self.navigation,
            "camera_view_problems": self.view_quality, "object_detection_running": self.detector_ready,
        }


def describe_track(t: TrackedObject) -> Dict[str, Any]:
    """Compact, rounded view of one track (approximate by design)."""
    sp = t.spatial
    return {
        "object": spoken_name(t.label),
        "direction": sp.horizontal if sp else "unknown",
        "height_in_view": sp.vertical if sp else "unknown",
        "apparent_distance": (sp.proximity + " (approximate)") if sp else "unknown",
        "movement": t.movement,
        "confidence": round(float(t.confidence), 2),
    }


def describe_record(r: MemoryRecord, now_ts: float) -> Dict[str, Any]:
    return {
        "object": r.name, "last_position": r.position, "near": r.near_object, "room": r.room,
        "freshness": r.freshness.value, "age": humanize_age(now_ts - r.observed_at),
        "user_confirmed": r.user_confirmed,
    }


class ContextEngine:
    """Builds :class:`Context` objects."""

    def __init__(self, max_objects: int = 12) -> None:
        self.max_objects = max_objects

    def build(self, command: Optional[Command], tracks: List[TrackedObject],
              recent_alerts: List[Alert], room: str, search_target: Optional[str],
              memory_records: Optional[List[MemoryRecord]] = None, now_ts: float = 0.0,
              navigation: Optional[str] = None, view_issues: Optional[List[str]] = None,
              detector_ready: bool = True) -> Context:
        visible = [t for t in tracks if t.visible and t.spatial is not None]
        visible.sort(key=lambda t: t.spatial.proximity_score, reverse=True)  # type: ignore[union-attr]
        objects = [describe_track(t) for t in visible[: self.max_objects]]

        serious = [a for a in recent_alerts if a.level >= AlertLevel.HIGH]
        if serious:
            focus = "safety"
        elif command is not None and command.intent not in (Intent.EMPTY,):
            focus = "command"
        elif search_target:
            focus = "search"
        else:
            focus = "quiet"

        relevant: List[Dict[str, Any]] = []
        wanted = None
        if command is not None and command.target and command.intent in (
                Intent.FIND, Intent.WHERE_IS, Intent.GUIDE, Intent.LAST_SEEN):
            wanted = command.target
        elif search_target:
            wanted = search_target
        if wanted:
            labels, _ = resolve_target(wanted)
            relevant = [describe_track(t) for t in visible if t.label in labels]

        return Context(
            focus=focus, intent=command.intent if command else None, objects=objects,
            relevant_objects=relevant,
            memory=[describe_record(r, now_ts) for r in (memory_records or [])],
            alerts=[a.message for a in recent_alerts[-3:]], room=room, search_target=search_target,
            navigation=navigation, view_quality=list(view_issues or []), detector_ready=detector_ready)

    @staticmethod
    def should_stay_silent(ctx: Context, new_alert: Optional[Alert], search_message: Optional[str]) -> bool:
        """True when there is nothing important to say proactively."""
        return new_alert is None and not search_message
