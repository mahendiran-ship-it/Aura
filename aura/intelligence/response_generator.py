"""Fixed, honest wording for responses. No language model involved."""

from __future__ import annotations

from typing import Any, Dict, List

from aura.safety.safety_rules import CROSSING_DISCLAIMER

NOT_RUNNING_VISION = ("Object detection is not running, so I cannot describe what is around you. "
                      "Run 'python scripts/download_models.py' to install the detection model.")
NO_CAMERA = "I cannot see right now because the camera is not working."
NO_FRAME_YET = "I have not processed a camera image yet. Please try again in a moment."
PAUSED = "Assistance is paused. Say 'resume' to start again."
UNKNOWN_COMMAND = "I did not understand that. Say 'help' to hear what I can do."
AI_OFF_NOTE = "My AI assistant is off, so I can only answer basic questions."


def degraded_prefix(issues: List[str]) -> str:
    """Warn when the camera view is poor so descriptions are not over-trusted."""
    if not issues:
        return ""
    return f"The camera view is {' and '.join(issues)}, so this may be incomplete. "


def status_text(snap: Dict[str, Any], sensors: Dict[str, str], memory_count: int) -> str:
    parts = [
        f"Camera: {snap['camera']}.",
        f"Object detection: {snap['detector']}.",
        f"{snap['ai']}",
        f"Room: {snap['room']}.",
        f"Memory holds {memory_count} observation{'s' if memory_count != 1 else ''}.",
    ]
    if snap.get("paused"):
        parts.append("Assistance is paused.")
    if snap.get("view_quality_issues"):
        parts.append("Camera view problems: " + ", ".join(snap["view_quality_issues"]) + ".")
    if sensors:
        parts.append("Sensors: " + ", ".join(f"{k} {v}" for k, v in sensors.items()) + ".")
    return " ".join(parts)


def crossing_response() -> str:
    return CROSSING_DISCLAIMER
