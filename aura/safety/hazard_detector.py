"""Turns tracked objects into candidate :class:`Alert` objects.

Deterministic. It only reports what the sensors/vision actually support and
never produces an "all clear" message.
"""

from __future__ import annotations

from typing import Any, Dict, List

from aura.safety import safety_rules as rules
from aura.types import Alert, AlertLevel, TrackedObject
from aura.utils.helpers import now, spoken_name


class HazardDetector:
    """Evaluate the visible tracks of one frame."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def evaluate(self, tracks: List[TrackedObject], frame_height: int) -> List[Alert]:
        """Return alerts (any level, including INFO) for the current frame."""
        alerts: List[Alert] = []
        vehicles = set(self.cfg.get("vehicle_classes", []))
        t = now()
        for trk in tracks:
            sp = trk.spatial
            if sp is None or not trk.visible:
                continue
            if trk.confidence < float(self.cfg["min_confidence"]):
                continue
            if trk.hits < int(self.cfg["min_persistence_frames"]):
                continue
            score = rules.risk_score(trk, self.cfg)
            level = rules.level_for_score(score, self.cfg)
            if level <= AlertLevel.INFO:
                continue
            name = spoken_name(trk.label)
            side = sp.horizontal if sp.horizontal != "ahead" else "ahead"
            where = {"ahead": "ahead", "slightly left": "slightly to your left",
                     "slightly right": "slightly to your right", "left": "on your left",
                     "far left": "on your far left", "right": "on your right",
                     "far right": "on your far right"}[side]
            low = (trk.bbox.y1 / max(1, frame_height)) > float(self.cfg["low_obstacle_max_top"])
            if trk.label in vehicles:
                msg = f"Possible {'moving ' if trk.movement == 'approaching' else ''}{name} {where}."
                if trk.movement == "approaching":
                    msg = f"Possible approaching {name} {where}."
            elif low and sp.in_walking_zone:
                msg = f"Low obstacle detected in the walking area: {name} {where}."
            elif trk.label == "person":
                msg = f"Person {where}." + (" Approaching." if trk.movement == "approaching" else "")
            else:
                msg = f"Obstacle {where}: {name}."
            if level >= AlertLevel.HIGH and sp.uncertain:
                msg += " Distance is approximate."
            alerts.append(Alert(level=level, message=msg, key=f"{trk.label}:{sp.horizontal}",
                                timestamp=t, source="vision"))
        alerts.sort(key=lambda a: a.level, reverse=True)
        return alerts

    def from_distance_sensor(self, distance_cm: float, direction: str = "ahead") -> List[Alert]:
        """Alert from a real range sensor reading (caller must pass real data)."""
        level = rules.sensor_level(distance_cm, self.cfg)
        if level is None:
            return []
        return [Alert(level=level, message=f"Obstacle {direction}, about {int(distance_cm)} centimetres.",
                      key=f"sensor:{direction}", timestamp=now(), source="sensor")]
