"""Fixed safety wording and the risk-scoring function.

Everything here is plain deterministic Python. No language model is ever
consulted for a safety decision.

Risk score (higher = worse), all inputs measurable from the tracker:

    score = class_weight
          * (0.4 + 0.9 * proximity_score)     # nearer => more risk
          * position_factor                    # in walking zone 1.0, side 0.55, far side 0.3
          * movement_factor                    # approaching 1.35, stationary 1.0, away 0.6
          * reliability                        # confidence x persistence x track hits

The score maps to INFO/LOW/MEDIUM/HIGH/CRITICAL using thresholds in
config/safety.yaml. These numbers are heuristics, NOT validated against
real-world injury data. See docs/SAFETY.md.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aura.types import AlertLevel, TrackedObject
from aura.utils.helpers import clamp

CROSSING_DISCLAIMER = ("I cannot reliably determine whether it is safe to cross. "
                       "Please stop and use your established crossing method.")
GENERAL_DISCLAIMER = ("I am an aid only. I cannot confirm that any path, stairs or road is safe. "
                      "Keep using your cane, guide dog, or mobility training.")
DEGRADED_VIEW = "My camera view is poor, so I may miss hazards. Please be extra careful."
NO_DETECTOR = "Obstacle detection is not running, so I cannot warn you about hazards."
CAMERA_LOST = "The camera has stopped working. I cannot warn you about obstacles. Please be careful."


def reliability(track: TrackedObject, min_hits: int) -> float:
    """0..1 trust in a track: detection confidence x how many frames it persisted."""
    persistence = clamp(track.hits / float(max(1, min_hits) * 2), 0.0, 1.0)
    return clamp(track.confidence, 0.0, 1.0) * (0.5 + 0.5 * persistence)


def position_factor(track: TrackedObject) -> float:
    sp = track.spatial
    if sp is None:
        return 0.3
    if sp.in_walking_zone:
        return 1.0
    return 0.55 if sp.horizontal in ("left", "right", "slightly left", "slightly right") else 0.3


def movement_factor(track: TrackedObject) -> float:
    return {"approaching": 1.35, "moving away": 0.6}.get(track.movement, 1.0)


def risk_score(track: TrackedObject, cfg: Dict[str, Any]) -> float:
    """Compute the numeric risk for one tracked object."""
    if track.spatial is None:
        return 0.0
    weights = cfg["class_weights"]
    weight = float(weights.get(track.label, weights.get("default", 0.4)))
    proximity = 0.4 + 0.9 * track.spatial.proximity_score
    rel = reliability(track, int(cfg.get("min_persistence_frames", 3)))
    return weight * proximity * position_factor(track) * movement_factor(track) * rel


def level_for_score(score: float, cfg: Dict[str, Any]) -> AlertLevel:
    th = cfg["level_thresholds"]
    if score >= th["CRITICAL"]:
        return AlertLevel.CRITICAL
    if score >= th["HIGH"]:
        return AlertLevel.HIGH
    if score >= th["MEDIUM"]:
        return AlertLevel.MEDIUM
    if score >= th["LOW"]:
        return AlertLevel.LOW
    return AlertLevel.INFO


def level_from_name(name: str) -> AlertLevel:
    try:
        return AlertLevel[str(name).upper()]
    except KeyError:
        return AlertLevel.LOW


def sensor_level(distance_cm: float, cfg: Dict[str, Any]) -> Optional[AlertLevel]:
    """Level from a *real* distance sensor reading (ultrasonic/ToF)."""
    th = cfg["sensor_distance_cm"]
    for name in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if distance_cm <= th[name]:
            return AlertLevel[name]
    return None
