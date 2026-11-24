"""Shared data types used across AURA modules.

Keeping these in one small module avoids circular imports and gives a
beginner a single place to see what flows through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


class AlertLevel(IntEnum):
    """Severity of a safety event. Higher number = more severe."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (x1,y1 top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def iou(self, other: "BBox") -> float:
        """Intersection-over-union with another box (0..1)."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """A single object found in one frame."""

    label: str
    confidence: float
    bbox: BBox


@dataclass
class SpatialInfo:
    """Coarse position of an object relative to the camera.

    ``proximity`` is a *rough, uncertain* estimate from apparent size and
    position in the image. It is NOT a measured distance.
    """

    horizontal: str  # far left | left | slightly left | ahead | slightly right | right | far right
    vertical: str  # upper | middle | lower
    proximity: str  # near | medium | far
    proximity_score: float  # 0 (far) .. 1 (near), heuristic
    in_walking_zone: bool = False
    uncertain: bool = True


@dataclass
class TrackedObject:
    """An object followed across frames."""

    track_id: int
    label: str
    confidence: float
    bbox: BBox
    spatial: Optional[SpatialInfo] = None
    hits: int = 1  # frames in which it was matched
    misses: int = 0  # consecutive frames it was not matched
    first_seen: float = 0.0
    last_seen: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)  # px/frame (dx, dy) of centre
    growth: float = 0.0  # relative bbox-area change per frame (+ = getting bigger)
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (cx, cy, area)

    @property
    def visible(self) -> bool:
        """True if matched in the most recent processed frame."""
        return self.misses == 0

    @property
    def movement(self) -> str:
        """Plain-language movement estimate."""
        if self.hits < 4:
            return "unknown"
        if self.growth > 0.04:
            return "approaching"
        if self.growth < -0.04:
            return "moving away"
        if abs(self.velocity[0]) > 4.0:
            return "moving right" if self.velocity[0] > 0 else "moving left"
        return "stationary"


@dataclass
class Alert:
    """A safety message produced by the deterministic safety engine."""

    level: AlertLevel
    message: str
    key: str  # used for de-duplication, e.g. "car:right"
    timestamp: float = 0.0
    source: str = "vision"
