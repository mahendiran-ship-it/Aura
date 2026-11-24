"""Relative position of objects (left/right/ahead, near/far).

Honesty note: a single ordinary camera cannot measure distance. The
``proximity`` here is a heuristic from how big the object looks and how
low it sits in the image. It is always flagged ``uncertain`` and spoken
with hedging words ("appears", "roughly"). Real depth needs a depth sensor,
a stereo camera, or a depth model - see :class:`DepthProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from aura.types import BBox, Detection, SpatialInfo
from aura.utils.helpers import clamp

# Heuristic "how large does this normally look when close" scaling per class.
# Big things fill the frame sooner, so they need a larger apparent size to
# count as "near". Unknown classes use 1.0.
_SIZE_SCALE: Dict[str, float] = {"car": 1.6, "bus": 1.8, "truck": 1.8, "train": 2.0,
                                 "person": 1.0, "chair": 0.9, "bottle": 0.3, "cup": 0.25,
                                 "cell phone": 0.2, "backpack": 0.6, "dining table": 1.3,
                                 "couch": 1.5, "bed": 1.6}


class DepthProvider(ABC):
    """Plug-in point for real depth (ToF sensor, stereo, depth model).

    Implement :meth:`distance_m` and pass the provider to
    :class:`SpatialEstimator`; real measurements then replace the heuristic.
    """

    @abstractmethod
    def distance_m(self, frame: np.ndarray, bbox: BBox) -> Optional[float]:
        """Metres to the object in ``bbox``, or None if unknown."""


class VisualLocalizer(ABC):
    """Future plug-in: visual SLAM / localisation. Not implemented yet."""

    @abstractmethod
    def pose(self) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, heading) in a map frame, or None if lost."""


class SpatialEstimator:
    """Turns a bounding box into a :class:`SpatialInfo`."""

    def __init__(self, frame_width: int, frame_height: int,
                 zone_x_min: float = 0.30, zone_x_max: float = 0.70,
                 depth_provider: Optional[DepthProvider] = None) -> None:
        self.w = max(1, frame_width)
        self.h = max(1, frame_height)
        self.zone = (zone_x_min, zone_x_max)
        self.depth = depth_provider

    @staticmethod
    def horizontal_name(cx_frac: float) -> str:
        if cx_frac < 0.15:
            return "far left"
        if cx_frac < 0.35:
            return "left"
        if cx_frac < 0.45:
            return "slightly left"
        if cx_frac <= 0.55:
            return "ahead"
        if cx_frac <= 0.65:
            return "slightly right"
        if cx_frac <= 0.85:
            return "right"
        return "far right"

    @staticmethod
    def vertical_name(cy_frac: float) -> str:
        return "upper" if cy_frac < 0.33 else ("middle" if cy_frac < 0.66 else "lower")

    def proximity_score(self, label: str, bbox: BBox) -> float:
        """0 (far) .. 1 (near), from apparent size and how low the box bottom sits."""
        size = max(bbox.height / self.h, float(np.sqrt(bbox.area / (self.w * self.h))))
        size = size / _SIZE_SCALE.get(label, 1.0)
        bottom = clamp(bbox.y2 / self.h, 0.0, 1.0)  # lower in image = closer to the ground-level user
        return clamp(0.7 * size + 0.3 * max(0.0, bottom - 0.4) / 0.6, 0.0, 1.0)

    def estimate(self, det: Detection, frame: Optional[np.ndarray] = None) -> SpatialInfo:
        """Estimate where ``det`` is relative to the camera."""
        b = det.bbox
        cx_frac, cy_frac = b.cx / self.w, b.cy / self.h
        score = self.proximity_score(det.label, b)
        uncertain = True
        if self.depth is not None and frame is not None:
            metres = self.depth.distance_m(frame, b)
            if metres is not None:
                score = clamp(1.0 - metres / 6.0, 0.0, 1.0)  # 0 m => 1.0, >=6 m => 0
                uncertain = False
        proximity = "near" if score >= 0.55 else ("medium" if score >= 0.25 else "far")
        in_zone = self.zone[0] <= cx_frac <= self.zone[1] or (
            b.x1 / self.w < self.zone[1] and b.x2 / self.w > self.zone[0] and b.width / self.w > 0.35)
        return SpatialInfo(horizontal=self.horizontal_name(cx_frac), vertical=self.vertical_name(cy_frac),
                           proximity=proximity, proximity_score=score, in_walking_zone=in_zone,
                           uncertain=uncertain)


def describe_direction(horizontal: str) -> str:
    """Spoken direction phrase: 'slightly right' -> 'slightly to your right'."""
    return {"far left": "far to your left", "left": "to your left", "slightly left": "slightly to your left",
            "ahead": "ahead", "slightly right": "slightly to your right", "right": "to your right",
            "far right": "far to your right"}.get(horizontal, horizontal)


def describe_proximity(info: SpatialInfo) -> str:
    """'near' -> 'appears close'; hedged because monocular distance is uncertain."""
    word = {"near": "appears close", "medium": "appears at a medium distance", "far": "appears far away"}[info.proximity]
    return word if info.uncertain else word.replace("appears", "is")
