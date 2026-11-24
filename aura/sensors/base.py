"""Sensor and haptic interfaces so hardware can be added without touching the AI code.

Rule: a sensor object only exists if real hardware is behind it. There is no
"simulated" sensor in the runtime package (tests use their own fakes).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DistanceReading:
    distance_cm: float
    direction: str  # "ahead", "left", "right", "down"
    sensor_id: str
    timestamp: float


class DistanceSensor(ABC):
    """Ultrasonic / ToF range sensor."""

    sensor_id = "distance"
    direction = "ahead"

    @abstractmethod
    def read(self) -> Optional[DistanceReading]:
        """Latest reading, or None if the sensor has nothing fresh."""

    def close(self) -> None:  # pragma: no cover
        pass


class HapticOutput(ABC):
    """Vibration motor / wearable buzzer."""

    @abstractmethod
    def pulse(self, level: int, direction: str = "ahead") -> None:
        """Vibrate. ``level`` 1 (gentle) .. 4 (urgent)."""


class SensorHub:
    """Collects readings from all configured distance sensors."""

    def __init__(self, sensors: Optional[List[DistanceSensor]] = None,
                 max_age_s: float = 1.0) -> None:
        self.sensors = sensors or []
        self.max_age_s = max_age_s

    def poll(self) -> List[DistanceReading]:
        """Fresh readings only; stale ones are ignored rather than trusted."""
        out = []
        now_t = time.time()
        for s in self.sensors:
            try:
                r = s.read()
            except Exception:  # noqa: BLE001 - a flaky sensor must not crash AURA
                continue
            if r is not None and (now_t - r.timestamp) <= self.max_age_s:
                out.append(r)
        return out

    def status(self) -> Dict[str, str]:
        return {s.sensor_id: "connected" for s in self.sensors}
