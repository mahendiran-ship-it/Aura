"""Distance sensor over a serial line (ESP32/Arduino with HC-SR04 or a ToF module).

Wire protocol (documented in docs/HARDWARE.md): the microcontroller prints
one text line per measurement, e.g. ``DIST:ahead:123.4`` (cm). Anything else
is ignored. Needs ``pip install pyserial`` and real hardware; this code has
NOT been validated against physical hardware in this repository's test environment
(it is tested with a fake serial port).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from aura.sensors.base import DistanceReading, DistanceSensor


def parse_line(line: str, default_id: str = "serial") -> Optional[DistanceReading]:
    """Parse ``DIST:<direction>:<cm>``. Returns None for malformed/implausible data."""
    parts = line.strip().split(":")
    if len(parts) != 3 or parts[0].upper() != "DIST":
        return None
    try:
        cm = float(parts[2])
    except ValueError:
        return None
    if not (1.0 <= cm <= 800.0):  # outside sensor spec => treat as noise
        return None
    direction = parts[1].lower()
    if direction not in ("ahead", "left", "right", "down"):
        return None
    return DistanceReading(cm, direction, f"{default_id}-{direction}", time.time())


class SerialDistanceSensor(DistanceSensor):
    def __init__(self, port: str, baudrate: int = 115200, serial_obj: Any = None) -> None:
        self.sensor_id = f"serial:{port}"
        if serial_obj is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is required: pip install pyserial") from exc
            serial_obj = serial.Serial(port, baudrate, timeout=0.05)
        self._ser = serial_obj
        self._last: Optional[DistanceReading] = None

    def read(self) -> Optional[DistanceReading]:
        try:
            while self._ser.in_waiting:
                raw = self._ser.readline().decode("ascii", errors="ignore")
                reading = parse_line(raw, "serial")
                if reading:
                    self._last = reading
        except Exception:  # noqa: BLE001
            return None
        return self._last

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass
