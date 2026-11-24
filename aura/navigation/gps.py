"""Location providers. A location is only reported if a real source gave it.

none    no location source configured (default). AURA says so.
fixed   coordinates from settings.yaml (useful for testing/home base).
termux  real GPS through Termux:API (``termux-location``) on Android.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Location:
    lat: float
    lon: float
    accuracy_m: Optional[float] = None
    timestamp: float = 0.0
    source: str = "unknown"


class LocationError(Exception):
    """No location could be obtained. The message is safe to speak."""


class LocationProvider(ABC):
    @abstractmethod
    def get_location(self) -> Location:
        """Return the current location or raise LocationError."""


class NoLocationProvider(LocationProvider):
    def get_location(self) -> Location:
        raise LocationError("No location source is configured. Set navigation.location_provider "
                            "in config/settings.yaml (for example 'termux' on Android).")


class FixedLocationProvider(LocationProvider):
    def __init__(self, lat: Optional[float], lon: Optional[float]) -> None:
        if lat is None or lon is None:
            raise LocationError("fixed_lat and fixed_lon must both be set for the 'fixed' provider.")
        self._loc = Location(float(lat), float(lon), None, 0.0, "fixed")

    def get_location(self) -> Location:
        return Location(self._loc.lat, self._loc.lon, None, time.time(), "fixed")


class TermuxLocationProvider(LocationProvider):
    """Reads GPS via ``termux-location`` (requires Termux:API app + permission)."""

    def __init__(self, timeout_s: float = 20.0) -> None:
        self.binary = shutil.which("termux-location")
        self.timeout_s = timeout_s

    def get_location(self) -> Location:
        if not self.binary:
            raise LocationError("termux-location is not installed. Install Termux:API and run "
                                "'pkg install termux-api'.")
        try:
            out = subprocess.run([self.binary, "-p", "gps", "-r", "once"], capture_output=True,
                                 text=True, timeout=self.timeout_s, check=True).stdout
            data = json.loads(out)
            return Location(float(data["latitude"]), float(data["longitude"]),
                            float(data.get("accuracy", 0)) or None, time.time(), "termux-gps")
        except Exception as exc:  # noqa: BLE001
            raise LocationError("I could not get a GPS fix. Check location permission and "
                                "that you have a clear view of the sky.") from exc


def create_location_provider(cfg: Dict[str, Any]) -> LocationProvider:
    kind = str(cfg.get("location_provider", "none")).lower()
    if kind == "fixed":
        return FixedLocationProvider(cfg.get("fixed_lat"), cfg.get("fixed_lon"))
    if kind == "termux":
        return TermuxLocationProvider()
    return NoLocationProvider()
