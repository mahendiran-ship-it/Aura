"""Nearby landmarks from OpenStreetMap (Overpass API). REQUIRES INTERNET.

Experimental: implemented against the documented Overpass API and unit
tested with recorded-style responses, but not validated in the field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aura.utils.helpers import bearing_deg, compass_name, haversine_m
from aura.utils.logger import get_logger

log = get_logger("landmarks")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class LandmarkEngine:
    def __init__(self, user_agent: str = "AURA-assistive", timeout_s: float = 15,
                 url: str = OVERPASS_URL, session: Any = None) -> None:
        self.user_agent, self.timeout, self.url, self._session = user_agent, timeout_s, url, session

    def nearby(self, lat: float, lon: float, radius_m: int = 100, limit: int = 5) -> List[Dict[str, Any]]:
        """Named places within ``radius_m``. Returns [] if none; raises RuntimeError if offline."""
        query = (f'[out:json][timeout:10];(node(around:{radius_m},{lat},{lon})["name"]'
                 f'["shop"];node(around:{radius_m},{lat},{lon})["name"]["amenity"];'
                 f'node(around:{radius_m},{lat},{lon})["name"]["public_transport"];);out 30;')
        try:
            if self._session is None:
                import requests
                self._session = requests
            resp = self._session.post(self.url, data={"data": query}, timeout=self.timeout,
                                      headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Landmark lookup failed: %s", type(exc).__name__)
            raise RuntimeError("I could not reach the map service for landmarks.") from exc
        out = []
        for el in elements:
            name = el.get("tags", {}).get("name")
            if not name or "lat" not in el:
                continue
            out.append({"name": name, "distance_m": haversine_m(lat, lon, el["lat"], el["lon"]),
                        "direction": compass_name(bearing_deg(lat, lon, el["lat"], el["lon"]))})
        out.sort(key=lambda d: d["distance_m"])
        return out[:limit]

    def describe(self, lat: float, lon: float) -> str:
        places = self.nearby(lat, lon)
        if not places:
            return "I found no named places in the map data within about 100 metres."
        parts = [f"{p['name']}, about {int(round(p['distance_m'], -1))} metres {p['direction']}" for p in places[:3]]
        return "Nearby in map data: " + "; ".join(parts) + ". Map data may be out of date."
