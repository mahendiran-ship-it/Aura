"""Geocoding and walking routes from open services. REQUIRES INTERNET.

Nominatim (OpenStreetMap) finds a place by name; OSRM computes a foot
route over OSM data. Both are public community servers with usage
policies (fair use, identify your app via ``user_agent``). For heavy use
host your own instances and change the URLs in settings.yaml.

If a service cannot be reached, AURA says so. It never invents a route.
OSM pedestrian data can be incomplete or wrong: see docs/SAFETY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aura.utils.logger import get_logger

log = get_logger("map")


class MapError(Exception):
    """Mapping failed; the message is safe to speak."""


@dataclass
class RouteStep:
    instruction: str
    distance_m: float
    lat: float  # where the manoeuvre happens
    lon: float


@dataclass
class Route:
    destination_name: str
    dest_lat: float
    dest_lon: float
    distance_m: float
    duration_s: float
    steps: List[RouteStep] = field(default_factory=list)


_MODIFIERS = {"left": "turn left", "right": "turn right", "slight left": "bear left",
              "slight right": "bear right", "sharp left": "turn sharply left",
              "sharp right": "turn sharply right", "straight": "continue straight",
              "uturn": "make a U-turn"}


def step_instruction(maneuver: Dict[str, Any], street: str) -> str:
    """OSRM manoeuvre -> short spoken instruction."""
    kind = maneuver.get("type", "")
    mod = maneuver.get("modifier", "")
    on = f" onto {street}" if street else ""
    if kind == "depart":
        return f"Start walking{(' on ' + street) if street else ''}"
    if kind == "arrive":
        return "You are arriving at the destination area"
    if kind in ("roundabout", "rotary"):
        return f"Enter the roundabout{on}"
    text = f"{_MODIFIERS.get(mod, 'continue')}{on}"
    return text[:1].upper() + text[1:]  # not str.capitalize(): that would lowercase street names


class MapEngine:
    def __init__(self, cfg: Dict[str, Any], session: Any = None) -> None:
        self.geocoder = str(cfg.get("geocoder_url", "")).rstrip("/")
        self.router = str(cfg.get("router_url", "")).rstrip("/")
        self.user_agent = str(cfg.get("user_agent", "AURA-assistive"))
        self.timeout = float(cfg.get("timeout_s", 10))
        self._session = session

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            if self._session is None:
                import requests
                self._session = requests
            resp = self._session.get(url, params=params, timeout=self.timeout,
                                     headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Map request failed: %s", type(exc).__name__)
            raise MapError("I could not reach the map service. Navigation needs an internet "
                           "connection.") from exc

    def geocode(self, query: str, near: Optional[tuple] = None) -> Dict[str, Any]:
        """Find a place by name. Raises MapError if none found."""
        params: Dict[str, Any] = {"q": query, "format": "jsonv2", "limit": 1}
        if near:  # prefer results close to the user (~+-0.2 degrees)
            lat, lon = near
            params.update({"viewbox": f"{lon - 0.2},{lat + 0.2},{lon + 0.2},{lat - 0.2}", "bounded": 0})
        data = self._get(f"{self.geocoder}/search", params)
        if not data:
            raise MapError(f"I could not find a place called {query}.")
        top = data[0]
        return {"name": top.get("display_name", query).split(",")[0], "lat": float(top["lat"]),
                "lon": float(top["lon"])}

    def walking_route(self, start: tuple, dest_name: str, dest: tuple) -> Route:
        """Compute a foot route. Raises MapError when none exists."""
        (slat, slon), (dlat, dlon) = start, dest
        data = self._get(f"{self.router}/route/v1/foot/{slon},{slat};{dlon},{dlat}",
                         {"overview": "false", "steps": "true"})
        if data.get("code") != "Ok" or not data.get("routes"):
            raise MapError("I could not find a walking route to that place.")
        r = data["routes"][0]
        steps: List[RouteStep] = []
        for leg in r.get("legs", []):
            for s in leg.get("steps", []):
                lon, lat = s["maneuver"]["location"]
                steps.append(RouteStep(step_instruction(s["maneuver"], s.get("name", "")),
                                       float(s.get("distance", 0)), float(lat), float(lon)))
        return Route(dest_name, dlat, dlon, float(r["distance"]), float(r["duration"]), steps)
