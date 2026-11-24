"""Turn-by-turn guidance from a Route and the current location.

Honest limits (also in docs/SAFETY.md): GPS error is typically 3-10 m and
much worse near buildings, so guidance is advisory. This module never
says a crossing or path is safe, and tells the user crossings are theirs
to judge.
"""

from __future__ import annotations

from typing import Optional

from aura.navigation.gps import Location
from aura.navigation.map_engine import Route
from aura.utils.helpers import bearing_deg, compass_name, haversine_m

ARRIVAL_RADIUS_M = 25.0
STEP_REACHED_M = 15.0
OFF_ROUTE_M = 60.0


class RouteGuidance:
    def __init__(self, route: Route) -> None:
        self.route = route
        self._step = 0
        self._announced_step = -1
        self.arrived = False

    def distance_to_destination(self, loc: Location) -> float:
        return haversine_m(loc.lat, loc.lon, self.route.dest_lat, self.route.dest_lon)

    def summary(self) -> str:
        mins = max(1, round(self.route.duration_s / 60))
        dist = self.route.distance_m
        d = f"{int(round(dist, -1))} metres" if dist < 1000 else f"{dist / 1000:.1f} kilometres"
        return (f"Route to {self.route.destination_name}: about {d}, roughly {mins} minutes on foot. "
                "Route data comes from OpenStreetMap and may be incomplete. "
                "You are responsible for judging crossings and obstacles.")

    def update(self, loc: Location) -> Optional[str]:
        """Return a spoken instruction when there is something new to say."""
        if self.arrived:
            return None
        dist = self.distance_to_destination(loc)
        if dist <= ARRIVAL_RADIUS_M:
            self.arrived = True
            return (f"You should be near {self.route.destination_name}, within about "
                    f"{int(dist)} metres. I cannot confirm the exact entrance.")
        steps = self.route.steps
        while self._step < len(steps) - 1:
            nxt = steps[self._step + 1]
            if haversine_m(loc.lat, loc.lon, nxt.lat, nxt.lon) <= STEP_REACHED_M:
                self._step += 1
            else:
                break
        if self._step == self._announced_step:
            return None
        self._announced_step = self._step
        if self._step + 1 < len(steps):
            nxt = steps[self._step + 1]
            d = haversine_m(loc.lat, loc.lon, nxt.lat, nxt.lon)
            return f"{steps[self._step].instruction}. Next: {nxt.instruction.lower()} in about {int(round(d, -1))} metres."
        return f"{steps[self._step].instruction}. About {int(round(dist, -1))} metres to go."

    def direction_hint(self, loc: Location) -> str:
        """Compass direction and distance to the destination (straight line)."""
        b = bearing_deg(loc.lat, loc.lon, self.route.dest_lat, self.route.dest_lon)
        d = self.distance_to_destination(loc)
        return (f"{self.route.destination_name} is about {int(round(d, -1))} metres to the "
                f"{compass_name(b)} in a straight line. The walking route may differ.")
