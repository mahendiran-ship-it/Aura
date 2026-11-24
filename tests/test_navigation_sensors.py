import pytest

from aura.navigation.gps import (FixedLocationProvider, Location, LocationError, NoLocationProvider,
                                 create_location_provider)
from aura.navigation.landmark_engine import LandmarkEngine
from aura.navigation.map_engine import MapEngine, MapError, step_instruction
from aura.navigation.route_guidance import RouteGuidance
from aura.sensors.base import DistanceReading, DistanceSensor, SensorHub
from aura.sensors.serial_sensor import SerialDistanceSensor, parse_line
from aura.utils.helpers import bearing_deg, compass_name, haversine_m


class Resp:
    def __init__(self, data, ok=True):
        self.data, self.ok = data, ok

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self.data


class Sess:
    def __init__(self, data=None, exc=None):
        self.data, self.exc, self.calls = data, exc, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params, headers))
        if self.exc:
            raise self.exc
        return Resp(self.data)

    post = None


CFG = {"geocoder_url": "https://geo.test", "router_url": "https://route.test", "user_agent": "UA", "timeout_s": 3}

OSRM = {"code": "Ok", "routes": [{"distance": 420.0, "duration": 360.0, "legs": [{"steps": [
    {"maneuver": {"type": "depart", "location": [77.0, 12.0]}, "name": "Main St", "distance": 200},
    {"maneuver": {"type": "turn", "modifier": "left", "location": [77.002, 12.0]}, "name": "Park Rd", "distance": 220},
    {"maneuver": {"type": "arrive", "location": [77.002, 12.002]}, "name": "", "distance": 0}]}]}]}


# ------------------------------------------------------------- map engine
def test_geocode_and_route_parse():
    eng = MapEngine(CFG, session=Sess([{"display_name": "Central Park, NYC", "lat": "12.002", "lon": "77.002"}]))
    place = eng.geocode("central park", near=(12.0, 77.0))
    assert place["name"] == "Central Park" and place["lat"] == pytest.approx(12.002)
    eng2 = MapEngine(CFG, session=Sess(OSRM))
    route = eng2.walking_route((12.0, 77.0), "Central Park", (12.002, 77.002))
    assert route.distance_m == 420
    assert [s.instruction for s in route.steps][1] == "Turn left onto Park Rd"
    url, _, headers = eng2._session.calls[0]
    assert "/route/v1/foot/77.0,12.0;77.002,12.002" in url and headers["User-Agent"] == "UA"


def test_unknown_place_and_no_route_raise_mapError():
    with pytest.raises(MapError):
        MapEngine(CFG, session=Sess([])).geocode("nowhereland")
    with pytest.raises(MapError):
        MapEngine(CFG, session=Sess({"code": "NoRoute", "routes": []})).walking_route((0, 0), "x", (1, 1))


def test_offline_map_failure_never_invents_a_route():
    with pytest.raises(MapError) as e:
        MapEngine(CFG, session=Sess(exc=ConnectionError("offline"))).geocode("x")
    assert "internet" in str(e.value)


def test_step_instruction():
    assert step_instruction({"type": "turn", "modifier": "right"}, "Oak Ave") == "Turn right onto Oak Ave"
    assert step_instruction({"type": "arrive"}, "").startswith("You are arriving")


# ------------------------------------------------------------- guidance
def make_guidance():
    route = MapEngine(CFG, session=Sess(OSRM)).walking_route((12.0, 77.0), "Central Park", (12.002, 77.002))
    return RouteGuidance(route)


def test_guidance_progresses_and_arrives():
    g = make_guidance()
    first = g.update(Location(12.0, 77.0))
    assert "Next:" in first and g.update(Location(12.0, 77.0)) is None  # no repetition
    nxt = g.update(Location(12.0, 77.00201))                          # reached the turn
    assert nxt and "Turn left" in nxt
    arrived = g.update(Location(12.00201, 77.00201))
    assert "near Central Park" in arrived and "cannot confirm" in arrived and g.arrived
    assert g.update(Location(12.0, 77.0)) is None


def test_summary_states_data_limits_and_crossings():
    s = make_guidance().summary()
    assert "OpenStreetMap" in s and "crossings" in s and "safe" not in s.lower()


def test_direction_hint():
    g = make_guidance()
    assert "north-east" in g.direction_hint(Location(11.999, 76.999))


# ------------------------------------------------------------- gps providers
def test_no_location_by_default_and_fixed_provider():
    with pytest.raises(LocationError):
        NoLocationProvider().get_location()
    assert isinstance(create_location_provider({}), NoLocationProvider)
    assert FixedLocationProvider(1.5, 2.5).get_location().source == "fixed"
    with pytest.raises(LocationError):
        FixedLocationProvider(None, 2.0)


def test_termux_provider_without_binary(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda n: None)
    from aura.navigation.gps import TermuxLocationProvider
    with pytest.raises(LocationError):
        TermuxLocationProvider().get_location()


# ------------------------------------------------------------- landmarks
def test_landmarks_sorted_and_offline_error():
    class S:
        def post(self, url, data=None, timeout=None, headers=None):
            return Resp({"elements": [
                {"lat": 12.001, "lon": 77.0, "tags": {"name": "Far Cafe"}},
                {"lat": 12.0002, "lon": 77.0, "tags": {"name": "Near Pharmacy"}},
                {"lat": 12.0, "lon": 77.0, "tags": {}}]})

    out = LandmarkEngine(session=S()).nearby(12.0, 77.0)
    assert [p["name"] for p in out] == ["Near Pharmacy", "Far Cafe"]

    class Bad:
        def post(self, *a, **k):
            raise ConnectionError

    with pytest.raises(RuntimeError):
        LandmarkEngine(session=Bad()).describe(12.0, 77.0)


def test_geo_math():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111195, rel=0.01)
    assert compass_name(bearing_deg(0, 0, 1, 0)) == "north" and compass_name(bearing_deg(0, 0, 0, 1)) == "east"


# ------------------------------------------------------------- sensors
def test_serial_line_parsing_rejects_garbage():
    r = parse_line("DIST:ahead:123.4\n")
    assert r.distance_cm == 123.4 and r.direction == "ahead"
    for bad in ("", "HELLO", "DIST:ahead:abc", "DIST:up:100", "DIST:ahead:0", "DIST:ahead:99999", "DIST:ahead"):
        assert parse_line(bad) is None


class FakeSerial:
    def __init__(self, lines):
        self.lines, self.closed = list(lines), False

    @property
    def in_waiting(self):
        return len(self.lines)

    def readline(self):
        return self.lines.pop(0)

    def close(self):
        self.closed = True


def test_serial_sensor_reads_latest_valid_reading():
    s = SerialDistanceSensor("/dev/fake", serial_obj=FakeSerial([b"junk\n", b"DIST:ahead:80\n", b"DIST:ahead:55\n"]))
    assert s.read().distance_cm == 55
    s.close()
    assert s._ser.closed


def test_sensor_hub_ignores_stale_and_broken_sensors(monkeypatch):
    import time as _t

    class Good(DistanceSensor):
        sensor_id = "good"

        def read(self):
            return DistanceReading(50, "ahead", "good", _t.time())

    class Stale(DistanceSensor):
        sensor_id = "stale"

        def read(self):
            return DistanceReading(50, "ahead", "stale", _t.time() - 60)

    class Broken(DistanceSensor):
        sensor_id = "broken"

        def read(self):
            raise OSError("unplugged")

    hub = SensorHub([Good(), Stale(), Broken()])
    assert [r.sensor_id for r in hub.poll()] == ["good"]
