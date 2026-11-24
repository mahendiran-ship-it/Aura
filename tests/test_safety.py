import re

from aura.safety import safety_rules as rules
from aura.safety.alert_manager import AlertManager
from aura.safety.hazard_detector import HazardDetector
from aura.types import Alert, AlertLevel
from tests.conftest import make_track

FORBIDDEN = re.compile(r"\b(is|it'?s|are|seems?|looks?)\b[^.]*\b(safe|clear)\b", re.I)


def test_level_mapping(safety_cfg):
    f = lambda s: rules.level_for_score(s, safety_cfg)  # noqa: E731
    assert f(0.0) == AlertLevel.INFO
    assert f(0.3) == AlertLevel.LOW
    assert f(0.6) == AlertLevel.MEDIUM
    assert f(0.9) == AlertLevel.HIGH
    assert f(1.3) == AlertLevel.CRITICAL


def test_nearer_is_riskier(safety_cfg):
    near = make_track("chair", proximity="near", score=0.9)
    far = make_track("chair", proximity="far", score=0.1)
    assert rules.risk_score(near, safety_cfg) > rules.risk_score(far, safety_cfg)


def test_vehicle_riskier_than_cup(safety_cfg):
    car = make_track("car", score=0.6)
    cup = make_track("cup", score=0.6)
    assert rules.risk_score(car, safety_cfg) > rules.risk_score(cup, safety_cfg)


def test_approaching_riskier_than_stationary_or_receding(safety_cfg):
    base = dict(label="car", score=0.6)
    appr = make_track(growth=0.2, **base)
    still = make_track(growth=0.0, **base)
    away = make_track(growth=-0.2, **base)
    s = lambda t: rules.risk_score(t, safety_cfg)  # noqa: E731
    assert appr.movement == "approaching" and away.movement == "moving away"
    assert s(appr) > s(still) > s(away)


def test_side_objects_riskier_in_walking_zone(safety_cfg):
    centre = make_track("chair", in_zone=True)
    side = make_track("chair", horizontal="far right", in_zone=False)
    assert rules.risk_score(centre, safety_cfg) > rules.risk_score(side, safety_cfg)


def test_unreliable_tracks_score_lower(safety_cfg):
    solid = make_track("chair", hits=20, conf=0.95)
    shaky = make_track("chair", hits=3, conf=0.5)
    assert rules.risk_score(solid, safety_cfg) > rules.risk_score(shaky, safety_cfg)


def test_config_changes_scoring(safety_cfg):
    t = make_track("chair", score=0.8)
    before = rules.risk_score(t, safety_cfg)
    safety_cfg["class_weights"]["chair"] = 1.0
    assert rules.risk_score(t, safety_cfg) > before


def test_hazard_detector_flags_close_car_with_hedged_wording(safety_cfg):
    hd = HazardDetector(safety_cfg)
    car = make_track("car", horizontal="right", proximity="near", score=0.9, in_zone=False, growth=0.2)
    alerts = hd.evaluate([car], 480)
    assert alerts and alerts[0].level >= AlertLevel.MEDIUM
    assert "Possible approaching car" in alerts[0].message and "right" in alerts[0].message
    assert "Distance is approximate" in alerts[0].message or alerts[0].level < AlertLevel.HIGH


def test_low_obstacle_wording(safety_cfg):
    hd = HazardDetector(safety_cfg)
    from aura.types import BBox
    t = make_track("chair", score=0.9, bbox=BBox(280, 400, 360, 470))
    msgs = [a.message for a in hd.evaluate([t], 480)]
    assert any(m.startswith("Low obstacle detected in the walking area") for m in msgs)


def test_ignores_low_confidence_and_short_tracks(safety_cfg):
    hd = HazardDetector(safety_cfg)
    assert hd.evaluate([make_track("car", conf=0.3, score=0.9)], 480) == []
    assert hd.evaluate([make_track("car", hits=1, score=0.9)], 480) == []
    assert hd.evaluate([make_track("car", score=0.9, misses=2)], 480) == []


def test_no_detections_produces_no_alerts_and_never_says_safe(safety_cfg):
    assert HazardDetector(safety_cfg).evaluate([], 480) == []


def test_no_runtime_message_claims_safety(safety_cfg):
    hd = HazardDetector(safety_cfg)
    for label in ["car", "person", "chair", "bicycle", "dog", "bottle"]:
        for prox, sc in (("near", 0.95), ("medium", 0.4), ("far", 0.1)):
            for a in hd.evaluate([make_track(label, proximity=prox, score=sc)], 480):
                assert not FORBIDDEN.search(a.message), a.message
    for text in (rules.CROSSING_DISCLAIMER, rules.GENERAL_DISCLAIMER, rules.DEGRADED_VIEW):
        assert "safe to cross" not in text.lower() or "cannot" in text.lower()


def test_crossing_disclaimer_is_exact():
    assert rules.CROSSING_DISCLAIMER == ("I cannot reliably determine whether it is safe to cross. "
                                         "Please stop and use your established crossing method.")


def test_sensor_levels(safety_cfg):
    assert rules.sensor_level(200, safety_cfg) is None
    assert rules.sensor_level(120, safety_cfg) == AlertLevel.LOW
    assert rules.sensor_level(20, safety_cfg) == AlertLevel.CRITICAL
    alerts = HazardDetector(safety_cfg).from_distance_sensor(25, "ahead")
    assert alerts[0].level == AlertLevel.CRITICAL and alerts[0].source == "sensor"


# ---------------- alert manager ----------------

def alert(level=AlertLevel.MEDIUM, key="car:right", msg="Possible car on your right."):
    return Alert(level, msg, key)


def test_cooldown_suppresses_repeats_then_allows(safety_cfg, clock):
    am = AlertManager(safety_cfg, clock)
    assert am.process([alert()]) is not None
    clock.advance(5)
    assert am.process([alert()]) is None          # inside 10 s MEDIUM cooldown
    clock.advance(6)
    assert am.process([alert()]) is not None      # cooldown over


def test_escalation_breaks_through_cooldown(safety_cfg, clock):
    am = AlertManager(safety_cfg, clock)
    am.process([alert(AlertLevel.LOW)])
    clock.advance(2)
    out = am.process([alert(AlertLevel.HIGH)])
    assert out is not None and out.level == AlertLevel.HIGH


def test_different_keys_are_independent_but_rate_limited(safety_cfg, clock):
    am = AlertManager(safety_cfg, clock)
    assert am.process([alert(key="a")]) is not None
    clock.advance(0.5)
    assert am.process([alert(key="b")]) is None   # global minimum gap
    clock.advance(1.5)
    assert am.process([alert(key="b")]) is not None


def test_critical_ignores_global_gap(safety_cfg, clock):
    am = AlertManager(safety_cfg, clock)
    am.process([alert(key="a")])
    clock.advance(0.2)
    assert am.process([alert(AlertLevel.CRITICAL, key="b")]) is not None


def test_only_most_severe_is_spoken(safety_cfg, clock):
    am = AlertManager(safety_cfg, clock)
    out = am.process([alert(AlertLevel.LOW, key="a"), alert(AlertLevel.HIGH, key="b")])
    assert out.key == "b"


def test_levels_below_threshold_stay_silent(safety_cfg, clock):
    safety_cfg["min_spoken_level"] = "HIGH"
    am = AlertManager(safety_cfg, clock)
    assert am.process([alert(AlertLevel.MEDIUM)]) is None
    assert am.process([alert(AlertLevel.INFO)]) is None
    assert am.process([]) is None


def test_reset_clears_cooldowns(safety_cfg, clock):
    am = AlertManager(safety_cfg, clock)
    am.process([alert()])
    am.reset()
    assert am.process([alert()]) is not None
