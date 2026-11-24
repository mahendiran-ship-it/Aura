"""Integration tests: orchestrator + assistant + safety + memory with fake hardware."""

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from aura.audio.audio_manager import AudioManager
from aura.audio.speech_output import ConsoleTTS, TTSBackend, create_tts
from aura.core.orchestrator import Orchestrator
from aura.intelligence.llm import LLMClient
from aura.memory.memory_manager import MemoryManager
from aura.safety import safety_rules as rules
from aura.safety.alert_manager import AlertManager
from aura.types import AlertLevel
from aura.vision.camera import CameraError, CameraSource
from aura.vision.detector import DetectorError, ModelNotFoundError
from aura.vision.ocr import OCRReader
from tests.conftest import FakeClock, det
from tests.test_camera_ocr import FakeEngine
from tests.test_intelligence import FakeSession

W, H = 640, 480
RNG = np.random.default_rng(1)
GOOD_FRAME = RNG.integers(40, 220, (H, W, 3), dtype=np.uint8)   # textured, mid-brightness => quality OK


class RecordingTTS(TTSBackend):
    name = "rec"

    def __init__(self):
        self.said, self.stops = [], 0

    def available(self):
        return True

    def speak(self, text):
        self.said.append(text)

    def stop(self):
        self.stops += 1


class ScriptedDetector:
    """Stands in for the neural network only; everything downstream is real."""

    def __init__(self, script=None, ready=True):
        self.script, self.i, self.ready = script or [], 0, ready

    def load(self):
        if not self.ready:
            raise ModelNotFoundError("Object detection model not found. Run: python scripts/download_models.py")

    def detect(self, frame):
        out = self.script[min(self.i, len(self.script) - 1)] if self.script else []
        self.i += 1
        return out


class FakeCam(CameraSource):
    def __init__(self, frames=None, fail_after=None):
        self.frames, self.n, self.fail_after, self.opened = frames, 0, fail_after, False

    description = "fake camera"

    def open(self):
        self.opened = True

    def read(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise CameraError("camera unplugged")
        return GOOD_FRAME

    def close(self):
        self.opened = False


def make(settings, safety_cfg, script=None, detector_ready=True, camera=None, llm=None, ocr=None, tts=None):
    tts = tts or RecordingTTS()
    s = json.loads(json.dumps(settings))
    s["vision"]["process_every_n_frames"] = 1
    orch = Orchestrator(s, safety_cfg, camera=camera or FakeCam(), detector=ScriptedDetector(script, detector_ready),
                        audio=AudioManager(tts, threaded=False), memory=MemoryManager(":memory:"),
                        ocr=ocr or OCRReader(engine=FakeEngine(["EXIT"], [95])), llm=llm)
    orch.tts = tts
    if detector_ready:
        orch.state.set(detector_ready=True, camera_ok=True)
    orch.state.set(camera_ok=True)
    return orch


def feed(orch, n):
    for _ in range(n):
        orch.process_frame(GOOD_FRAME)


def approaching_car(i):
    s = i * 12
    return [det("car", 430 - s, 200 - s, 530 + s, 330 + s)]


# ------------------------------------------------------------- silence & alerts
def test_stays_silent_when_nothing_happens(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[]])
    feed(o, 10)
    assert o.tts.said == []


def test_calm_distant_object_does_not_trigger_alert(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[det("cup", 10, 10, 30, 30)]])
    feed(o, 10)
    assert o.tts.said == []


def test_approaching_car_alerts_once_with_hedged_wording(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[approaching_car(i) for i in range(10)])
    clock = FakeClock()
    o.alerts = AlertManager(safety_cfg, clock)
    feed(o, 10)
    msgs = [m for m in o.tts.said if "car" in m]
    assert msgs, o.tts.said
    assert msgs[0].startswith("Possible") and "safe" not in msgs[0].lower()
    assert len(msgs) <= 2   # de-duplicated (maybe one escalation), not one per frame


def test_alert_cooldown_across_frames(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[det("chair", 250, 150, 400, 470)]])
    clock = FakeClock()
    o.alerts = AlertManager(safety_cfg, clock)
    feed(o, 6)
    first = len(o.tts.said)
    assert first >= 1
    feed(o, 20)                       # same obstacle, no time passes
    assert len(o.tts.said) == first
    clock.advance(60)
    feed(o, 3)
    assert len(o.tts.said) == first + 1


def test_pause_stops_proactive_alerts_and_warns_user(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[det("chair", 250, 150, 400, 470)]])
    resp = o.handle_text("stop assistance")
    assert "will not warn you about obstacles" in resp.text and "cane" in resp.text
    n = len(o.tts.said)
    feed(o, 10)
    assert len(o.tts.said) == n
    o.handle_text("resume")
    feed(o, 5)
    assert len(o.tts.said) > n + 1


def test_poor_camera_view_warns_and_taints_descriptions(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[det("chair", 250, 150, 400, 470)]])
    dark = np.full((H, W, 3), 3, np.uint8)
    for _ in range(4):
        o.process_frame(dark)
    assert sum(rules.DEGRADED_VIEW in m for m in o.tts.said) == 1     # warned once, not every frame
    text = o.handle_text("describe").text
    assert text.startswith("The camera view is")


# ------------------------------------------------------------- commands
def test_crossing_question_gets_exact_refusal(settings, safety_cfg):
    o = make(settings, safety_cfg)
    for q in ("Is it safe to cross?", "can I cross now", "is the road clear to cross"):
        assert o.handle_text(q).text == rules.CROSSING_DISCLAIMER


def test_describe_and_front_use_real_pipeline(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[det("chair", 40, 200, 120, 330), det("dining table", 220, 180, 420, 400)]])
    feed(o, 5)
    d = o.handle_text("describe").text
    assert "table" in d and "chair" in d and "left" in d
    assert "table" in o.handle_text("what is in front of me").text


def test_describe_with_no_objects_never_says_clear(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[]])
    feed(o, 3)
    text = o.handle_text("describe").text
    assert "does not mean the area is clear" in text


def test_no_detector_degrades_gracefully(settings, safety_cfg):
    o = make(settings, safety_cfg, detector_ready=False)
    o.state.set(camera_ok=True)
    o.start(announce=True)
    try:
        assert any("detection is not running" in m.lower() or "not running" in m for m in o.tts.said)
        assert "Object detection is not running" in o.handle_text("describe").text
        assert "You can say" in o.handle_text("help").text           # other features still work
        assert "bag" in o.handle_text("where did I last see my bag").text
    finally:
        o.stop()


def test_find_flow_guides_and_can_be_cancelled(settings, safety_cfg):
    bottle = [det("bottle", 380, 250, 420, 330)]
    o = make(settings, safety_cfg, script=[bottle])
    feed(o, 5)
    r = o.handle_text("Find my bottle")
    assert r.text.startswith("I see a bottle") and "to your right" in r.text
    assert "bottle" in o.handle_text("guide me toward it").text
    assert o.handle_text("cancel").text == "Cancelled."
    assert o.state.get("search_target") is None
    assert o.handle_text("cancel").text == "Nothing to cancel."


def test_find_unsupported_object_explains_limit_and_checks_memory(settings, safety_cfg):
    o = make(settings, safety_cfg)
    o.memory.objects.record("cell phone", "desk", "ahead", 0.9)
    text = o.handle_text("find my keys").text
    assert "cannot recognize keys" in text and not o.search.active
    assert "cannot recognize door" in o.handle_text("where is the door").text


def test_last_seen_current_vs_remembered(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[]])
    clock = FakeClock()
    o.memory.objects._clock = clock
    o.memory.objects.record("backpack", "living room", "to your left", 0.9, near_object="sofa")
    clock.advance(700)
    text = o.handle_text("Where did I last see my bag?").text
    assert "near the sofa" in text and "living room" in text and "may have been moved" in text
    assert "no record" in o.handle_text("where did I last see my wallet").text
    o.script = None


def test_memory_commands_end_to_end(settings, safety_cfg):
    o = make(settings, safety_cfg)
    assert "kitchen" in o.handle_text("I am in the kitchen").text
    assert "Saved this location" in o.handle_text("remember this location").text
    o.handle_text("remember that my keys are on the shelf")
    hist = o.handle_text("memory history").text
    assert "keys" in hist and "user-confirmed" in hist
    assert "Deleted 1 memory" in o.handle_text("forget my keys").text
    o.memory.objects.record("cup", "kitchen", "ahead", 0.9)
    first = o.handle_text("clear all memory").text
    assert "permanently delete" in first and o.memory.objects.count() == 1   # not deleted yet
    assert "All memory cleared" in o.handle_text("clear all memory").text
    assert o.memory.objects.count() == 0


def test_cancel_aborts_pending_memory_wipe(settings, safety_cfg):
    o = make(settings, safety_cfg)
    o.memory.objects.record("cup", "kitchen", "ahead", 0.9)
    o.handle_text("clear all memory")
    o.handle_text("cancel")
    assert "permanently delete" in o.handle_text("clear all memory").text
    assert o.memory.objects.count() == 1


def test_read_uses_ocr_and_reports_failures(settings, safety_cfg):
    o = make(settings, safety_cfg)
    o._latest_frame = GOOD_FRAME
    assert o.handle_text("read this").text == "EXIT"
    bad = make(settings, safety_cfg, ocr=OCRReader(engine=FakeEngine([], [], raises=RuntimeError("x"))))
    bad._latest_frame = GOOD_FRAME
    assert "failed" in bad.handle_text("read this").text


def test_audio_controls(settings, safety_cfg):
    o = make(settings, safety_cfg)
    o.handle_text("help")
    before = len(o.tts.said)
    o.handle_text("repeat")
    assert len(o.tts.said) == before + 1 and o.tts.said[-1] == o.tts.said[-2]
    o.handle_text("stop talking")
    assert o.tts.stops >= 1


def test_emergency_is_honest_about_no_calling(settings, safety_cfg):
    settings["user"]["emergency_contact_phone"] = "5551234"
    o = make(settings, safety_cfg)
    text = o.handle_text("emergency").text
    assert "cannot call" in text and "5 5 5 1 2 3 4" in text
    settings["user"]["emergency_contact_phone"] = ""


def test_unknown_input_is_handled(settings, safety_cfg):
    o = make(settings, safety_cfg)
    assert o.handle_text("").text == ""
    assert o.handle_text("hmm").text == ""


# ------------------------------------------------------------- AI backend
def test_ai_failure_does_not_break_basic_features(settings, safety_cfg):
    cfg = {**settings["ai"], "enabled": True}
    llm = LLMClient(cfg, session=FakeSession(exc=ConnectionError("down")))
    o = make(settings, safety_cfg, script=[[det("chair", 40, 200, 120, 330)]], llm=llm)
    feed(o, 5)
    ans = o.handle_text("what is the difference between these two objects?").text
    assert "could not be reached" in ans and "describe your surroundings" in ans
    assert "chair" in o.handle_text("describe").text


def test_ai_answer_is_used_when_available_and_cloud_is_announced(settings, safety_cfg):
    cfg = {**settings["ai"], "enabled": True, "endpoint": "https://api.example.com/v1", "allow_cloud": True}
    from tests.test_intelligence import OK
    llm = LLMClient(cfg, session=FakeSession(OK))
    o = make(settings, safety_cfg, llm=llm)
    ans = o.handle_text("which direction is the door?").text
    assert ans.startswith("Using the cloud assistant.") and "door" in ans


def test_ai_off_by_default(settings, safety_cfg):
    o = make(settings, safety_cfg)
    ans = o.handle_text("what is the capital of france").text
    assert "disabled" in ans and "describe your surroundings" in ans
    assert "AI assistant is off" in o.handle_text("status").text


# ------------------------------------------------------------- camera failure in the live loop
def test_camera_failure_is_announced_and_vision_stops(settings, safety_cfg):
    o = make(settings, safety_cfg, script=[[det("chair", 250, 150, 400, 470)]], camera=FakeCam(fail_after=3))
    o.start(announce=False)
    deadline = time.time() + 3
    while time.time() < deadline and o.state.get("camera_ok"):
        time.sleep(0.02)
    try:
        assert not o.state.get("camera_ok")
        assert rules.CAMERA_LOST in o.tts.said
        assert o.get_tracks() == []                       # stale objects are not kept
        assert "camera is not working" in o.handle_text("describe").text
    finally:
        o.stop()


def test_camera_that_cannot_open_is_reported(settings, safety_cfg):
    class Dead(FakeCam):
        def open(self):
            raise CameraError("no camera found")
    o = make(settings, safety_cfg, camera=Dead())
    o.state.set(camera_ok=False)
    o.start(announce=True)
    try:
        assert any("camera could not be opened" in m for m in o.tts.said)
        assert not o.state.get("camera_ok")
    finally:
        o.stop()


def test_detector_runtime_failure_is_announced(settings, safety_cfg):
    class Exploding(ScriptedDetector):
        def detect(self, frame):
            raise DetectorError("inference failed")
    o = make(settings, safety_cfg)
    o.detector = Exploding()
    o.process_frame(GOOD_FRAME)
    assert rules.NO_DETECTOR in o.tts.said and not o.state.get("detector_ready")


# ------------------------------------------------------------- privacy
def test_processing_writes_no_images_or_video(settings, safety_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = make(settings, safety_cfg, script=[[det("chair", 250, 150, 400, 470)]])
    feed(o, 6)
    o.handle_text("describe")
    assert list(tmp_path.iterdir()) == []


def test_user_speech_is_not_stored_in_state(settings, safety_cfg):
    o = make(settings, safety_cfg)
    o.handle_text("find my super secret thing")
    assert "secret" not in json.dumps(o.state.snapshot()) and o.state.get("last_command") == "find"


def test_tts_backend_fallback_is_console():
    assert isinstance(create_tts({"tts_backend": "console"}), ConsoleTTS)
    assert isinstance(create_tts({"tts_backend": "does-not-exist"}), ConsoleTTS)
    assert isinstance(create_tts({"tts_backend": "termux"}), ConsoleTTS)   # termux-tts-speak absent here


# ------------------------------------------------------------- audio manager
def test_audio_manager_empty_and_repeat_and_priority_interrupt():
    tts = RecordingTTS()
    am = AudioManager(tts, threaded=False)
    am.repeat()
    assert tts.said == ["There is nothing to repeat."]
    am.speak("hello")
    am.speak("danger", AlertLevel.CRITICAL)
    assert tts.said[-1] == "danger" and tts.stops >= 1
    am.repeat()
    assert tts.said[-1] == "danger"
    am.speak("   ")
    assert tts.said[-1] == "danger"


def test_threaded_audio_manager_speaks_and_closes():
    tts = RecordingTTS()
    am = AudioManager(tts, threaded=True)
    am.speak("one")
    deadline = time.time() + 2
    while not tts.said and time.time() < deadline:
        time.sleep(0.01)
    am.close()
    assert tts.said == ["one"]


# ------------------------------------------------------------- HTTP API
def test_api_requires_token_and_runs_commands(settings, safety_cfg, monkeypatch):
    from aura.interfaces.api import ApiConfigError, make_server
    monkeypatch.delenv("AURA_API_TOKEN", raising=False)
    o = make(settings, safety_cfg)
    with pytest.raises(ApiConfigError):
        make_server(o, "127.0.0.1", 0, "AURA_API_TOKEN")
    monkeypatch.setenv("AURA_API_TOKEN", "tok123")
    server = make_server(o, "127.0.0.1", 0, "AURA_API_TOKEN")
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/status")
        assert e.value.code == 401
        req = urllib.request.Request(base + "/command", data=json.dumps({"text": "is it safe to cross"}).encode(),
                                     headers={"Authorization": "Bearer tok123"}, method="POST")
        assert json.load(urllib.request.urlopen(req))["response"] == rules.CROSSING_DISCLAIMER
        req = urllib.request.Request(base + "/status", headers={"Authorization": "Bearer tok123"})
        assert "camera" in json.load(urllib.request.urlopen(req))
        bad = urllib.request.Request(base + "/command", data=b"{", headers={"Authorization": "Bearer tok123"},
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as e2:
            urllib.request.urlopen(bad)
        assert e2.value.code == 400
    finally:
        server.shutdown()
