import json

import pytest

from aura.audio.command_parser import Command, Intent
from aura.intelligence.context_engine import ContextEngine
from aura.intelligence.llm import SYSTEM_PROMPT, LLMClient, LLMUnavailable, is_local_endpoint
from aura.memory.database import Database
from aura.memory.object_memory import ObjectMemory
from aura.types import Alert, AlertLevel
from tests.conftest import make_track


# ------------------------------------------------------------- context engine
def build(cmd=None, tracks=(), alerts=(), search=None, memory=None, now_ts=1000.0, **kw):
    return ContextEngine().build(cmd, list(tracks), list(alerts), "kitchen", search, memory, now_ts, **kw)


def test_search_prioritises_target_objects():
    tracks = [make_track("bottle", "right", "medium", 0.4, track_id=1),
              make_track("chair", "left", "near", 0.9, track_id=2)]
    ctx = build(None, tracks, search="bottle")
    assert ctx.focus == "search"
    assert [o["object"] for o in ctx.relevant_objects] == ["bottle"]


def test_describe_command_gives_broad_context():
    tracks = [make_track("bottle", track_id=1), make_track("chair", track_id=2)]
    ctx = build(Command(Intent.DESCRIBE), tracks)
    assert ctx.focus == "command" and len(ctx.objects) == 2


def test_serious_alert_overrides_everything():
    alert = Alert(AlertLevel.HIGH, "Possible car on your right.", "car:right")
    ctx = build(Command(Intent.DESCRIBE), [make_track("car")], [alert], search="bottle")
    assert ctx.focus == "safety"


def test_quiet_when_nothing_happening():
    ctx = build(None, [make_track("chair")])
    assert ctx.focus == "quiet"
    assert ContextEngine.should_stay_silent(ctx, None, None) is True
    assert ContextEngine.should_stay_silent(ctx, None, "I see a bottle") is False


def test_memory_enters_context_with_freshness(clock):
    mem = ObjectMemory(Database(":memory:"), clock=clock)
    mem.record("cell phone", "desk", "to your left", 0.9)
    clock.advance(3000)
    ctx = build(Command(Intent.LAST_SEEN, "phone"), [], memory=mem.history(), now_ts=clock())
    assert ctx.memory[0]["freshness"] == "previously observed"


def test_hidden_tracks_excluded_and_llm_payload_has_no_pixels():
    tracks = [make_track("chair", track_id=1), make_track("cup", track_id=2, misses=3)]
    payload = build(Command(Intent.DESCRIBE), tracks).to_llm_payload()
    assert [o["object"] for o in payload["visible_objects"]] == ["chair"]
    blob = json.dumps(payload)
    assert "apparent_distance" in blob and "(approximate)" in blob
    assert "bbox" not in blob and "frame" not in blob.lower()


# ------------------------------------------------------------- LLM backend
class FakeResp:
    def __init__(self, payload=None, status_ok=True):
        self._p, self._ok = payload, status_ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json, timeout))
        if self.exc:
            raise self.exc
        return self.resp


LOCAL = {"enabled": True, "endpoint": "http://127.0.0.1:11434/v1", "model": "m", "timeout_s": 5,
         "api_key_env": "AURA_TEST_LLM_KEY", "allow_cloud": False}
OK = FakeResp({"choices": [{"message": {"content": " The door is on your left. "}}]})


def test_disabled_llm_is_unavailable():
    with pytest.raises(LLMUnavailable):
        LLMClient({**LOCAL, "enabled": False}).ask("q", {})


def test_llm_success_sends_only_structured_facts_and_safety_prompt(monkeypatch):
    monkeypatch.setenv("AURA_TEST_LLM_KEY", "sekret")
    sess = FakeSession(OK)
    out = LLMClient(LOCAL, session=sess).ask("Which direction is the door?", {"visible_objects": []})
    assert out == "The door is on your left."
    url, headers, body, timeout = sess.calls[0]
    assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer sekret"
    assert body["messages"][0]["content"] == SYSTEM_PROMPT and "NEVER say" in SYSTEM_PROMPT
    assert "FACTS" in body["messages"][1]["content"] and timeout == 5


@pytest.mark.parametrize("exc", [ConnectionError("refused"), TimeoutError("slow")])
def test_llm_network_failure_becomes_unavailable(exc):
    with pytest.raises(LLMUnavailable):
        LLMClient(LOCAL, session=FakeSession(exc=exc)).ask("q", {})


def test_llm_http_error_and_bad_json_and_empty():
    with pytest.raises(LLMUnavailable):
        LLMClient(LOCAL, session=FakeSession(FakeResp({}, status_ok=False))).ask("q", {})
    with pytest.raises(LLMUnavailable):
        LLMClient(LOCAL, session=FakeSession(FakeResp({"unexpected": 1}))).ask("q", {})
    with pytest.raises(LLMUnavailable):
        LLMClient(LOCAL, session=FakeSession(FakeResp({"choices": [{"message": {"content": "  "}}]}))).ask("q", {})


def test_cloud_endpoint_blocked_unless_allowed():
    cfg = {**LOCAL, "endpoint": "https://api.example.com/v1"}
    sess = FakeSession(OK)
    c = LLMClient(cfg, session=sess)
    assert c.is_cloud and "blocked" in c.status()
    with pytest.raises(LLMUnavailable):
        c.ask("q", {})
    assert sess.calls == []                      # nothing was sent
    allowed = LLMClient({**cfg, "allow_cloud": True}, session=sess)
    assert "cloud" in allowed.status() and allowed.ask("q", {})


def test_endpoint_locality():
    assert all(is_local_endpoint(u) for u in ("http://localhost:1", "http://127.0.0.1/v1",
                                              "http://192.168.1.20:8000", "http://10.0.0.5"))
    assert not any(is_local_endpoint(u) for u in ("https://api.openai.com/v1", "http://8.8.8.8"))


def test_llm_error_does_not_leak_key(monkeypatch):
    monkeypatch.setenv("AURA_TEST_LLM_KEY", "sekret")
    with pytest.raises(LLMUnavailable) as e:
        LLMClient(LOCAL, session=FakeSession(exc=RuntimeError("bad sekret"))).ask("q", {})
    assert "sekret" not in str(e.value)
