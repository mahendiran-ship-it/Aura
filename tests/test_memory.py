from aura.memory.database import Database
from aura.memory.memory_manager import MemoryManager
from aura.memory.object_memory import Freshness, ObjectMemory
from tests.conftest import FakeClock, make_track


def make_memory(clock, **kw) -> ObjectMemory:
    return ObjectMemory(Database(":memory:"), recent_seconds=600, stale_seconds=86400,
                        retention_days=30, write_interval_s=30, clock=clock, **kw)


def test_database_roundtrip_and_schema():
    db = Database(":memory:")
    db.execute("INSERT INTO personal_notes(kind,text,room,created_at) VALUES ('note','x','r',1)")
    assert db.query_one("SELECT text FROM personal_notes")["text"] == "x"


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "m.sqlite3")
    m1 = MemoryManager(path)
    m1.objects.record("backpack", "living room", "to your left", 0.9, near_object="sofa")
    m1.close()
    m2 = MemoryManager(path)
    rec = m2.objects.last_seen(["backpack"])
    assert rec is not None and rec.near_object == "sofa" and rec.room == "living room"
    m2.close()


def test_freshness_progression(clock):
    mem = make_memory(clock)
    mem.record("cell phone", "desk", "ahead", 0.9)
    assert mem.last_seen(["cell phone"]).freshness == Freshness.RECENT
    clock.advance(601)
    assert mem.last_seen(["cell phone"]).freshness == Freshness.PREVIOUS
    clock.advance(86400)
    assert mem.last_seen(["cell phone"]).freshness == Freshness.STALE


def test_user_confirmed_is_flagged_and_still_goes_stale(clock):
    mem = make_memory(clock)
    mem.record("cell phone", "desk", "on the shelf", 1.0, source="user", user_confirmed=True)
    assert mem.last_seen(["cell phone"]).freshness == Freshness.USER_CONFIRMED
    clock.advance(86401)
    assert mem.last_seen(["cell phone"]).freshness == Freshness.STALE


def test_unknown_when_no_record(clock):
    mem = make_memory(clock)
    assert mem.last_seen(["cell phone"]) is None
    assert "no record" in mem.describe(None, "phone").lower()


def test_stale_memory_is_never_described_as_current(clock):
    mem = make_memory(clock)
    mem.record("backpack", "living room", "to your left", 0.9, near_object="sofa")
    clock.advance(90000)
    text = mem.describe(mem.last_seen(["backpack"]), "bag")
    assert "probably moved" in text
    assert "currently" not in text.lower()


def test_recent_memory_still_says_cannot_confirm(clock):
    mem = make_memory(clock)
    mem.record("backpack", "living room", "to your left", 0.9, near_object="sofa")
    text = mem.describe(mem.last_seen(["backpack"]), "bag")
    assert "near the sofa" in text and "can't confirm" in text


def test_duplicate_writes_are_throttled(clock):
    mem = make_memory(clock)
    assert mem.record("cup", "kitchen", "ahead", 0.9) is True
    assert mem.record("cup", "kitchen", "ahead", 0.9) is False
    clock.advance(31)
    assert mem.record("cup", "kitchen", "ahead", 0.9) is True
    assert mem.record("cup", "kitchen", "to your left", 0.9) is True  # moved => written immediately
    assert mem.count() == 3


def test_history_search_delete_clear(clock):
    mem = make_memory(clock)
    mem.record("cup", "kitchen", "ahead", 0.9)
    clock.advance(40)
    mem.record("bottle", "kitchen", "to your right", 0.8)
    assert [r.label for r in mem.history()] == ["bottle", "cup"]
    assert len(mem.search("kitchen")) == 2
    assert mem.delete_label(["cup"]) == 1
    assert mem.count() == 1
    assert mem.clear_all() == 1 and mem.count() == 0


def test_retention_purge(clock):
    mem = make_memory(clock)
    mem.record("cup", "kitchen", "ahead", 0.9)
    clock.advance(31 * 86400)
    assert mem.purge_expired() == 1 and mem.count() == 0


def test_manager_never_stores_people_or_vehicles():
    mm = MemoryManager(":memory:")
    mm.observe([make_track("person", hits=10), make_track("car", hits=10, track_id=2)])
    assert mm.objects.count() == 0


def test_manager_needs_stable_tracks_and_records_nearby_landmark():
    mm = MemoryManager(":memory:")
    sofa = make_track("couch", horizontal="slightly left", track_id=1)
    phone = make_track("cell phone", horizontal="slightly left", track_id=2, hits=10)
    flicker = make_track("cup", track_id=3, hits=2)  # too short-lived
    mm.observe([sofa, phone, flicker])
    rec = mm.objects.last_seen(["cell phone"])
    assert rec is not None and rec.near_object == "sofa"
    assert mm.objects.last_seen(["cup"]) is None


def test_disabled_memory_stores_nothing():
    mm = MemoryManager(":memory:", enabled=False)
    assert mm.observe([make_track("cell phone")]) == 0
    assert mm.objects.count() == 0


def test_forget_uses_aliases():
    mm = MemoryManager(":memory:")
    mm.objects.record("backpack", "hall", "ahead", 0.9)
    assert mm.forget("bag") == 1
