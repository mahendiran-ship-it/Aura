import numpy as np
import pytest

from aura.types import BBox
from aura.vision.detector import DetectorError, ModelNotFoundError, ObjectDetector, decode_yolov8
from aura.vision.image_processor import FrameSkipper, check_quality, letterbox, limit_size
from aura.vision.labels import COCO_LABELS, resolve_target
from aura.vision.object_search import ObjectSearch
from aura.vision.scene_analyzer import EMPTY_SCENE_MESSAGE, SceneAnalyzer
from aura.vision.spatial_estimator import DepthProvider, SpatialEstimator
from aura.vision.tracker import ObjectTracker
from tests.conftest import det, make_track


# ------------------------------------------------------------- tracker
def test_tracker_keeps_identity_and_confirms_after_min_hits(clock):
    tr = ObjectTracker(min_hits=3, clock=clock)
    ids = set()
    new_total = []
    for i in range(5):
        up = tr.update([det("cup", 100 + i * 2, 100, 160 + i * 2, 160)])
        new_total += up.new_ids
        ids |= {t.track_id for t in tr.tracks}
    assert len(ids) == 1                      # one identity throughout
    assert len(new_total) == 1                # announced as "new" exactly once
    assert tr.is_confirmed(next(iter(ids)))


def test_tracker_does_not_confirm_flickers(clock):
    tr = ObjectTracker(min_hits=3, clock=clock)
    up = tr.update([det("cup", 10, 10, 50, 50)])
    assert up.visible == [] and up.new_ids == []


def test_tracker_does_not_swap_labels(clock):
    tr = ObjectTracker(min_hits=1, clock=clock)
    tr.update([det("cup", 10, 10, 50, 50)])
    tr.update([det("bottle", 10, 10, 50, 50)])
    assert {t.label for t in tr.tracks} == {"cup", "bottle"}


def test_tracker_loss_and_reappearance(clock):
    tr = ObjectTracker(min_hits=2, max_misses=3, clock=clock)
    box = lambda: det("chair", 100, 100, 200, 200)  # noqa: E731
    for _ in range(3):
        tr.update([box()])
    tid = tr.tracks[0].track_id
    for _ in range(2):
        up = tr.update([])
        assert up.visible == []               # not visible while missing
    up = tr.update([box()])
    assert up.reappeared_ids == [tid]         # same identity returns
    lost = []
    for _ in range(5):
        lost += tr.update([]).lost_ids
    assert lost == [tid] and tr.tracks == []


def test_tracker_estimates_motion_direction(clock):
    tr = ObjectTracker(min_hits=1, clock=clock)
    for i in range(8):                        # moving right, constant size
        tr.update([det("person", 100 + i * 12, 100, 160 + i * 12, 220)])
    assert tr.tracks[0].movement == "moving right"
    tr.reset()
    for i in range(8):                        # growing => approaching
        s = i * 8
        tr.update([det("car", 200 - s, 150 - s, 300 + s, 250 + s)])
    assert tr.tracks[0].movement == "approaching"


def test_tracker_handles_two_objects_of_same_class(clock):
    tr = ObjectTracker(min_hits=1, clock=clock)
    for _ in range(3):
        up = tr.update([det("chair", 10, 10, 60, 80), det("chair", 300, 10, 360, 80)])
    assert len({t.track_id for t in up.visible}) == 2


# ------------------------------------------------------------- spatial
def test_horizontal_names():
    names = [SpatialEstimator.horizontal_name(x) for x in (0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 0.95)]
    assert names == ["far left", "left", "slightly left", "ahead", "slightly right", "right", "far right"]


def test_larger_object_is_nearer():
    est = SpatialEstimator(640, 480)
    big = est.estimate(det("chair", 200, 100, 440, 470))
    small = est.estimate(det("chair", 300, 200, 330, 240))
    assert big.proximity_score > small.proximity_score
    assert big.proximity == "near" and small.proximity == "far"


def test_walking_zone_and_uncertainty_flag():
    est = SpatialEstimator(640, 480)
    assert est.estimate(det("chair", 290, 200, 350, 300)).in_walking_zone
    assert not est.estimate(det("chair", 10, 200, 60, 300)).in_walking_zone
    assert est.estimate(det("chair", 290, 200, 350, 300)).uncertain is True


def test_real_depth_provider_replaces_heuristic():
    class Fixed(DepthProvider):
        def distance_m(self, frame, bbox):
            return 1.0

    est = SpatialEstimator(640, 480, depth_provider=Fixed())
    info = est.estimate(det("cup", 300, 200, 320, 220), frame=np.zeros((480, 640, 3), np.uint8))
    assert info.uncertain is False and info.proximity == "near"


# ------------------------------------------------------------- scene
def test_scene_summary_groups_and_orders():
    a = SceneAnalyzer()
    tracks = [make_track("chair", "left", "medium", 0.4, track_id=1, in_zone=False),
              make_track("chair", "left", "medium", 0.35, track_id=2, in_zone=False),
              make_track("dining table", "ahead", "near", 0.8, track_id=3, in_zone=False)]
    text = a.summarize(tracks)
    assert text.startswith("I can see a table ahead and two chairs on the left.")
    assert text.count("chair") == 1  # grouped, not listed twice


def test_obstruction_message_does_not_repeat_names():
    chairs = [make_track("chair", "ahead", "near", 0.8, track_id=1),
              make_track("chair", "ahead", "near", 0.7, track_id=2)]
    msg = SceneAnalyzer().path_obstruction(chairs)
    assert msg == "The path ahead appears obstructed by chair."


def test_scene_empty_never_implies_clear():
    text = SceneAnalyzer().summarize([])
    assert text == EMPTY_SCENE_MESSAGE and "does not mean" in text


def test_scene_reports_path_obstruction():
    bike = make_track("bicycle", "ahead", "near", 0.8)
    assert "obstructed" in SceneAnalyzer().summarize([bike])


def test_small_items_do_not_obstruct_path():
    cup = make_track("cup", "ahead", "near", 0.8)
    assert "obstructed" not in SceneAnalyzer().summarize([cup])


def test_where_is_and_in_front():
    a = SceneAnalyzer()
    t = [make_track("bottle", "slightly right", "medium", 0.4)]
    assert a.where_is(["bottle"], "bottle", t) == (
        "I see a bottle slightly to your right, appears at a medium distance.")
    assert a.where_is(["cup"], "cup", t) is None
    assert "bottle" in a.what_is_in_front(t)


# ------------------------------------------------------------- object search
def test_resolve_target_aliases_and_unsupported():
    assert resolve_target("phone") == (["cell phone"], None)
    assert resolve_target("bag")[0][0] == "backpack"
    labels, problem = resolve_target("door")
    assert labels == [] and "cannot recognize door" in problem
    assert resolve_target("keys")[1] is not None
    assert resolve_target("zzz")[1] is not None


def test_search_flow(clock):
    s = ObjectSearch(clock=clock)
    assert s.start("bottle") is None and s.active
    # nothing visible yet: silent
    assert s.update([]) is None
    t = [make_track("bottle", "slightly right", "medium", 0.4)]
    first = s.update(t)
    assert first == "I see a bottle slightly to your right."
    assert s.update(t) is None                         # same position: stays quiet
    clock.advance(2)
    t = [make_track("bottle", "ahead", "medium", 0.4)]
    assert "ahead" in s.update(t)                      # position changed
    clock.advance(1)
    assert s.update([]) is None                        # brief miss ignored
    clock.advance(4)
    assert "lost sight" in s.update([])
    assert s.update([]) is None                        # does not repeat 'lost'
    s.cancel()
    assert not s.active and s.update(t) is None


def test_search_never_claims_arrival(clock):
    s = ObjectSearch(clock=clock)
    s.start("cup")
    msg = s.update([make_track("cup", "ahead", "near", 0.9)])
    assert "appears close" in msg and "reached" not in msg.lower() and "you have" not in msg.lower()


def test_search_times_out_honestly(clock):
    s = ObjectSearch(timeout_s=10, clock=clock)
    s.start("cup")
    clock.advance(11)
    msg = s.update([])
    assert "could not find" in msg and not s.active


def test_search_rejects_unsupported_target():
    assert "cannot recognize" in ObjectSearch().start("keys")


def test_guide_command(clock):
    s = ObjectSearch(clock=clock)
    assert "not searching" in s.guide([])
    s.start("bottle")
    assert "can't see" in s.guide([])
    assert "ahead and" not in s.guide([make_track("bottle", "right", "medium", 0.4)])
    assert "to your right" in s.guide([make_track("bottle", "right", "medium", 0.4)])


# ------------------------------------------------------------- image processing
def test_quality_checks():
    dark = np.full((240, 320, 3), 5, np.uint8)
    assert "too dark" in check_quality(dark).issues
    bright = np.full((240, 320, 3), 252, np.uint8)
    assert "too bright" in check_quality(bright).issues
    rng = np.random.default_rng(0)
    sharp = rng.integers(60, 200, (240, 320, 3), dtype=np.uint8)
    assert check_quality(sharp).ok
    assert not check_quality(None).ok  # type: ignore[arg-type]


def test_letterbox_and_limit_size():
    img = np.zeros((480, 640, 3), np.uint8)
    out, ratio, pad = letterbox(img, 416)
    assert out.shape == (416, 416, 3) and ratio == pytest.approx(0.65) and pad[0] == 0 and pad[1] > 0
    assert limit_size(img, 320).shape[1] == 320
    assert limit_size(img, 2000).shape == img.shape


def test_frame_skipper():
    sk = FrameSkipper(3)
    assert [sk.should_process() for _ in range(6)] == [True, False, False, True, False, False]
    assert all(FrameSkipper(1).should_process() for _ in range(3))


# ------------------------------------------------------------- detector
def test_decode_yolov8_synthetic_output_maps_boxes_back():
    nc = len(COCO_LABELS)
    out = np.zeros((1, 4 + nc, 3), np.float32)
    # one confident 'cup' (class 41) centred at (208,208) in a 416 letterbox of a 640x480 frame
    out[0, :4, 0] = [208, 208, 100, 100]
    out[0, 4 + COCO_LABELS.index("cup"), 0] = 0.9
    out[0, :4, 1] = [10, 10, 5, 5]          # below threshold
    out[0, 4 + 0, 1] = 0.1
    dets = decode_yolov8(out, ratio=0.65, pad=(0, 73), orig_size=(640, 480), conf_threshold=0.45, nms_threshold=0.45)
    assert len(dets) == 1 and dets[0].label == "cup"
    b = dets[0].bbox
    assert b.cx == pytest.approx(208 / 0.65, rel=1e-3) and b.cy == pytest.approx((208 - 73) / 0.65, rel=1e-3)


def test_decode_rejects_wrong_model_shape():
    with pytest.raises(DetectorError):
        decode_yolov8(np.zeros((1, 10, 5), np.float32), 1, (0, 0), (100, 100), 0.4, 0.4)


def test_nms_removes_duplicates():
    nc = len(COCO_LABELS)
    out = np.zeros((1, 4 + nc, 2), np.float32)
    for i, conf in enumerate((0.9, 0.8)):
        out[0, :4, i] = [200 + i, 200, 100, 100]
        out[0, 4 + 39, i] = conf
    dets = decode_yolov8(out, 1.0, (0, 0), (416, 416), 0.45, 0.45)
    assert len(dets) == 1 and dets[0].confidence == pytest.approx(0.9)


def test_missing_model_gives_helpful_error(tmp_path):
    d = ObjectDetector(str(tmp_path / "nope.onnx"))
    with pytest.raises(ModelNotFoundError) as e:
        d.load()
    assert "download_models.py" in str(e.value)
    assert d.is_ready is False


def test_detect_before_load_raises():
    with pytest.raises(DetectorError):
        ObjectDetector("x.onnx").detect(np.zeros((10, 10, 3), np.uint8))


def test_corrupt_model_file_is_reported(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(DetectorError):
        ObjectDetector(str(bad)).load()


@pytest.fixture(scope="module")
def sample_photo():
    import cv2
    try:
        import ultralytics
        import os
        path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    except ImportError:
        pytest.skip("ultralytics not installed; no sample photo available")
    img = cv2.imread(path)
    if img is None:
        pytest.skip("sample photo missing")
    return img


def test_real_model_detects_objects_in_real_photo(sample_photo):
    from aura.config import resolve_path
    model = resolve_path("models/yolov8n.onnx")
    if not model.exists():
        pytest.skip("model not downloaded (python scripts/download_models.py)")
    d = ObjectDetector(str(model))
    d.load()
    labels = [x.label for x in d.detect(sample_photo)]
    assert "bus" in labels and labels.count("person") >= 2
