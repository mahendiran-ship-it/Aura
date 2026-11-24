import cv2
import numpy as np
import pytest

from aura.vision.camera import CameraError, ImageFileCamera, OpenCVCamera, create_camera
from aura.vision.ocr import OCRReader, OCRResult


class FakeCapture:
    def __init__(self, opened=True, frames=None, raise_on_read=False):
        self._opened, self.frames, self.released = opened, list(frames or []), False
        self.raise_on_read = raise_on_read

    def isOpened(self):
        return self._opened

    def set(self, *a):
        return True

    def read(self):
        if self.raise_on_read:
            raise RuntimeError("driver crashed")
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


FRAME = np.zeros((48, 64, 3), np.uint8) + 100


# ------------------------------------------------------------- camera failure handling
def test_camera_open_failure_is_clear_error():
    cam = OpenCVCamera(0, capture_factory=lambda s: FakeCapture(opened=False))
    with pytest.raises(CameraError) as e:
        cam.open()
    assert "settings.yaml" in str(e.value)


def test_camera_factory_exception_is_wrapped():
    def boom(_):
        raise OSError("no device")
    with pytest.raises(CameraError):
        OpenCVCamera(0, capture_factory=boom).open()


def test_read_before_open_raises():
    with pytest.raises(CameraError):
        OpenCVCamera(0).read()


def test_camera_returns_none_on_transient_failure_then_raises_after_limit():
    cam = OpenCVCamera(0, max_consecutive_failures=3, capture_factory=lambda s: FakeCapture(frames=[FRAME]))
    cam.open()
    assert cam.read() is not None
    assert cam.read() is None and cam.read() is None   # transient
    with pytest.raises(CameraError):
        cam.read()                                      # third consecutive failure


def test_successful_frame_resets_failure_counter():
    cap = FakeCapture(frames=[FRAME, FRAME])
    cam = OpenCVCamera(0, max_consecutive_failures=2, capture_factory=lambda s: cap)
    cam.open()
    cam.read()
    assert cam.read() is not None


def test_driver_exception_during_read_counts_as_failure():
    cam = OpenCVCamera(0, max_consecutive_failures=1, capture_factory=lambda s: FakeCapture(raise_on_read=True))
    cam.open()
    with pytest.raises(CameraError):
        cam.read()


def test_reconnect_and_close_idempotent():
    caps = [FakeCapture(frames=[FRAME]), FakeCapture(frames=[FRAME])]
    cam = OpenCVCamera(0, capture_factory=lambda s: caps.pop(0), sleep=lambda s: None)
    cam.open()
    assert cam.reconnect() is True and cam.read() is not None
    cam.close()
    cam.close()


def test_reconnect_failure_returns_false():
    seq = [FakeCapture(), FakeCapture(opened=False)]
    cam = OpenCVCamera(0, capture_factory=lambda s: seq.pop(0), sleep=lambda s: None)
    cam.open()
    assert cam.reconnect() is False


def test_description_hides_stream_credentials():
    cam = OpenCVCamera("http://user:secret@192.168.1.5:8080/video")
    assert "secret" not in cam.description


def test_create_camera_from_config():
    assert create_camera({"source": "2"}).source == 2
    assert create_camera({"source": "http://x/video"}).source == "http://x/video"


def test_image_file_camera(tmp_path):
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), FRAME)
    cam = ImageFileCamera(str(p))
    cam.open()
    assert cam.read().shape == FRAME.shape
    cam.close()
    with pytest.raises(CameraError):
        ImageFileCamera(str(tmp_path / "missing.png")).open()


# ------------------------------------------------------------- OCR
class FakeEngine:
    class Output:
        DICT = "dict"

    def __init__(self, words, confs, raises=None):
        self.words, self.confs, self.raises = words, confs, raises

    def image_to_data(self, img, lang=None, config=None, output_type=None):
        if self.raises:
            raise self.raises
        n = len(self.words)
        return {"text": self.words, "conf": [str(c) for c in self.confs], "block_num": [1] * n,
                "par_num": [1] * n, "line_num": list(range(1, n + 1)), "height": [20] * n}


def test_ocr_confident_text_is_read():
    r = OCRReader(engine=FakeEngine(["EXIT", "LEFT"], [95, 92]))
    out = r.read(FRAME)
    assert out.error is None and r.speak_text(out) == "EXIT LEFT"


def test_ocr_borderline_confidence_is_hedged():
    r = OCRReader(min_confidence=60, engine=FakeEngine(["MENU"], [50]))
    assert r.speak_text(r.read(FRAME)).startswith("I'm not fully sure")


def test_ocr_low_confidence_text_is_withheld():
    r = OCRReader(min_confidence=60, engine=FakeEngine(["xzq"], [10]))
    msg = r.speak_text(r.read(FRAME))
    assert "xzq" not in msg and "won't guess" in msg


def test_ocr_no_text():
    r = OCRReader(engine=FakeEngine([], []))
    assert "could not find any readable text" in r.speak_text(r.read(FRAME))


def test_ocr_engine_failure_is_reported_not_raised():
    r = OCRReader(engine=FakeEngine([], [], raises=RuntimeError("tesseract is not installed")))
    out = r.read(FRAME)
    assert out.error and "failed" in r.speak_text(out)


def test_ocr_no_image():
    r = OCRReader(engine=FakeEngine([], []))
    assert "don't have a camera image" in r.speak_text(r.read(None))


def test_ocr_missing_library(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pytesseract":
            raise ImportError
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    out = OCRReader().read(FRAME)
    assert "unavailable" in (out.error or "")


def test_important_only_prefers_large_text_and_numbers():
    from aura.vision.ocr import OCRLine
    lines = [OCRLine("small print here", 90, 10), OCRLine("SALE 50%", 90, 40), OCRLine("more small", 90, 10)]
    r = OCRReader()
    picked = r.important_lines(OCRResult(lines), limit=1)
    assert [l.text for l in picked] == ["SALE 50%"]


def test_real_tesseract_reads_rendered_text():
    r = OCRReader()
    if not r.available():
        pytest.skip("tesseract not installed")
    img = np.full((200, 700, 3), 255, np.uint8)
    cv2.putText(img, "EXIT 42", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
    out = r.read(img)
    assert out.error is None
    assert "EXIT" in out.text and "42" in out.text
    assert r.speak_text(out).startswith("EXIT")
