"""Camera abstraction.

Everything else in AURA only knows the :class:`CameraSource` interface, so
a phone stream, a wearable camera, a video file, or a future Android camera
bridge can be swapped in without touching the AI code.

No frames are ever written to disk by this module.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from aura.utils.logger import get_logger

log = get_logger("camera")


class CameraError(Exception):
    """Raised when a camera cannot be opened or has failed permanently."""


class CameraSource(ABC):
    """Interface every camera implementation must follow."""

    @abstractmethod
    def open(self) -> None:
        """Open the device/stream. Raises CameraError on failure."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if no frame is available right now."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Must be safe to call twice."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable name for status messages (no secrets)."""


class OpenCVCamera(CameraSource):
    """Webcam index, video file, or network stream (e.g. a phone running an
    IP-camera app, ``http://<phone-ip>:8080/video``) through OpenCV."""

    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480,
                 max_consecutive_failures: int = 30, reconnect_delay_s: float = 2.0,
                 capture_factory: Optional[Callable[[Any], Any]] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.max_failures = max_consecutive_failures
        self.reconnect_delay_s = reconnect_delay_s
        self._factory = capture_factory
        self._sleep = sleep
        self._cap: Any = None
        self._failures = 0

    @property
    def description(self) -> str:
        src = self.source
        if isinstance(src, str) and "://" in src:
            src = src.split("://", 1)[0] + "://<redacted>"  # URLs may embed credentials
        return f"OpenCV camera ({src})"

    def _new_capture(self) -> Any:
        if self._factory is not None:
            return self._factory(self.source)
        import cv2  # local import keeps module importable without OpenCV
        return cv2.VideoCapture(self.source)

    def open(self) -> None:
        try:
            cap = self._new_capture()
        except Exception as exc:  # noqa: BLE001
            raise CameraError(f"Could not create camera for {self.description}: {exc}") from exc
        if cap is None or not cap.isOpened():
            raise CameraError(
                f"Could not open {self.description}. Check that the camera is connected, "
                "not used by another app, and that 'camera.source' in config/settings.yaml is correct."
            )
        try:
            import cv2
            if isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
        except Exception:  # noqa: BLE001 - properties are best-effort
            pass
        self._cap = cap
        self._failures = 0
        log.info("Camera opened: %s", self.description)

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise CameraError("Camera is not open.")
        try:
            ok, frame = self._cap.read()
        except Exception as exc:  # noqa: BLE001
            ok, frame = False, None
            log.warning("Camera read raised: %s", exc)
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            self._failures = 0
            return frame
        self._failures += 1
        if self._failures >= self.max_failures:
            raise CameraError(
                f"{self.description} stopped delivering frames "
                f"({self._failures} failures in a row)."
            )
        return None

    def reconnect(self) -> bool:
        """Try once to reopen the camera. Returns True on success."""
        self.close()
        self._sleep(self.reconnect_delay_s)
        try:
            self.open()
            return True
        except CameraError as exc:
            log.warning("Reconnect failed: %s", exc)
            return False

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None


class ImageFileCamera(CameraSource):
    """Serves one real image file repeatedly. For testing the pipeline on a
    photo you took (``--image photo.jpg``). Not a simulation of a camera."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._frame: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return f"image file ({self.path})"

    def open(self) -> None:
        import cv2
        frame = cv2.imread(self.path)
        if frame is None:
            raise CameraError(f"Could not read image file: {self.path}")
        self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        if self._frame is None:
            raise CameraError("Image camera is not open.")
        return self._frame.copy()

    def close(self) -> None:
        self._frame = None


def create_camera(cfg: Dict[str, Any]) -> CameraSource:
    """Build a camera from the ``camera`` config section."""
    source = cfg.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return OpenCVCamera(
        source=source, width=int(cfg.get("width", 640)), height=int(cfg.get("height", 480)),
        max_consecutive_failures=int(cfg.get("max_consecutive_failures", 30)),
        reconnect_delay_s=float(cfg.get("reconnect_delay_s", 2.0)),
    )
