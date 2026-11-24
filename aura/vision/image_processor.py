"""Frame preprocessing: resize, quality checks, frame skipping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class QualityReport:
    """Result of :func:`check_quality`. ``ok`` False means detections are unreliable."""

    ok: bool
    blur_variance: float
    brightness: float
    issues: List[str] = field(default_factory=list)


def check_quality(frame: np.ndarray, min_blur_variance: float = 40.0,
                  min_brightness: float = 35.0, max_brightness: float = 235.0) -> QualityReport:
    """Detect a dark, over-exposed, blurry or covered camera view.

    Blur is measured by the variance of the Laplacian (low = blurry).
    """
    if frame is None or frame.size == 0:
        return QualityReport(False, 0.0, 0.0, ["no image"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (160, 120)) if gray.shape[1] > 160 else gray
    brightness = float(small.mean())
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    issues: List[str] = []
    if brightness < min_brightness:
        issues.append("too dark")
    if brightness > max_brightness:
        issues.append("too bright")
    if blur < min_blur_variance:
        issues.append("blurry or covered")
    return QualityReport(not issues, blur, brightness, issues)


def letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size x size``.

    Returns ``(image, ratio, (pad_x, pad_y))`` so boxes can be mapped back.
    """
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, ratio, (pad_x, pad_y)


def limit_size(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side`` (never upscales)."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class FrameSkipper:
    """Say 'process' once every ``n`` frames (n=1 processes every frame)."""

    def __init__(self, every_n: int = 3) -> None:
        self.every_n = max(1, int(every_n))
        self._count = 0

    def should_process(self) -> bool:
        process = self._count % self.every_n == 0
        self._count += 1
        return process
