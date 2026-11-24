"""Shared test helpers. No camera, microphone or network is needed for these tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.config import DEFAULT_SAFETY, DEFAULT_SETTINGS, deep_merge  # noqa: E402
from aura.types import BBox, Detection, SpatialInfo, TrackedObject  # noqa: E402


class FakeClock:
    """Manually advanced clock so cooldown/staleness tests are instant and exact."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def safety_cfg() -> dict:
    return deep_merge(DEFAULT_SAFETY, {})


@pytest.fixture
def settings() -> dict:
    return deep_merge(DEFAULT_SETTINGS, {})


def make_track(label: str = "chair", horizontal: str = "ahead", proximity: str = "near",
               score: float = 0.8, hits: int = 10, conf: float = 0.9, growth: float = 0.0,
               in_zone: bool = True, track_id: int = 1, misses: int = 0,
               bbox: Optional[BBox] = None, uncertain: bool = True) -> TrackedObject:
    bbox = bbox or BBox(280, 300, 360, 460)
    return TrackedObject(
        track_id=track_id, label=label, confidence=conf, bbox=bbox,
        spatial=SpatialInfo(horizontal, "lower", proximity, score, in_zone, uncertain),
        hits=hits, misses=misses, growth=growth)


def det(label: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(label, conf, BBox(x1, y1, x2, y2))
