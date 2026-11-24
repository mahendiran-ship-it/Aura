"""Multi-object tracker (greedy IoU matching, same-label only).

Deliberately simple and dependency-free so a beginner can read it. It gives
each object a stable ``track_id`` across frames, estimates motion, and
reports when objects appear, disappear or come back, so AURA announces an
object once instead of every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from aura.types import Detection, SpatialInfo, TrackedObject
from aura.utils.helpers import now

SpatialFn = Callable[[Detection], SpatialInfo]


@dataclass
class TrackUpdate:
    """What changed in the latest frame."""

    visible: List[TrackedObject] = field(default_factory=list)  # confirmed + seen this frame
    new_ids: List[int] = field(default_factory=list)  # became confirmed this frame
    lost_ids: List[int] = field(default_factory=list)  # confirmed tracks that were dropped
    reappeared_ids: List[int] = field(default_factory=list)  # came back after a gap


class ObjectTracker:
    """Follow objects across processed frames."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 10, min_hits: int = 3,
                 clock: Callable[[], float] = now, history_len: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self._clock = clock
        self._history_len = history_len
        self._tracks: Dict[int, TrackedObject] = {}
        self._confirmed: Set[int] = set()
        self._next_id = 1

    @property
    def tracks(self) -> List[TrackedObject]:
        """All live tracks (including unconfirmed and recently missing)."""
        return list(self._tracks.values())

    def is_confirmed(self, track_id: int) -> bool:
        return track_id in self._confirmed

    def reset(self) -> None:
        self._tracks.clear()
        self._confirmed.clear()

    def update(self, detections: List[Detection], spatial_fn: Optional[SpatialFn] = None) -> TrackUpdate:
        """Match ``detections`` to existing tracks and return what changed."""
        t = self._clock()
        result = TrackUpdate()

        pairs = []
        for di, det in enumerate(detections):
            for tid, trk in self._tracks.items():
                if trk.label != det.label:
                    continue
                iou = trk.bbox.iou(det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)

        matched_det: Set[int] = set()
        matched_trk: Set[int] = set()
        for _iou, di, tid in pairs:
            if di in matched_det or tid in matched_trk:
                continue
            matched_det.add(di)
            matched_trk.add(tid)
            self._apply_match(self._tracks[tid], detections[di], t, spatial_fn, result)

        for di, det in enumerate(detections):
            if di not in matched_det:
                self._create(det, t, spatial_fn)

        for tid in list(self._tracks):
            if tid in matched_trk:
                continue
            trk = self._tracks[tid]
            trk.misses += 1
            if trk.misses > self.max_misses:
                if tid in self._confirmed:
                    result.lost_ids.append(tid)
                self._confirmed.discard(tid)
                del self._tracks[tid]

        for tid, trk in self._tracks.items():
            if trk.hits >= self.min_hits and tid not in self._confirmed:
                self._confirmed.add(tid)
                result.new_ids.append(tid)
        result.visible = [trk for tid, trk in self._tracks.items()
                          if tid in self._confirmed and trk.misses == 0]
        return result

    # -- internals ---------------------------------------------------------
    def _create(self, det: Detection, t: float, spatial_fn: Optional[SpatialFn]) -> None:
        trk = TrackedObject(track_id=self._next_id, label=det.label, confidence=det.confidence,
                            bbox=det.bbox, first_seen=t, last_seen=t,
                            spatial=spatial_fn(det) if spatial_fn else None)
        trk.history.append((det.bbox.cx, det.bbox.cy, det.bbox.area))
        self._tracks[trk.track_id] = trk
        self._next_id += 1

    def _apply_match(self, trk: TrackedObject, det: Detection, t: float,
                     spatial_fn: Optional[SpatialFn], result: TrackUpdate) -> None:
        was_missing = trk.misses
        prev = trk.bbox
        dx, dy = det.bbox.cx - prev.cx, det.bbox.cy - prev.cy
        growth = (det.bbox.area / prev.area - 1.0) if prev.area > 0 else 0.0
        # exponential smoothing (alpha 0.5) reduces jitter
        trk.velocity = (0.5 * dx + 0.5 * trk.velocity[0], 0.5 * dy + 0.5 * trk.velocity[1])
        trk.growth = 0.5 * growth + 0.5 * trk.growth
        trk.bbox = det.bbox
        trk.confidence = 0.6 * det.confidence + 0.4 * trk.confidence
        trk.hits += 1
        trk.misses = 0
        trk.last_seen = t
        trk.spatial = spatial_fn(det) if spatial_fn else trk.spatial
        trk.history.append((det.bbox.cx, det.bbox.cy, det.bbox.area))
        del trk.history[:-self._history_len]
        if was_missing >= 2 and trk.track_id in self._confirmed:
            result.reappeared_ids.append(trk.track_id)
