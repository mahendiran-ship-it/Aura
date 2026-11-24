"""Scene summaries: groups and prioritises objects instead of listing them all."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from aura.types import TrackedObject
from aura.utils.helpers import join_phrases, pluralize, spoken_name
from aura.vision.spatial_estimator import describe_direction, describe_proximity

# Objects that are not walking-level obstacles (so they never trigger "path obstructed").
NOT_PATH_OBSTACLES = {"traffic light", "tv", "clock", "laptop", "keyboard", "mouse", "remote",
                      "cell phone", "book", "cup", "bottle", "bowl", "fork", "knife", "spoon",
                      "wine glass", "vase", "scissors", "toothbrush", "stop sign", "bird"}

EMPTY_SCENE_MESSAGE = ("I don't recognize any objects in view right now. "
                       "That does not mean the area is clear.")


def region_of(horizontal: str) -> str:
    """Collapse seven horizontal names into left / ahead / right."""
    if horizontal in ("far left", "left"):
        return "on the left"
    if horizontal in ("far right", "right"):
        return "on the right"
    return "ahead"


class SceneAnalyzer:
    """Builds human-friendly summaries from tracked objects."""

    def __init__(self, min_confidence: float = 0.4, max_items: int = 6) -> None:
        self.min_confidence = min_confidence
        self.max_items = max_items

    def _usable(self, tracks: List[TrackedObject]) -> List[TrackedObject]:
        return [t for t in tracks if t.confidence >= self.min_confidence and t.spatial is not None]

    def summarize(self, tracks: List[TrackedObject]) -> str:
        """One or two sentences about the scene, most prominent objects first."""
        usable = self._usable(tracks)
        if not usable:
            return EMPTY_SCENE_MESSAGE

        groups: Dict[Tuple[str, str], List[TrackedObject]] = defaultdict(list)
        for t in usable:
            groups[(t.label, region_of(t.spatial.horizontal))].append(t)  # type: ignore[union-attr]

        def prominence(item: Tuple[Tuple[str, str], List[TrackedObject]]) -> float:
            return max(o.spatial.proximity_score for o in item[1])  # type: ignore[union-attr]

        ordered = sorted(groups.items(), key=prominence, reverse=True)
        shown, hidden = ordered[: self.max_items], ordered[self.max_items:]
        phrases = [f"{pluralize(label, len(objs))} {region}" for (label, region), objs in shown]
        text = "I can see " + join_phrases(phrases) + "."
        if hidden:
            extra = sum(len(o) for _, o in hidden)
            text += f" Plus {extra} other object{'s' if extra != 1 else ''}."
        obstruction = self.path_obstruction(tracks)
        if obstruction:
            text += " " + obstruction
        return text

    def path_obstruction(self, tracks: List[TrackedObject]) -> Optional[str]:
        """Describe objects that seem to be in the walking path, or None."""
        blockers = [t for t in self._usable(tracks)
                    if t.spatial.in_walking_zone  # type: ignore[union-attr]
                    and t.spatial.proximity in ("near", "medium")  # type: ignore[union-attr]
                    and t.label not in NOT_PATH_OBSTACLES]
        if not blockers:
            return None
        blockers.sort(key=lambda t: t.spatial.proximity_score, reverse=True)  # type: ignore[union-attr]
        unique: list = []
        for t in blockers:
            name = spoken_name(t.label)
            if name not in unique:
                unique.append(name)
        names = join_phrases(unique[:2])
        word = "partially obstructed" if len(blockers) == 1 and blockers[0].spatial.horizontal != "ahead" else "obstructed"  # type: ignore[union-attr]
        return f"The path ahead appears {word} by {names}."

    def where_is(self, labels: List[str], display: str, tracks: List[TrackedObject]) -> Optional[str]:
        """Describe visible instances of ``labels`` or return None if not visible."""
        found = [t for t in self._usable(tracks) if t.label in labels]
        if not found:
            return None
        found.sort(key=lambda t: t.spatial.proximity_score, reverse=True)  # type: ignore[union-attr]
        best = found[0]
        sp = best.spatial
        assert sp is not None
        text = f"I see {pluralize(best.label, 1)} {describe_direction(sp.horizontal)}, {describe_proximity(sp)}."
        if len(found) > 1:
            text += f" I can see {len(found)} in total."
        return text

    def what_is_in_front(self, tracks: List[TrackedObject]) -> str:
        """Only the objects roughly ahead of the user."""
        ahead = [t for t in self._usable(tracks)
                 if t.spatial.horizontal in ("slightly left", "ahead", "slightly right")]  # type: ignore[union-attr]
        if not ahead:
            return ("I don't recognize any objects directly ahead. "
                    "That does not mean the way is clear.")
        ahead.sort(key=lambda t: t.spatial.proximity_score, reverse=True)  # type: ignore[union-attr]
        parts = [f"{pluralize(t.label, 1)} that {describe_proximity(t.spatial)}" for t in ahead[:3]]  # type: ignore[arg-type]
        return "Ahead of you I see " + join_phrases(parts) + "."
