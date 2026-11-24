"""Object finder: "Find my phone" -> short spoken guidance until found/cancelled.

Guidance is rate-limited: it speaks when the object is first seen, when its
position bucket changes, or when it is lost - not every frame. AURA never
claims you have *reached* the object; at most it says the object appears
close and to reach out carefully.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from aura.types import TrackedObject
from aura.utils.helpers import now, spoken_name
from aura.vision.labels import resolve_target
from aura.vision.spatial_estimator import describe_direction


class ObjectSearch:
    """State machine for one active search."""

    def __init__(self, timeout_s: float = 60.0, repeat_after_s: float = 8.0,
                 lost_after_s: float = 3.0, clock: Callable[[], float] = now) -> None:
        self.timeout_s = timeout_s
        self.repeat_after_s = repeat_after_s
        self.lost_after_s = lost_after_s
        self._clock = clock
        self.target: Optional[str] = None
        self.labels: List[str] = []
        self._started = 0.0
        self._last_bucket: Optional[str] = None
        self._last_spoken = 0.0
        self._last_seen = 0.0
        self._seen_ever = False
        self._announced_lost = False

    @property
    def active(self) -> bool:
        return self.target is not None

    def start(self, target: str) -> Optional[str]:
        """Begin searching. Returns an explanation string if the target is unsupported."""
        labels, problem = resolve_target(target)
        if problem:
            return problem
        self.target, self.labels = target, labels
        self._started = self._clock()
        self._last_bucket = None
        self._last_spoken = 0.0
        self._last_seen = 0.0
        self._seen_ever = False
        self._announced_lost = False
        return None

    def cancel(self) -> None:
        self.target, self.labels = None, []

    def update(self, tracks: List[TrackedObject]) -> Optional[str]:
        """Feed the latest tracks. Returns a sentence to speak, or None to stay quiet."""
        if not self.active:
            return None
        t = self._clock()
        found = [o for o in tracks if o.label in self.labels and o.spatial is not None and o.visible]
        name = spoken_name(self.labels[0]) if len(self.labels) == 1 else self.target

        if found:
            best = max(found, key=lambda o: o.spatial.proximity_score)  # type: ignore[union-attr]
            sp = best.spatial
            assert sp is not None
            bucket = f"{sp.horizontal}|{sp.proximity}"
            self._last_seen = t
            self._announced_lost = False
            first = not self._seen_ever
            self._seen_ever = True
            changed = bucket != self._last_bucket
            due = (t - self._last_spoken) >= self.repeat_after_s
            if first or (changed and (t - self._last_spoken) >= 1.5) or due:
                self._last_bucket, self._last_spoken = bucket, t
                msg = f"I see a {name} {describe_direction(sp.horizontal)}."
                if sp.proximity == "near":
                    msg += " It appears close. Reach out slowly and carefully."
                return msg
            return None

        if self._seen_ever and not self._announced_lost and (t - self._last_seen) >= self.lost_after_s:
            self._announced_lost = True
            return f"I lost sight of the {name}. Try turning slowly."
        if (t - self._started) >= self.timeout_s:
            self.cancel()
            return (f"I could not find a {name} after searching for a minute. "
                    "It may be out of view, or my model may have missed it.")
        return None

    def guide(self, tracks: List[TrackedObject]) -> str:
        """Answer 'guide me toward it' with the current direction (on demand)."""
        if not self.active:
            return "I'm not searching for anything. Say 'find' and an object name."
        found = [o for o in tracks if o.label in self.labels and o.spatial is not None and o.visible]
        name = spoken_name(self.labels[0]) if len(self.labels) == 1 else self.target
        if not found:
            return f"I can't see the {name} right now. Turn slowly and I'll keep looking."
        sp = max(found, key=lambda o: o.spatial.proximity_score).spatial  # type: ignore[union-attr]
        assert sp is not None
        near = " It appears close, so reach out carefully." if sp.proximity == "near" else ""
        return f"The {name} appears to be {describe_direction(sp.horizontal)}.{near}"
