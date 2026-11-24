"""Small pure helper functions (no hardware, no I/O) - easy to unit test."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict

Clock = Callable[[], float]

_NUMBER_WORDS: Dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                                 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
_IRREGULAR_PLURALS = {"person": "people", "bus": "buses", "couch": "couches",
                      "knife": "knives", "bench": "benches", "potted plant": "potted plants",
                      "dining table": "dining tables", "cell phone": "phones"}

# Friendly names for COCO labels that sound odd when spoken.
SPOKEN_NAMES = {"cell phone": "phone", "dining table": "table", "couch": "sofa",
                "tv": "television", "potted plant": "plant", "wine glass": "glass",
                "traffic light": "traffic light"}


def now() -> float:
    """Current wall-clock time as epoch seconds (patched in tests via Clock)."""
    return time.time()


def spoken_name(label: str) -> str:
    """Return a speech-friendly name for a detector label."""
    return SPOKEN_NAMES.get(label, label)


def pluralize(label: str, count: int) -> str:
    """'chair', 2 -> 'two chairs'; 'person', 1 -> 'a person'."""
    name = spoken_name(label)
    if count == 1:
        article = "an" if name[:1].lower() in "aeiou" else "a"
        return f"{article} {name}"
    number = _NUMBER_WORDS.get(count, str(count))
    plural = _IRREGULAR_PLURALS.get(label)
    if plural is None:
        plural = name + ("es" if name.endswith(("s", "x", "ch", "sh")) else "s")
    return f"{number} {plural}"


def join_phrases(items: list) -> str:
    """['a', 'b', 'c'] -> 'a, b, and c'."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def humanize_age(seconds: float) -> str:
    """Seconds -> '5 minutes ago' style text."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "less than a minute ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"{days} days ago"


def clock_time(epoch: float) -> str:
    """Epoch -> '6:42 PM' in local time."""
    t = time.localtime(epoch)
    hour = t.tm_hour % 12 or 12
    return f"{hour}:{t.tm_min:02d} {'AM' if t.tm_hour < 12 else 'PM'}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (0=N, 90=E) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def compass_name(bearing: float) -> str:
    """Bearing -> 'north', 'north-east', ..."""
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return names[int((bearing + 22.5) % 360 // 45)]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
