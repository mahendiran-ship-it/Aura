"""A tiny thread-safe publish/subscribe bus.

Modules talk through events instead of importing each other, so new
components (a vibration motor, an Android bridge) can be added by simply
subscribing to a topic.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, DefaultDict, List

from aura.utils.logger import get_logger

log = get_logger("events")

Handler = Callable[[str, Any], None]

# Topic names used by the built-in modules.
TOPIC_ALERT = "alert"
TOPIC_SPEAK = "speak"
TOPIC_FRAME_PROCESSED = "frame.processed"
TOPIC_CAMERA_STATUS = "camera.status"
TOPIC_COMMAND = "command"


class EventManager:
    """Synchronous pub/sub. Handler errors are logged, never propagated."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register ``handler(topic, payload)`` for ``topic`` ('*' = everything)."""
        with self._lock:
            self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            if handler in self._handlers.get(topic, []):
                self._handlers[topic].remove(handler)

    def publish(self, topic: str, payload: Any = None) -> int:
        """Deliver to subscribers. Returns how many handlers were called."""
        with self._lock:
            targets = list(self._handlers.get(topic, [])) + list(self._handlers.get("*", []))
        for handler in targets:
            try:
                handler(topic, payload)
            except Exception:  # noqa: BLE001 - a bad subscriber must not break the pipeline
                log.exception("Event handler failed for topic %s", topic)
        return len(targets)
