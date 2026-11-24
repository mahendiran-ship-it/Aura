"""Speech output queue with priorities, interruption and 'repeat'.

* CRITICAL/HIGH alerts interrupt whatever is being said.
* Normal messages queue up; stale low-priority chatter is dropped when the
  queue backs up, so the user never hears outdated information.
* The most recent message is kept for the "repeat" command.
"""

from __future__ import annotations

import queue
import threading
from typing import List, Optional, Tuple

from aura.audio.speech_output import TTSBackend
from aura.types import AlertLevel
from aura.utils.logger import get_logger

log = get_logger("audio")

MAX_QUEUE = 4


class AudioManager:
    """Owns the TTS backend and a worker thread."""

    def __init__(self, backend: TTSBackend, threaded: bool = True) -> None:
        self.backend = backend
        self.threaded = threaded
        self._q: "queue.PriorityQueue[Tuple[int, int, str]]" = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._last_message = ""
        self._muted = False
        self.spoken: List[str] = []  # recent log (for tests/status), capped
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if threaded:
            self._worker = threading.Thread(target=self._run, name="aura-tts", daemon=True)
            self._worker.start()

    @property
    def last_message(self) -> str:
        return self._last_message

    def speak(self, text: str, priority: AlertLevel = AlertLevel.INFO, remember: bool = True) -> None:
        """Queue ``text``. Higher ``priority`` plays first and may interrupt."""
        text = (text or "").strip()
        if not text:
            return
        if remember:
            self._last_message = text
        if self._muted and priority < AlertLevel.HIGH:
            return
        self._muted = False if priority >= AlertLevel.HIGH else self._muted
        if priority >= AlertLevel.HIGH:
            self._drain()
            self.backend.stop()
        with self._lock:
            self._seq += 1
            if self._q.qsize() >= MAX_QUEUE and priority < AlertLevel.MEDIUM:
                return  # too busy: drop low-value message rather than let it go stale
            self._q.put((-int(priority), self._seq, text))
        if not self.threaded:
            self._drain_and_speak()

    def repeat(self) -> None:
        if self._last_message:
            self.speak(self._last_message, remember=False)
        else:
            self.speak("There is nothing to repeat.", remember=False)

    def stop_talking(self) -> None:
        """Silence current speech and clear the queue."""
        self._drain()
        self.backend.stop()

    def _drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _drain_and_speak(self) -> None:
        while True:
            try:
                _, _, text = self._q.get_nowait()
            except queue.Empty:
                return
            self._say(text)

    def _say(self, text: str) -> None:
        self.spoken.append(text)
        del self.spoken[:-20]
        try:
            self.backend.speak(text)
        except Exception:  # noqa: BLE001
            log.exception("TTS backend raised")

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                _, _, text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._say(text)

    def close(self) -> None:
        self._stop_evt.set()
        self.backend.stop()
        if self._worker:
            self._worker.join(timeout=1.0)
