"""Text-to-speech backends. All are optional; the console backend always works.

Backends:
  console   prints the text (also what a screen reader reads). Always available.
  espeak    espeak-ng / espeak binary (Linux, Raspberry Pi).
  pyttsx3   desktop TTS engines (Windows SAPI, macOS, Linux espeak).
  termux    ``termux-tts-speak`` from Termux:API (uses Android's own TTS engine).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from aura.utils.logger import get_logger

log = get_logger("tts")


class TTSBackend(ABC):
    name = "base"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak ``text``; blocks until done or :meth:`stop` is called."""

    def stop(self) -> None:  # pragma: no cover - default no-op
        """Interrupt current speech."""


class ConsoleTTS(TTSBackend):
    name = "console"

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream

    def available(self) -> bool:
        return True

    def speak(self, text: str) -> None:
        print(f"AURA: {text}", file=self.stream, flush=True)


class _ProcessTTS(TTSBackend):
    """Shared logic for backends that run an external command."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _command(self, text: str) -> List[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def speak(self, text: str) -> None:
        try:
            with self._lock:
                self._proc = subprocess.Popen(self._command(text), stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
                proc = self._proc
            proc.wait()
        except Exception as exc:  # noqa: BLE001
            log.warning("%s speech failed: %s", self.name, exc)

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()


class EspeakTTS(_ProcessTTS):
    name = "espeak"

    def __init__(self, rate: int = 170, volume: float = 1.0, voice: Optional[str] = None) -> None:
        super().__init__()
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        self.rate, self.volume, self.voice = rate, volume, voice

    def available(self) -> bool:
        return self.binary is not None

    def _command(self, text: str) -> List[str]:
        cmd = [self.binary or "espeak", "-s", str(self.rate), "-a", str(int(max(0, min(1, self.volume)) * 200))]
        if self.voice:
            cmd += ["-v", self.voice]
        return cmd + ["--", text]


class TermuxTTS(_ProcessTTS):
    name = "termux"

    def __init__(self, rate: int = 170, volume: float = 1.0, voice: Optional[str] = None) -> None:
        super().__init__()
        self.binary = shutil.which("termux-tts-speak")
        self.rate = rate
        self.voice = voice

    def available(self) -> bool:
        return self.binary is not None

    def _command(self, text: str) -> List[str]:
        # Termux rate: 1.0 = normal. Map words-per-minute (~170 normal) to that scale.
        cmd = [self.binary or "termux-tts-speak", "-r", f"{self.rate / 170.0:.2f}"]
        if self.voice:
            cmd += ["-l", self.voice]
        return cmd + [text]


class Pyttsx3TTS(TTSBackend):
    name = "pyttsx3"

    def __init__(self, rate: int = 170, volume: float = 1.0, voice: Optional[str] = None) -> None:
        self.rate, self.volume, self.voice = rate, volume, voice
        self._engine: Any = None

    def available(self) -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def _get(self) -> Any:
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            if self.voice:
                self._engine.setProperty("voice", self.voice)
        return self._engine

    def speak(self, text: str) -> None:
        try:
            engine = self._get()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            log.warning("pyttsx3 speech failed: %s", exc)

    def stop(self) -> None:
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:  # noqa: BLE001
            pass


def create_tts(cfg: Dict[str, Any]) -> TTSBackend:
    """Pick a TTS backend. ``auto`` tries termux, espeak, pyttsx3, then console."""
    choice = str(cfg.get("tts_backend", "auto")).lower()
    rate, vol, voice = int(cfg.get("rate", 170)), float(cfg.get("volume", 1.0)), cfg.get("voice")
    makers = {"termux": lambda: TermuxTTS(rate, vol, voice), "espeak": lambda: EspeakTTS(rate, vol, voice),
              "pyttsx3": lambda: Pyttsx3TTS(rate, vol, voice), "console": ConsoleTTS}
    order = ["termux", "espeak", "pyttsx3"] if choice == "auto" else [choice]
    for name in order:
        maker = makers.get(name)
        if maker is None:
            log.warning("Unknown tts_backend %r; using console output.", name)
            break
        backend = maker()
        if backend.available():
            return backend
        if choice != "auto":
            log.warning("TTS backend %r is not available; falling back to console text.", name)
    return ConsoleTTS()
