"""Speech/command input backends.

keyboard  typed commands - always available, screen-reader friendly, and the
          default. This is real and working.
vosk      OFFLINE microphone recognition (needs ``pip install vosk sounddevice``
          and a downloaded Vosk model). Implemented but only usable if those are
          installed; otherwise AURA tells you and falls back to the keyboard.

Audio is processed in memory only and never saved.
"""

from __future__ import annotations

import json
import queue
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from aura.config import resolve_path
from aura.utils.logger import get_logger

log = get_logger("stt")


class SpeechInputError(Exception):
    """The chosen recognition backend cannot start. The message says why."""


class SpeechInput(ABC):
    name = "base"

    @abstractmethod
    def listen(self) -> Optional[str]:
        """Block until one utterance is available. Returns None at end of input."""

    def close(self) -> None:  # pragma: no cover
        pass


class KeyboardInput(SpeechInput):
    name = "keyboard"

    def __init__(self, prompt: str = "> ") -> None:
        self.prompt = prompt

    def listen(self) -> Optional[str]:
        try:
            return input(self.prompt)
        except EOFError:
            return None


class VoskInput(SpeechInput):
    name = "vosk"

    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise SpeechInputError(
                "Offline speech recognition needs: pip install vosk sounddevice "
                "(not available on every platform)."
            ) from exc
        path = resolve_path(model_path)
        if not path.exists():
            raise SpeechInputError(
                f"Vosk model not found at '{path}'. Download a small English model from "
                "https://alphacephei.com/vosk/models and unpack it there."
            )
        self._sd = sd
        self._rate = sample_rate
        self._recognizer = KaldiRecognizer(Model(str(path)), sample_rate)
        self._audio: "queue.Queue[bytes]" = queue.Queue()
        self._stream: Any = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self._audio.put(bytes(indata))

    def listen(self) -> Optional[str]:
        try:
            if self._stream is None:
                self._stream = self._sd.RawInputStream(samplerate=self._rate, blocksize=4000, dtype="int16",
                                                       channels=1, callback=self._callback)
                self._stream.start()
            while True:
                data = self._audio.get()
                if self._recognizer.AcceptWaveform(data):
                    text = json.loads(self._recognizer.Result()).get("text", "").strip()
                    if text:
                        return text
        except Exception as exc:  # noqa: BLE001
            log.error("Microphone input failed: %s", exc)
            return None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def create_speech_input(cfg: Dict[str, Any]) -> SpeechInput:
    """Create the configured input; falls back to keyboard with a clear message."""
    backend = str(cfg.get("stt_backend", "keyboard")).lower()
    if backend == "vosk":
        try:
            return VoskInput(cfg.get("vosk_model_path", "models/vosk-model"))
        except SpeechInputError as exc:
            print(f"[AURA] Voice input unavailable: {exc}\n[AURA] Using keyboard input instead.")
            return KeyboardInput()
    if backend != "keyboard":
        print(f"[AURA] Unknown stt_backend {backend!r}; using keyboard input.")
    return KeyboardInput()
