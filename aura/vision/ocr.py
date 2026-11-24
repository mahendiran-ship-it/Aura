"""Reading text from a camera frame (Tesseract by default).

AURA never says it read something accurately when confidence is low: the
spoken result is prefixed with a warning, or the text is withheld.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from aura.utils.logger import get_logger

log = get_logger("ocr")

MAX_SPOKEN_CHARS = 700


@dataclass
class OCRLine:
    text: str
    confidence: float  # 0..100
    height: float  # pixel height, a proxy for how prominent the text is


@dataclass
class OCRResult:
    lines: List[OCRLine] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        total = sum(len(l.text) for l in self.lines) or 1
        return sum(l.confidence * len(l.text) for l in self.lines) / total


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Grayscale, upscale small images, boost local contrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        scale = 1200 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.fastNlMeansDenoising(gray, None, h=10) if gray.size < 4_000_000 else gray


class OCRReader:
    """Wraps the OCR engine; all failures are returned in ``OCRResult.error``."""

    def __init__(self, backend: str = "tesseract", min_confidence: float = 60.0,
                 language: str = "eng", engine: Any = None) -> None:
        self.backend = backend
        self.min_confidence = min_confidence
        self.language = language
        self._engine = engine  # injectable for tests

    def available(self) -> bool:
        if self._engine is not None:
            return True
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:  # noqa: BLE001
            return False

    def read(self, frame: Optional[np.ndarray]) -> OCRResult:
        """Run OCR on a BGR frame."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return OCRResult(error="I don't have a camera image to read.")
        try:
            engine = self._engine
            if engine is None:
                import pytesseract
                engine = pytesseract
            data: Dict[str, List] = engine.image_to_data(
                preprocess(frame), lang=self.language, config="--psm 6",
                output_type=engine.Output.DICT)
        except ImportError:
            return OCRResult(error="Text reading is unavailable: pytesseract is not installed.")
        except Exception as exc:  # noqa: BLE001 - tesseract binary missing, bad language pack, etc.
            log.warning("OCR failed: %s", exc)
            return OCRResult(error="Text reading failed. Tesseract may not be installed correctly.")
        return OCRResult(lines=self._group(data))

    @staticmethod
    def _group(data: Dict[str, List]) -> List[OCRLine]:
        grouped: Dict[tuple, List[tuple]] = {}
        for i, word in enumerate(data.get("text", [])):
            word = (word or "").strip()
            try:
                conf = float(data["conf"][i])
            except (ValueError, KeyError, IndexError):
                continue
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append((word, conf, float(data["height"][i])))
        lines = []
        for words in grouped.values():
            text = " ".join(w for w, _, _ in words)
            if not re.search(r"[A-Za-z0-9]", text):
                continue  # ignore pure-symbol noise
            lines.append(OCRLine(text=text, confidence=sum(c for _, c, _ in words) / len(words),
                                 height=max(h for _, _, h in words)))
        return lines

    def important_lines(self, result: OCRResult, limit: int = 5) -> List[OCRLine]:
        """Pick prominent lines: large text, or text containing numbers/prices/times."""
        if not result.lines:
            return []
        heights = sorted(l.height for l in result.lines)
        median = heights[len(heights) // 2]
        scored = []
        for l in result.lines:
            score = l.height / (median or 1)
            if re.search(r"\d", l.text):
                score += 0.5
            if l.text.isupper() and len(l.text) > 3:
                score += 0.3
            scored.append((score, l))
        scored.sort(key=lambda s: s[0], reverse=True)
        top = {id(l) for _, l in scored[:limit]}
        return [l for l in result.lines if id(l) in top]  # keep reading order

    def speak_text(self, result: OCRResult, important_only: bool = False) -> str:
        """Turn an OCR result into a safe, honest spoken message."""
        if result.error:
            return result.error
        if not result.lines:
            return ("I could not find any readable text. Try holding the camera steady, "
                    "closer to the text, with good lighting.")
        lines = self.important_lines(result) if important_only else result.lines
        text = " ".join(l.text for l in lines)[:MAX_SPOKEN_CHARS]
        conf = result.mean_confidence
        if conf < self.min_confidence * 0.6:
            return ("I found some text but I'm not confident I can read it correctly, "
                    "so I won't guess. Try again with better lighting or a steadier view.")
        if conf < self.min_confidence:
            return f"I'm not fully sure I read this correctly. It may say: {text}"
        return text
