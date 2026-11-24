"""Object detection with a YOLOv8-style ONNX model on CPU (onnxruntime).

Why ONNX Runtime: it is small, runs on CPU, and works on Raspberry Pi and
many Android (Termux) setups without installing PyTorch.

The model file is NOT bundled (size/licence). Run
``python scripts/download_models.py`` or see models/README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from aura.types import BBox, Detection
from aura.utils.logger import get_logger
from aura.vision.image_processor import letterbox
from aura.vision.labels import COCO_LABELS

log = get_logger("detector")


class ModelNotFoundError(Exception):
    """The ONNX model file does not exist. The message explains how to get it."""


class DetectorError(Exception):
    """The model exists but could not be loaded or run."""


def decode_yolov8(output: np.ndarray, ratio: float, pad: Tuple[int, int],
                  orig_size: Tuple[int, int], conf_threshold: float, nms_threshold: float,
                  labels: Sequence[str] = COCO_LABELS) -> List[Detection]:
    """Convert raw YOLOv8 output into :class:`Detection` objects.

    ``output`` has shape (1, 4+nc, N) (Ultralytics default) or (1, N, 4+nc).
    Boxes are (cx, cy, w, h) in letterboxed-input pixels; they are mapped back
    to the original frame using ``ratio`` and ``pad``. ``orig_size`` = (w, h).
    """
    pred = np.squeeze(output, axis=0)
    nc = len(labels)
    if pred.shape[0] == 4 + nc:  # (4+nc, N) -> (N, 4+nc)
        pred = pred.T
    if pred.ndim != 2 or pred.shape[1] != 4 + nc:
        raise DetectorError(
            f"Unexpected model output shape {output.shape}; expected a YOLOv8 model "
            f"with {nc} classes (output (1,{4 + nc},N))."
        )
    scores_all = pred[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    confidences = scores_all[np.arange(len(class_ids)), class_ids]
    keep = confidences >= conf_threshold
    if not keep.any():
        return []
    boxes, confidences, class_ids = pred[keep, :4], confidences[keep], class_ids[keep]

    orig_w, orig_h = orig_size
    pad_x, pad_y = pad
    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / ratio
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / ratio
    w = boxes[:, 2] / ratio
    h = boxes[:, 3] / ratio
    x1, y1 = np.clip(x1, 0, orig_w), np.clip(y1, 0, orig_h)
    x2, y2 = np.clip(x1 + w, 0, orig_w), np.clip(y1 + h, 0, orig_h)

    nms_boxes = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    idx = cv2.dnn.NMSBoxes(nms_boxes, confidences.astype(float).tolist(),
                           float(conf_threshold), float(nms_threshold))
    idx = np.array(idx).flatten() if len(idx) else []
    detections: List[Detection] = []
    for i in idx:
        detections.append(Detection(label=labels[int(class_ids[i])], confidence=float(confidences[i]),
                                    bbox=BBox(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))))
    return detections


class ObjectDetector:
    """Loads the ONNX model lazily and runs detection on BGR frames."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.45,
                 nms_threshold: float = 0.45, inference_size: int = 416,
                 labels: Sequence[str] = COCO_LABELS, allowed_labels: Optional[Sequence[str]] = None) -> None:
        self.model_path = Path(model_path)
        self.conf = confidence_threshold
        self.nms = nms_threshold
        self.size = inference_size
        self.labels = list(labels)
        self.allowed = set(allowed_labels) if allowed_labels else None
        self._session = None
        self._input_name = ""

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        """Load the model. Raises ModelNotFoundError / DetectorError with a helpful message."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Object detection model not found at '{self.model_path}'.\n"
                "Download it with:  python scripts/download_models.py\n"
                "or place a YOLOv8 ONNX file there (see models/README.md)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorError("onnxruntime is not installed. Run: pip install onnxruntime") from exc
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  # leave CPU for other tasks on small devices
            self._session = ort.InferenceSession(str(self.model_path), sess_options=opts,
                                                 providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"Could not load model '{self.model_path}': {exc}") from exc
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape  # [1,3,H,W]; H/W may be ints or symbolic strings
        if isinstance(shape[2], int) and shape[2] > 0:
            self.size = int(shape[2])  # a fixed-size model dictates the input size
        log.info("Detector loaded: %s (input %dpx, CPU)", self.model_path.name, self.size)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one BGR frame."""
        if self._session is None:
            raise DetectorError("Detector is not loaded. Call load() first.")
        h, w = frame.shape[:2]
        boxed, ratio, pad = letterbox(frame, self.size)
        blob = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        try:
            output = self._session.run(None, {self._input_name: blob})[0]
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"Inference failed: {exc}") from exc
        dets = decode_yolov8(output, ratio, pad, (w, h), self.conf, self.nms, self.labels)
        if self.allowed is not None:
            dets = [d for d in dets if d.label in self.allowed]
        return dets
