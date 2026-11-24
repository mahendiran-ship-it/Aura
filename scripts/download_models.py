#!/usr/bin/env python3
"""Download YOLOv8n weights and produce models/yolov8n.onnx.

Steps:
  1. If models/yolov8n.onnx already exists, do nothing.
  2. Download official yolov8n.pt from the Ultralytics release (needs internet).
  3. Export to ONNX. This needs `pip install ultralytics onnx` (large, includes PyTorch).
     If it is not installed, the script explains how to export elsewhere instead.

The weights are AGPL-3.0 licensed by Ultralytics (see models/README.md).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
TARGET = MODELS / "yolov8n.onnx"
IMGSZ = 416


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    if TARGET.exists():
        print(f"Model already present: {TARGET}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        pt = Path(tmp) / "yolov8n.pt"
        print(f"Downloading {URL} ...")
        try:
            urllib.request.urlretrieve(URL, pt)
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}\nDownload yolov8n.pt manually and export it (see models/README.md).")
            return 1
        digest = hashlib.sha256(pt.read_bytes()).hexdigest()
        if digest != SHA256:
            print(f"WARNING: checksum differs from the expected value.\n  expected {SHA256}\n  got      {digest}\n"
                  "The release may have been updated. Continuing is at your own risk; aborting.")
            return 1
        try:
            from ultralytics import YOLO
        except ImportError:
            keep = MODELS / "yolov8n.pt"
            shutil.copy(pt, keep)
            print("\nDownloaded weights to models/yolov8n.pt, but 'ultralytics' is needed to convert them to ONNX.\n"
                  "  pip install ultralytics onnx\n  python scripts/download_models.py\n"
                  "or export on another machine and copy yolov8n.onnx into models/.")
            return 1
        out = YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, opset=12, simplify=False)
        shutil.copy(out, TARGET)
    leftover = MODELS / "yolov8n.pt"
    if leftover.exists():
        leftover.unlink()
    print(f"Model ready: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
