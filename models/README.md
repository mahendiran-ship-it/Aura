# Models

AURA needs one object-detection model. It is **not committed** to Git (size and licence).

## Quick install

```bash
python scripts/download_models.py
```

This downloads the official **YOLOv8n** weights (COCO, 80 classes) and, if you have `ultralytics` installed, exports them to `models/yolov8n.onnx`. If you do not want to install PyTorch, export it once on any machine and copy the `.onnx` file here:

```bash
pip install ultralytics onnx
yolo export model=yolov8n.pt format=onnx imgsz=416 opset=12
cp yolov8n.onnx /path/to/aura/models/
```

## Requirements for a custom model

Any YOLOv8-style ONNX detector with output `(1, 4+classes, N)`. If it has 80 classes it must use the COCO order (`aura/vision/labels.py`). For other class lists, pass your own `labels` to `ObjectDetector`.

## Licence

YOLOv8 weights are provided by Ultralytics under **AGPL-3.0** (commercial licence available). If you redistribute AURA with the model, or run it as a network service, review that licence. For a permissive alternative, train or export a model with a licence that suits you.

## What the default model cannot recognise

COCO has **no** classes for doors, stairs, keys, wallets, glasses, signs or curbs. AURA says so when asked instead of guessing. See `docs/ROADMAP.md`.

## Optional: Vosk speech model

Download a small English model from <https://alphacephei.com/vosk/models>, unpack it to `models/vosk-model`, and set `audio.stt_backend: vosk`.
