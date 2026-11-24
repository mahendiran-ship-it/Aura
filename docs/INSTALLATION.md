# Installation

Requires Python 3.9+ (developed on 3.13). Everything runs on CPU; no GPU or PyTorch is needed at runtime.

## Linux / Raspberry Pi OS

```bash
sudo apt install python3-venv tesseract-ocr espeak-ng     # tesseract = reading, espeak-ng = speech (both optional)
git clone https://github.com/mahendiran-ship-it/Aura.git && cd Aura
bash scripts/setup.sh
. .venv/bin/activate
python scripts/download_models.py
python -m aura check
scripts/run.sh
```

`scripts/setup.sh` creates `.venv`, installs `requirements.txt` only, and prints what is missing. It does not install optional packages.

### The model

`scripts/download_models.py` downloads the official YOLOv8n weights, verifies a SHA-256 checksum, and converts them to ONNX. The conversion needs `pip install ultralytics onnx` (large, includes PyTorch) — only on the machine doing the conversion. Alternatively export on another computer and copy `yolov8n.onnx` into `models/` (see `models/README.md`). Note the AGPL-3.0 licence of the weights.

### Optional packages

```bash
pip install -r requirements-optional.txt     # pyttsx3, vosk, sounddevice, pyserial — pick what you need
```

- **Speech output:** `espeak-ng` (Linux) or `pyttsx3`; otherwise AURA prints text (screen readers can read it).
- **Offline microphone:** `pip install vosk sounddevice`, unpack a [Vosk model](https://alphacephei.com/vosk/models) to `models/vosk-model`, set `audio.stt_backend: vosk` or use `--voice-input`. Not verified in the development environment.
- **Distance sensors:** `pip install pyserial`, see [HARDWARE.md](HARDWARE.md).

## Camera setup

`camera.source` in `config/settings.yaml`, or `--camera`:

| Source | Value |
|---|---|
| USB/built-in webcam | `0`, `1`, … |
| Video file (testing) | `"/path/to/video.mp4"` |
| Phone as camera | `"http://PHONE_IP:8080/video"` using an IP-camera app on the same Wi-Fi. Keep it on a private network; stream URLs may contain credentials (AURA hides them in status text). |
| Single photo | `python -m aura analyze photo.jpg` |

Linux webcam permission problems: add your user to the `video` group. If the camera cannot be opened AURA says so and continues with non-vision features.

## Configuration

Edit `config/settings.yaml` and `config/safety.yaml` (every key optional). Use `--config` / `--safety-config` for other files. API keys: copy `.env.example` to `.env` (git-ignored); `scripts/run.sh` loads it.

## Optional AI assistant

Off by default. To use a **local** model, run an OpenAI-compatible server (for example Ollama: `ollama serve`, `ollama pull llama3.2:1b`) and set `ai.enabled: true`. A non-local endpoint is refused unless `ai.allow_cloud: true`; AURA then announces that cloud AI is active. Only structured text facts are sent.

## Verify

```bash
python -m aura check
python -m pytest tests -q          # after: pip install -r requirements-dev.txt
python -m aura analyze some_photo.jpg --read
```

## Troubleshooting

| Message | Fix |
|---|---|
| "Object detection model not found" | `python scripts/download_models.py` |
| "Could not open OpenCV camera" | Check device index, other apps using it, permissions |
| "Text reading is unavailable" / "failed" | `sudo apt install tesseract-ocr` |
| Silence instead of voice | No TTS engine found → install `espeak-ng`; text is still printed |
| "I could not reach the map service" | Navigation needs internet |
