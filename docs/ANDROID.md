# Android

**There is no Android app yet.** This page says what you can do on a phone today, what is partial, and how a native app would plug in.

## What works / does not

| Approach | Status | Notes |
|---|---|---|
| **Phone as camera + speaker, AURA on a laptop / Raspberry Pi** | 🟡 plausible, not field-tested | Run an IP-camera app on the phone, set `camera.source: "http://PHONE_IP:8080/video"`. OpenCV's stream reading is the standard path but was only tested with fakes here. Adds latency; needs Wi-Fi. |
| **AURA inside Termux on the phone** | 🟡 partial, unverified | Python, OpenCV (`pkg install opencv-python`) and Tesseract are available as Termux packages. **`onnxruntime` is not provided by Termux's package manager** and pip wheels for Android/ARM may not exist; you may need to build it or use another runtime. Until the detector runs, AURA reports object detection as unavailable (memory/help/etc. still work). |
| Speech output on Android | 🟡 implemented | `tts_backend: termux` uses `termux-tts-speak` (Android's own TTS) — needs the Termux:API app. Untested on a device here. |
| GPS | 🟡 implemented | `navigation.location_provider: termux` uses `termux-location`. Untested on a device here. |
| Phone camera inside Termux | ❌ not provided | Termux cannot stream the live camera to OpenCV directly. Use the IP-camera approach or a native app. |
| Offline microphone recognition on Android | ❌ | Vosk + sounddevice are not reliably available in Termux. Use Android's speech recogniser from a native app. |
| Native accessible app (TalkBack-friendly) | 🗺️ planned | See below. |

`scripts/setup.sh` detects Termux, prints the `pkg install` lines, and installs what pip can. Expect to adapt.

## Termux quick path

```bash
pkg install python opencv-python python-numpy tesseract termux-api git
git clone https://github.com/mahendiran-ship-it/Aura.git && cd Aura
bash scripts/setup.sh         # onnxruntime may fail here: see note above
python -m aura check
```

## Designing the native app (planned)

The Python core already exposes a small local API (`api.enabled: true`, set `AURA_API_TOKEN`):

```
POST /command {"text": "find my phone"}   ->  {"response": "I see a phone slightly to your right."}
GET  /status                               ->  camera/detector/AI/room/paused ...
```

Planned split, keeping safety logic in the tested core:

```mermaid
flowchart LR
  subgraph Phone app (Kotlin, TalkBack-friendly)
    CamX[CameraX frames] --> Bridge
    Mic[Android SpeechRecognizer] --> Bridge
    TTS[Android TextToSpeech]
  end
  Bridge -->|"frames over localhost / on-device runtime"| Core[AURA core]
  Core -->|"text to speak + priority"| TTS
```

Two realistic options: (a) embed the core via a Python-on-Android runtime and implement `CameraSource` as a bridge that receives CameraX frames; (b) port the pipeline to Kotlin with ONNX Runtime Mobile / TFLite and reuse `config/safety.yaml` and the test cases as a spec. Either way: Android `TextToSpeech` for output, `SpeechRecognizer` (on-device where available) for input, and all UI exposed to TalkBack with large touch targets and no visual-only state. Not started.

## Safety on phones

Phones are held or worn at unpredictable angles and may overheat or throttle under continuous inference (use `LOW_POWER`). Do not rely on a pocketed phone for hazard warnings.
