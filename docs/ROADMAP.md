# Roadmap

Status: **Fully implemented** (tested), **Partial** (code exists, unverified on real hardware/services), **Experimental**, **Planned**.

## Fully implemented (v0.1)
Detection (YOLOv8n ONNX), tracking, relative position heuristics, scene summaries, object finder, deterministic safety engine with cooldowns, SQLite memory with freshness, context engine, rule-based commands, OCR with confidence gating, config + performance modes, camera/AI/OCR failure handling, privacy controls, local HTTP API, CLI.

## Partial (needs verification on real devices)
Real webcam / IP-camera streams · espeak/pyttsx3/Termux speech · Vosk microphone · OSM geocoding/routing (live services untested) · Termux GPS and battery · serial distance sensors.

## Experimental
Landmark lookup (Overpass) · Termux-based on-phone operation.

## Planned

**Next (highest value, safety-driven)**
1. **Field validation** with O&M instructors and blind/low-vision users: tune `safety.yaml`, wording, alert frequency; record findings in `docs/`.
2. **Better detector** covering doors, stairs, curbs, poles, steps, wet-floor signs, handles, keys/wallet; keep the existing deterministic safety layer and add class weights.
3. **Depth**: plug a ToF/stereo sensor or a small monocular depth model into `DepthProvider`; make distances non-"approximate" only when measured.
4. **Android app** (TalkBack-first) using the local API or an embedded core — see [ANDROID.md](ANDROID.md).

**Later**
- Haptic output driver (`HapticOutput`) and direction-coded vibration.
- Wake word + on-device speech recognition.
- Visual localisation / SLAM, room mapping, automatic room recognition, indoor route graphs (`VisualLocalizer`).
- Camera-motion compensation with an IMU.
- Emergency contact calling/SMS with explicit user confirmation.
- Face/person recognition **only** with explicit opt-in consent and local storage.
- Multilingual commands and speech; custom vocabulary per user.
- Packaging (pip/APK), CI, telemetry-free crash reports.

## Explicitly out of scope
Autonomous road-crossing decisions, "it is safe" statements, replacing mobility aids, cloud video upload.
