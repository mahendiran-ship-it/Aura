"""Command-line interface (screen-reader friendly: plain lines, no colour or cursor tricks)."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import time
from typing import List, Optional

from aura import __version__
from aura.audio.command_parser import parse_command
from aura.config import ConfigError, load_safety, load_settings, resolve_path
from aura.utils.logger import setup_logging

DISCLAIMER = ("AURA is an assistive aid. It does NOT replace a white cane, a guide dog, or professional "
              "mobility training, and it can miss hazards. See docs/SAFETY.md.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aura", description="AURA - assistive intelligence for the visually impaired")
    p.add_argument("--version", action="version", version=f"AURA {__version__}")
    p.add_argument("--config", help="path to settings.yaml")
    p.add_argument("--safety-config", help="path to safety.yaml")
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start the assistant (default)")
    run.add_argument("--camera", help="camera source: index, file path or http stream URL")
    run.add_argument("--mode", choices=["LOW_POWER", "BALANCED", "HIGH_PERFORMANCE"], help="performance mode")
    run.add_argument("--no-memory", action="store_true", help="disable persistent memory for this run")
    run.add_argument("--no-ai", action="store_true", help="disable the optional AI assistant")
    run.add_argument("--voice-input", action="store_true", help="use microphone (vosk) instead of keyboard")

    sub.add_parser("check", help="check dependencies, model, camera and configuration")

    an = sub.add_parser("analyze", help="analyze one image file with the real pipeline")
    an.add_argument("image")
    an.add_argument("--read", action="store_true", help="also run OCR on the image")

    pr = sub.add_parser("parse", help="show how a sentence is parsed as a command")
    pr.add_argument("text", nargs="+")
    return p


def battery_status() -> Optional[str]:
    """Battery percentage if the platform exposes it, else None."""
    try:
        import json
        import subprocess
        if shutil.which("termux-battery-status"):
            out = subprocess.run(["termux-battery-status"], capture_output=True, text=True, timeout=5).stdout
            return f"{json.loads(out)['percentage']}%"
        base = "/sys/class/power_supply"
        if os.path.isdir(base):
            for name in os.listdir(base):
                cap = os.path.join(base, name, "capacity")
                if os.path.exists(cap):
                    return f"{open(cap).read().strip()}%"
    except Exception:  # noqa: BLE001
        pass
    return None


def cmd_check() -> int:
    """Print a plain-text environment report. Exit code 1 if a core component is missing."""
    ok = True
    print(f"AURA {__version__} environment check\n")
    for mod, why, core in [("numpy", "core", True), ("cv2", "camera + image processing (opencv-python-headless)", True),
                           ("yaml", "configuration (pyyaml)", True), ("onnxruntime", "object detection", True),
                           ("requests", "navigation / optional AI", False), ("pytesseract", "text reading", False),
                           ("pyttsx3", "desktop speech (optional)", False), ("vosk", "offline microphone (optional)", False),
                           ("serial", "ESP32/Arduino sensors (optional)", False)]:
        try:
            importlib.import_module(mod)
            print(f"  [ok]      {mod}: {why}")
        except ImportError:
            print(f"  [{'MISSING' if core else 'absent '}] {mod}: {why}")
            ok = ok and not core
    for binary, why in [("tesseract", "OCR engine"), ("espeak-ng", "speech"), ("termux-tts-speak", "Android speech")]:
        print(f"  [{'ok' if shutil.which(binary) else 'absent'}] binary {binary}: {why}")
    try:
        settings = load_settings(None)
        load_safety(None)
        print("  [ok]      configuration loads")
        model = resolve_path(settings["vision"]["model_path"])
        if model.exists():
            print(f"  [ok]      detection model: {model}")
        else:
            print(f"  [MISSING] detection model: {model}\n            -> run: python scripts/download_models.py")
            ok = False
    except ConfigError as exc:
        print(f"  [MISSING] configuration: {exc}")
        ok = False
    print("\n" + ("Core components ready." if ok else "Some core components are missing (see above)."))
    return 0 if ok else 1


def cmd_analyze(args: argparse.Namespace, settings: dict, safety: dict) -> int:
    """Run detection, tracking, hazards and a scene summary on one real image."""
    import cv2
    from aura.core.orchestrator import Orchestrator
    from aura.audio.audio_manager import AudioManager
    from aura.audio.speech_output import ConsoleTTS
    from aura.memory.memory_manager import MemoryManager
    from aura.vision.camera import ImageFileCamera

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 2
    settings["tracker"]["min_hits"] = 1  # a single still image: confirm immediately
    orch = Orchestrator(settings, safety, camera=ImageFileCamera(args.image),
                        audio=AudioManager(ConsoleTTS(), threaded=False), memory=MemoryManager(":memory:", False))
    try:
        orch.detector.load()
    except Exception as exc:  # noqa: BLE001
        print(exc)
        return 1
    orch.state.set(detector_ready=True, camera_ok=True)
    orch.tracker.min_hits = 1
    orch.safety_cfg["min_persistence_frames"] = 1
    dets = orch.process_frame(frame)
    orch.process_frame(frame)  # second pass lets the tracker confirm and the alert engine speak
    print(f"\nDetections ({len(dets)}):")
    for d in dets:
        print(f"  {d.label:15s} {d.confidence:.2f}")
    print("\nScene: " + orch.scene.summarize(orch.get_tracks()))
    if args.read:
        print("Text:  " + orch.ocr.speak_text(orch.ocr.read(frame)))
    return 0


def cmd_run(args: argparse.Namespace, settings: dict, safety: dict) -> int:
    from aura.audio.speech_input import create_speech_input
    from aura.core.orchestrator import Orchestrator
    from aura.interfaces.api import ApiConfigError, start_api

    if getattr(args, "camera", None):
        settings["camera"]["source"] = int(args.camera) if str(args.camera).isdigit() else args.camera
    if getattr(args, "no_memory", False):
        settings["memory"]["enabled"] = False
    if getattr(args, "no_ai", False):
        settings["ai"]["enabled"] = False
    if getattr(args, "voice_input", False):
        settings["audio"]["stt_backend"] = "vosk"

    print(f"AURA {__version__}\n{DISCLAIMER}\n")
    orch = Orchestrator(settings, safety)
    orch.start()
    try:
        start_api(orch, settings["api"])
    except ApiConfigError as exc:
        print(f"[AURA] API not started: {exc}")
    bat = battery_status()
    if bat:
        print(f"[AURA] Battery: {bat}")
    print("Type a command (for example 'describe', 'find my phone', 'help'). Say 'quit' to exit.\n")
    try:
        orch.run_interactive(create_speech_input(settings["audio"]))
    finally:
        orch.stop()
    return 0


def _with_default_command(argv: List[str]) -> List[str]:
    """``aura --camera 1`` means ``aura run --camera 1``. Global flags may precede a subcommand."""
    commands = {"run", "check", "analyze", "parse"}
    value_flags = {"--config", "--safety-config"}
    i = 0
    while i < len(argv):
        if argv[i] in commands or argv[i] in ("-h", "--help", "--version"):
            return argv
        i += 2 if argv[i] in value_flags else 1
    return argv + ["run"]


def main(argv: Optional[List[str]] = None) -> int:
    argv = _with_default_command(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return cmd_check()
    if args.command == "parse":
        cmd = parse_command(" ".join(args.text))
        print(f"intent={cmd.intent.value} target={cmd.target!r} extra={cmd.extra!r}")
        return 0
    try:
        settings = load_settings(args.config, mode_override=getattr(args, "mode", None))
        safety = load_safety(args.safety_config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    log_cfg = settings["logging"]
    setup_logging(log_cfg["level"], log_cfg["file"])
    if args.command == "analyze":
        return cmd_analyze(args, settings, safety)
    return cmd_run(args, settings, safety)


if __name__ == "__main__":
    raise SystemExit(main())
