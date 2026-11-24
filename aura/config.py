"""Configuration loading.

Settings come from three layers (later wins):
1. Built-in defaults (below) - AURA always starts, even with no files.
2. ``config/settings.yaml`` / ``config/safety.yaml`` (or a path you pass).
3. The selected performance mode, which overrides a few speed knobs.

Secrets (API keys) are NEVER read from YAML. Put them in environment
variables; the YAML only stores the *name* of the variable.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SETTINGS: Dict[str, Any] = {
    "performance_mode": "BALANCED",
    "camera": {
        # int = device index, path = video/image file, "http://..." = phone IP-camera stream
        "source": 0,
        "width": 640,
        "height": 480,
        "max_consecutive_failures": 30,
        "reconnect_delay_s": 2.0,
    },
    "vision": {
        "model_path": "models/yolov8n.onnx",
        "confidence_threshold": 0.45,
        "nms_threshold": 0.45,
        "inference_size": 416,
        "process_every_n_frames": 3,
        "min_blur_variance": 40.0,
        "min_brightness": 35.0,
        "max_brightness": 235.0,
    },
    "tracker": {"iou_threshold": 0.3, "max_misses": 10, "min_hits": 3},
    "memory": {
        "database_path": "data/aura_memory.sqlite3",
        "enabled": True,
        "recent_seconds": 600,  # < 10 min since seen => "recently observed"
        "stale_seconds": 86400,  # > 24 h => stale warning
        "retention_days": 30,  # older observations are purged
        "write_interval_s": 30,  # do not rewrite same object more often than this
        "default_room": "unknown room",
    },
    "audio": {
        "tts_backend": "auto",  # auto | pyttsx3 | espeak | termux | console
        "rate": 170,
        "volume": 1.0,
        "voice": None,
        "stt_backend": "keyboard",  # keyboard | vosk
        "vosk_model_path": "models/vosk-model",
        "wake_word": "aura",
    },
    "ai": {
        "enabled": False,
        "endpoint": "http://127.0.0.1:11434/v1",  # e.g. local Ollama / llama.cpp server
        "model": "llama3.2:1b",
        "api_key_env": "AURA_LLM_API_KEY",
        "timeout_s": 15,
        "allow_cloud": False,
    },
    "ocr": {"backend": "tesseract", "min_confidence": 60.0, "language": "eng"},
    "navigation": {
        "location_provider": "none",  # none | fixed | termux
        "fixed_lat": None,
        "fixed_lon": None,
        "geocoder_url": "https://nominatim.openstreetmap.org",
        "router_url": "https://router.project-osrm.org",
        "user_agent": "AURA-assistive/0.1 (open-source)",
        "timeout_s": 10,
    },
    "api": {"enabled": False, "host": "127.0.0.1", "port": 8765, "token_env": "AURA_API_TOKEN"},
    "user": {"emergency_contact_name": "", "emergency_contact_phone": ""},
    "logging": {"level": "INFO", "file": None, "log_user_text": False},
}

DEFAULT_SAFETY: Dict[str, Any] = {
    # Base danger weight per class (0..1). Unknown classes get "default".
    "class_weights": {
        "car": 1.0, "bus": 1.0, "truck": 1.0, "motorcycle": 1.0, "bicycle": 0.8,
        "train": 1.0, "person": 0.5, "dog": 0.5, "chair": 0.5, "bench": 0.5,
        "dining table": 0.5, "couch": 0.5, "potted plant": 0.4, "suitcase": 0.5,
        "backpack": 0.3, "bottle": 0.2, "cup": 0.1, "default": 0.4,
    },
    "vehicle_classes": ["car", "bus", "truck", "motorcycle", "bicycle", "train"],
    # score thresholds (score is 0..~2) -> level
    "level_thresholds": {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.15},
    "min_confidence": 0.5,
    "min_persistence_frames": 3,
    "walking_zone": {"x_min": 0.30, "x_max": 0.70},
    "low_obstacle_max_top": 0.55,  # box top below this fraction of frame height => "low" obstacle
    # alert policy
    "min_spoken_level": "LOW",
    "cooldown_seconds": {"INFO": 120, "LOW": 20, "MEDIUM": 10, "HIGH": 5, "CRITICAL": 2},
    "global_min_gap_seconds": 1.5,
    "sensor_distance_cm": {"LOW": 150, "MEDIUM": 100, "HIGH": 60, "CRITICAL": 30},
    "degraded_view_warning": True,
}

PERFORMANCE_MODES: Dict[str, Dict[str, Any]] = {
    "LOW_POWER": {"vision": {"inference_size": 320, "process_every_n_frames": 6},
                  "camera": {"width": 480, "height": 360}},
    "BALANCED": {"vision": {"inference_size": 416, "process_every_n_frames": 3},
                 "camera": {"width": 640, "height": 480}},
    "HIGH_PERFORMANCE": {"vision": {"inference_size": 640, "process_every_n_frames": 1},
                         "camera": {"width": 1280, "height": 720}},
}


class ConfigError(Exception):
    """Raised when a configuration file is present but unusable."""


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: ``override`` merged recursively onto ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_settings(path: Optional[str] = None, mode_override: Optional[str] = None) -> Dict[str, Any]:
    """Load settings: defaults < YAML file < performance mode.

    A missing *default-location* file is fine (defaults are used). A missing
    file that you explicitly passed raises ConfigError. ``mode_override``
    (e.g. from ``--mode``) replaces the file's performance_mode.
    """
    explicit = path is not None
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if cfg_path.exists():
        settings = deep_merge(settings, _read_yaml(cfg_path))
    elif explicit:
        raise ConfigError(f"Settings file not found: {cfg_path}")

    mode = str(mode_override or settings.get("performance_mode", "BALANCED")).upper()
    if mode not in PERFORMANCE_MODES:
        raise ConfigError(
            f"performance_mode must be one of {sorted(PERFORMANCE_MODES)}, got {mode!r}"
        )
    settings["performance_mode"] = mode
    # YAML values the user wrote explicitly win over the mode presets only if
    # they differ from the defaults; otherwise the mode preset applies.
    preset = PERFORMANCE_MODES[mode]
    for section, values in preset.items():
        for key, value in values.items():
            if settings[section].get(key) == DEFAULT_SETTINGS[section].get(key):
                settings[section][key] = value
    return settings


def load_safety(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the safety configuration (defaults < YAML file)."""
    explicit = path is not None
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "safety.yaml"
    safety = copy.deepcopy(DEFAULT_SAFETY)
    if cfg_path.exists():
        safety = deep_merge(safety, _read_yaml(cfg_path))
    elif explicit:
        raise ConfigError(f"Safety file not found: {cfg_path}")
    return safety


def resolve_path(p: str) -> Path:
    """Resolve a config path relative to the project root unless absolute."""
    path = Path(os.path.expanduser(p))
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_secret(env_name: str) -> Optional[str]:
    """Read a secret from the environment. Returns None if unset/empty."""
    value = os.environ.get(env_name, "").strip()
    return value or None
