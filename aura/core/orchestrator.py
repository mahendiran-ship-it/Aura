"""The orchestrator wires every module into the running system.

Pipeline for each processed camera frame:

    camera -> quality check -> detector -> tracker(+spatial) -> hazard rules
           -> alert manager (cooldowns) -> audio          (safety, no LLM)
           -> object search guidance -> audio
           -> memory observe

User commands arrive as text (typed or from speech recognition), are parsed
by the rule-based parser, and answered by the Assistant.

All heavy parts can be injected, which is how the tests run without a camera.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from aura.audio.audio_manager import AudioManager
from aura.audio.command_parser import Command, Intent, parse_command
from aura.audio.speech_input import SpeechInput
from aura.config import load_safety, load_settings, resolve_path
from aura.core.event_manager import (TOPIC_ALERT, TOPIC_CAMERA_STATUS, TOPIC_COMMAND, TOPIC_FRAME_PROCESSED,
                                     EventManager)
from aura.core.state_manager import StateManager
from aura.intelligence.assistant import Assistant, Response
from aura.intelligence.context_engine import ContextEngine
from aura.intelligence.llm import LLMClient
from aura.memory.memory_manager import MemoryManager
from aura.navigation.gps import create_location_provider
from aura.navigation.landmark_engine import LandmarkEngine
from aura.navigation.map_engine import MapEngine
from aura.safety import safety_rules as rules
from aura.safety.alert_manager import AlertManager
from aura.safety.hazard_detector import HazardDetector
from aura.sensors.base import SensorHub
from aura.types import Alert, AlertLevel, Detection, TrackedObject
from aura.utils.logger import get_logger
from aura.vision.camera import CameraError, CameraSource, create_camera
from aura.vision.detector import DetectorError, ModelNotFoundError, ObjectDetector
from aura.vision.image_processor import FrameSkipper, check_quality
from aura.vision.object_search import ObjectSearch
from aura.vision.ocr import OCRReader
from aura.vision.scene_analyzer import SceneAnalyzer
from aura.vision.spatial_estimator import SpatialEstimator
from aura.vision.tracker import ObjectTracker

log = get_logger("orchestrator")

MAX_CAMERA_RECONNECTS = 5


class Orchestrator:
    def __init__(self, settings: Optional[Dict[str, Any]] = None, safety: Optional[Dict[str, Any]] = None,
                 camera: Optional[CameraSource] = None, detector: Optional[ObjectDetector] = None,
                 audio: Optional[AudioManager] = None, memory: Optional[MemoryManager] = None,
                 sensors: Optional[SensorHub] = None, ocr: Optional[OCRReader] = None,
                 llm: Optional[LLMClient] = None) -> None:
        self.settings = settings or load_settings()
        self.safety_cfg = safety or load_safety()
        s = self.settings
        self.events = EventManager()
        self.state = StateManager()

        vcfg = s["vision"]
        self.camera = camera if camera is not None else create_camera(s["camera"])
        self.detector = detector if detector is not None else ObjectDetector(
            str(resolve_path(vcfg["model_path"])), vcfg["confidence_threshold"],
            vcfg["nms_threshold"], vcfg["inference_size"])
        if audio is None:
            from aura.audio.speech_output import create_tts
            audio = AudioManager(create_tts(s["audio"]))
        self.audio = audio
        self.memory = memory if memory is not None else MemoryManager.from_config(
            s["memory"], str(resolve_path(s["memory"]["database_path"])))
        self.sensors = sensors or SensorHub()
        self.tracker = ObjectTracker(**{k: s["tracker"][k] for k in ("iou_threshold", "max_misses", "min_hits")})
        self.scene = SceneAnalyzer()
        self.search = ObjectSearch()
        self.hazards = HazardDetector(self.safety_cfg)
        self.alerts = AlertManager(self.safety_cfg)
        self.skipper = FrameSkipper(vcfg["process_every_n_frames"])
        self.llm = llm if llm is not None else LLMClient(s["ai"])
        self.ocr = ocr if ocr is not None else OCRReader(s["ocr"]["backend"], s["ocr"]["min_confidence"],
                                                          s["ocr"]["language"])
        self._spatial: Optional[SpatialEstimator] = None
        self._visible: List[TrackedObject] = []
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_quality_warn = -1e9
        self._last_mem_write = 0.0
        self._last_nav_tick = 0.0
        self._reconnects = 0
        self.state.set(ai_status=self.llm.status(), cloud_in_use=self.llm.enabled and self.llm.is_cloud
                       and self.llm.allow_cloud)

        self.assistant = Assistant(
            self.state, self.memory, self.scene, self.search, self.ocr, ContextEngine(), self.llm,
            self.alerts, self.get_tracks, self.get_frame, create_location_provider_safe(s["navigation"]),
            MapEngine(s["navigation"]), LandmarkEngine(s["navigation"]["user_agent"]), s["user"])

    # ------------------------------------------------------------------ accessors
    def get_tracks(self) -> List[TrackedObject]:
        return list(self._visible)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def say(self, text: str, priority: AlertLevel = AlertLevel.INFO) -> None:
        self.state.set(last_spoken=text)
        self.audio.speak(text, priority)

    # ------------------------------------------------------------------ lifecycle
    def start(self, announce: bool = True) -> None:
        """Load the model, open the camera and start the vision thread."""
        try:
            self.detector.load()
            self.state.set(detector_ready=True, detector_message="running")
        except (ModelNotFoundError, DetectorError) as exc:
            self.state.set(detector_ready=False, detector_message="not running")
            log.error("%s", exc)
            print(f"[AURA] {exc}")
            if announce:
                self.say(rules.NO_DETECTOR, AlertLevel.HIGH)
        self._open_camera(announce)
        if self.llm.enabled and self.llm.is_cloud and self.llm.allow_cloud and announce:
            self.say("Notice: cloud AI is enabled. Facts about your surroundings, not images, "
                     "are sent over the internet when you ask open questions.")
        self.state.set(running=True)
        if announce:
            self.say("AURA is ready. I am an aid only and do not replace your cane or guide dog. "
                     "Say help for commands.")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aura-vision", daemon=True)
        self._thread.start()

    def _open_camera(self, announce: bool) -> bool:
        try:
            self.camera.open()
            self.state.set(camera_ok=True, camera_status=f"working ({self.camera.description})")
            self.events.publish(TOPIC_CAMERA_STATUS, "working")
            return True
        except CameraError as exc:
            self.state.set(camera_ok=False, camera_status="not working")
            print(f"[AURA] {exc}")
            if announce:
                self.say("The camera could not be opened, so I cannot see. Voice commands for memory "
                         "and help still work.", AlertLevel.HIGH)
            return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.state.set(running=False)
        try:
            self.camera.close()
        finally:
            self.audio.close()
            self.memory.close()

    # ------------------------------------------------------------------ vision loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.state.get("camera_ok"):
                    if not self._try_reconnect():
                        self._stop.wait(2.0)
                    continue
                frame = self.camera.read()
                if frame is None:
                    self._stop.wait(0.02)
                    continue
                if not self.skipper.should_process():
                    continue
                self.process_frame(frame)
                self._periodic()
            except CameraError as exc:
                log.error("Camera failed: %s", exc)
                self.state.set(camera_ok=False, camera_status="lost", view_quality_ok=False)
                self._visible = []
                self.say(rules.CAMERA_LOST, AlertLevel.CRITICAL)
            except Exception:  # noqa: BLE001 - keep the safety loop alive
                log.exception("Frame processing error")
                self._stop.wait(0.2)

    def _try_reconnect(self) -> bool:
        if self._reconnects >= MAX_CAMERA_RECONNECTS or not hasattr(self.camera, "reconnect"):
            return False
        self._reconnects += 1
        if self.camera.reconnect():  # type: ignore[attr-defined]
            self._reconnects = 0
            self.state.set(camera_ok=True, camera_status=f"working ({self.camera.description})")
            self.say("The camera is working again.")
            return True
        return False

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run the whole pipeline on one frame. Public so tests / --image can call it."""
        t = time.time()
        h, w = frame.shape[:2]
        with self._frame_lock:
            self._latest_frame = frame
        if self._spatial is None or (self._spatial.w, self._spatial.h) != (w, h):
            zone = self.safety_cfg["walking_zone"]
            self._spatial = SpatialEstimator(w, h, zone["x_min"], zone["x_max"])
        v = self.settings["vision"]
        quality = check_quality(frame, v["min_blur_variance"], v["min_brightness"], v["max_brightness"])
        self.state.set(view_quality_ok=quality.ok, view_quality_issues=quality.issues, frame_size=(w, h),
                       last_frame_time=t)
        if not quality.ok and self.safety_cfg.get("degraded_view_warning", True):
            if t - self._last_quality_warn > 60 and not self.state.get("paused"):
                self._last_quality_warn = t
                self.say(rules.DEGRADED_VIEW, AlertLevel.MEDIUM)

        detections: List[Detection] = []
        if self.state.get("detector_ready"):
            try:
                detections = self.detector.detect(frame)
            except DetectorError as exc:
                log.error("%s", exc)
                self.state.set(detector_ready=False, detector_message="failed")
                self.say(rules.NO_DETECTOR, AlertLevel.HIGH)
        # A frame that is too dark/blurry must not produce confident detections
        # of "nothing"; detections that do appear are kept, absence is never trusted.
        update = self.tracker.update(detections, lambda d: self._spatial.estimate(d))  # type: ignore[union-attr]
        self._visible = update.visible
        self.state.set(tracked=update.visible, frames_processed=self.state.get("frames_processed") + 1)

        if not self.state.get("paused"):
            alerts: List[Alert] = self.hazards.evaluate(update.visible, h) if self.state.get("walking_mode") else []
            for reading in self.sensors.poll():
                alerts.extend(self.hazards.from_distance_sensor(reading.distance_cm, reading.direction))
            chosen = self.alerts.process(alerts)
            if chosen:
                self.events.publish(TOPIC_ALERT, chosen)
                self.say(chosen.message, chosen.level)
            if self.search.active:
                msg = self.search.update(update.visible)
                if msg:
                    self.say(msg, AlertLevel.LOW)
                if not self.search.active:
                    self.state.set(search_target=None)
        self.events.publish(TOPIC_FRAME_PROCESSED, len(detections))
        return detections

    def _periodic(self) -> None:
        t = time.time()
        if self.memory.enabled and t - self._last_mem_write > 2.0:
            self._last_mem_write = t
            self.memory.observe(self._visible)
        if self.assistant.guidance is not None and t - self._last_nav_tick > 3.0:
            self._last_nav_tick = t
            msg = self.assistant.navigation_tick()
            if msg and not self.state.get("paused"):
                self.say(msg, AlertLevel.LOW)

    # ------------------------------------------------------------------ commands
    def handle_text(self, text: str) -> Response:
        """Parse and execute one user utterance. Returns what AURA said."""
        cmd = parse_command(text)
        self.events.publish(TOPIC_COMMAND, cmd.intent.value)
        self.state.set(last_command=cmd.intent.value)  # intent only: no user text stored
        local = self._local_control(cmd)
        if local is not None:
            return local
        resp = self.assistant.handle(cmd)
        self.say(resp.text, resp.priority)
        return resp

    def _local_control(self, cmd: Command) -> Optional[Response]:
        if cmd.intent == Intent.EMPTY:
            return Response("")
        if cmd.intent == Intent.STOP_TALKING:
            self.audio.stop_talking()
            return Response("")
        if cmd.intent == Intent.REPEAT:
            self.audio.repeat()
            return Response(self.audio.last_message)
        if cmd.intent == Intent.PAUSE:
            self.state.set(paused=True)
            msg = ("Assistance paused. I will not warn you about obstacles until you say resume. "
                   "Please use your cane or guide dog.")
            self.say(msg, AlertLevel.HIGH)
            return Response(msg)
        if cmd.intent == Intent.RESUME:
            self.state.set(paused=False)
            self.alerts.reset()
            self.say("Assistance resumed.")
            return Response("Assistance resumed.")
        if cmd.intent == Intent.QUIT:
            self.say("Goodbye.")
            self._stop.set()
            return Response("Goodbye.")
        return None

    # ------------------------------------------------------------------ interactive
    def run_interactive(self, speech_input: SpeechInput) -> None:
        """Blocking command loop (keyboard or microphone)."""
        try:
            while not self._stop.is_set():
                text = speech_input.listen()
                if text is None:
                    break
                self.handle_text(text)
        except KeyboardInterrupt:
            pass
        finally:
            speech_input.close()


def create_location_provider_safe(nav_cfg: Dict[str, Any]):
    """Location provider that degrades to 'no location' instead of crashing on bad config."""
    from aura.navigation.gps import LocationError, NoLocationProvider
    try:
        return create_location_provider(nav_cfg)
    except LocationError as exc:
        print(f"[AURA] {exc}")
        return NoLocationProvider()
