"""The assistant: turns a parsed :class:`Command` into a spoken answer.

Everything safety-related and everything about positions comes from
deterministic code. The optional LLM is used only for free-form questions
and only ever sees the structured ``Context``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np

from aura.audio.command_parser import HELP_TEXT, Command, Intent
from aura.core.state_manager import StateManager
from aura.intelligence import response_generator as rg
from aura.intelligence.context_engine import ContextEngine
from aura.intelligence.llm import LLMClient, LLMUnavailable
from aura.memory.memory_manager import MemoryManager
from aura.memory.object_memory import Freshness
from aura.navigation.gps import LocationError, LocationProvider
from aura.navigation.landmark_engine import LandmarkEngine
from aura.navigation.map_engine import MapEngine, MapError
from aura.navigation.route_guidance import RouteGuidance
from aura.safety.alert_manager import AlertManager
from aura.types import AlertLevel, TrackedObject
from aura.utils.helpers import humanize_age, now, spoken_name
from aura.vision.labels import resolve_target
from aura.vision.object_search import ObjectSearch
from aura.vision.ocr import OCRReader
from aura.vision.scene_analyzer import SceneAnalyzer
from aura.vision.spatial_estimator import describe_direction

CONFIRM_WINDOW_S = 20.0


@dataclass
class Response:
    text: str
    priority: AlertLevel = AlertLevel.INFO


class Assistant:
    """Handles user commands. Holds no threads; the orchestrator calls it."""

    def __init__(self, state: StateManager, memory: MemoryManager, scene: SceneAnalyzer,
                 search: ObjectSearch, ocr: OCRReader, context: ContextEngine, llm: LLMClient,
                 alerts: AlertManager, get_tracks: Callable[[], List[TrackedObject]],
                 get_frame: Callable[[], Optional[np.ndarray]],
                 location: LocationProvider, maps: MapEngine, landmarks: LandmarkEngine,
                 user_cfg: Optional[dict] = None, clock: Callable[[], float] = now) -> None:
        self.state, self.memory, self.scene, self.search = state, memory, scene, search
        self.ocr, self.context, self.llm, self.alerts = ocr, context, llm, alerts
        self.get_tracks, self.get_frame = get_tracks, get_frame
        self.location, self.maps, self.landmarks = location, maps, landmarks
        self.user_cfg = user_cfg or {}
        self._clock = clock
        self.guidance: Optional[RouteGuidance] = None
        self._pending_clear = 0.0

    # ------------------------------------------------------------------ helpers
    def _vision_problem(self) -> Optional[str]:
        if not self.state.get("camera_ok"):
            return rg.NO_CAMERA
        if not self.state.get("detector_ready"):
            return rg.NOT_RUNNING_VISION
        if self.state.get("frames_processed") == 0:
            return rg.NO_FRAME_YET
        return None

    def _prefix(self) -> str:
        return rg.degraded_prefix(self.state.get("view_quality_issues"))

    # ------------------------------------------------------------------ dispatch
    def handle(self, cmd: Command) -> Response:
        """Answer a command. Never raises; failures become honest messages."""
        try:
            handler = getattr(self, f"_do_{cmd.intent.value}", None)
            if handler is None:
                return Response(rg.UNKNOWN_COMMAND)
            return handler(cmd)
        except Exception:  # noqa: BLE001 - the assistant must never crash the app
            from aura.utils.logger import get_logger
            get_logger("assistant").exception("Command failed: %s", cmd.intent)
            return Response("Sorry, something went wrong with that request.")

    # ------------------------------------------------------------------ scene
    def _do_describe(self, cmd: Command) -> Response:
        problem = self._vision_problem()
        if problem:
            return Response(problem)
        return Response(self._prefix() + self.scene.summarize(self.get_tracks()))

    def _do_what_in_front(self, cmd: Command) -> Response:
        problem = self._vision_problem()
        if problem:
            return Response(problem)
        return Response(self._prefix() + self.scene.what_is_in_front(self.get_tracks()))

    # ------------------------------------------------------------------ search
    def _memory_line(self, target: str) -> str:
        rec, _ = self.memory.last_seen(target)
        return self.memory.objects.describe(rec, target) if (self.memory.enabled and rec) else ""

    def _do_find(self, cmd: Command) -> Response:
        target = cmd.target or ""
        problem = self.search.start(target)
        if problem:
            mem = self._memory_line(target)
            return Response((problem + " " + mem).strip())
        self.state.set(search_target=target)
        vp = self._vision_problem()
        if vp:
            self.search.cancel()
            self.state.set(search_target=None)
            return Response(vp)
        labels, _ = resolve_target(target)
        seen = self.scene.where_is(labels, target, self.get_tracks())
        if seen:
            return Response(seen + " Say 'guide me' for directions, or 'cancel' to stop.")
        return Response(f"Looking for {target}. Turn slowly and I'll tell you when I see it. "
                        "Say 'cancel' to stop.")

    def _do_where_is(self, cmd: Command) -> Response:
        target = cmd.target or ""
        labels, problem = resolve_target(target)
        if problem:
            return Response(problem)
        vp = self._vision_problem()
        if vp:
            return Response(vp)
        seen = self.scene.where_is(labels, target, self.get_tracks())
        if seen:
            return Response(self._prefix() + seen)
        text = f"I don't see a {target} right now."
        mem = self._memory_line(target)
        return Response(text + (" " + mem if mem else ""))

    def _do_guide(self, cmd: Command) -> Response:
        return Response(self.search.guide(self.get_tracks()))

    def _do_cancel(self, cmd: Command) -> Response:
        was = self.search.active or self.guidance is not None
        self.search.cancel()
        self.guidance = None
        self.state.set(search_target=None, navigating=False)
        self._pending_clear = 0.0
        return Response("Cancelled." if was else "Nothing to cancel.")

    # ------------------------------------------------------------------ OCR
    def _do_read(self, cmd: Command) -> Response:
        if not self.state.get("camera_ok"):
            return Response(rg.NO_CAMERA)
        frame = self.get_frame()
        result = self.ocr.read(frame)
        return Response(self.ocr.speak_text(result, important_only=(cmd.extra == "important")))

    # ------------------------------------------------------------------ memory
    def _do_remember_location(self, cmd: Command) -> Response:
        if not self.memory.enabled:
            return Response("Memory is turned off in your settings.")
        name = self.memory.spatial.remember_location(cmd.target)
        return Response(f"Saved this location as {name}.")

    def _do_set_room(self, cmd: Command) -> Response:
        room = self.memory.spatial.set_room(cmd.target or "")
        self.state.set(current_room=room)
        return Response(f"Okay, you are in the {room}. I will use that for new memories.")

    def _do_last_seen(self, cmd: Command) -> Response:
        target = cmd.target or ""
        if not self.memory.enabled:
            return Response("Memory is turned off in your settings.")
        labels, problem = resolve_target(target)
        if labels:
            live = self.scene.where_is(labels, target, self.get_tracks())
            if live:
                return Response(f"Currently detected: {live}")
        rec, _ = self.memory.last_seen(target)
        return Response(self.memory.objects.describe(rec, target))

    def _do_confirm_location(self, cmd: Command) -> Response:
        if not self.memory.enabled:
            return Response("Memory is turned off in your settings.")
        self.memory.confirm(cmd.target or "", cmd.extra or "somewhere")
        return Response(f"Okay, I noted that your {cmd.target} is {cmd.extra}. "
                        "I will treat that as something you told me, and it may become outdated.")

    def _do_memory_history(self, cmd: Command) -> Response:
        if not self.memory.enabled:
            return Response("Memory is turned off in your settings.")
        labels = None
        if cmd.target:
            labels, _ = resolve_target(cmd.target)
            recs = self.memory.objects.search(cmd.target, 5) if not labels else self.memory.objects.history(labels, 5)
        else:
            recs = self.memory.objects.history(None, 5)
        if not recs:
            return Response("I have no matching memories.")
        t = self._clock()
        parts = []
        for r in recs:
            where = f"near the {r.near_object}" if r.near_object else r.position
            parts.append(f"{r.name} {where} in {r.room}, {humanize_age(t - r.observed_at)} ({r.freshness.value})")
        return Response("Recent memories: " + "; ".join(parts) + ".")

    def _do_forget(self, cmd: Command) -> Response:
        if not self.memory.enabled:
            return Response("Memory is turned off in your settings.")
        n = self.memory.forget(cmd.target or "")
        return Response(f"Deleted {n} memor{'y' if n == 1 else 'ies'} of {cmd.target}." if n
                        else f"I had no memories of {cmd.target}.")

    def _do_clear_memory(self, cmd: Command) -> Response:
        t = self._clock()
        if t - self._pending_clear <= CONFIRM_WINDOW_S and self._pending_clear > 0:
            n = self.memory.clear_all()
            self._pending_clear = 0.0
            return Response(f"All memory cleared. {n} entries deleted.")
        self._pending_clear = t
        return Response("This will permanently delete everything I remember. "
                        "Say 'clear all memory' again within 20 seconds to confirm, or 'cancel'.")

    # ------------------------------------------------------------------ safety / status
    def _do_crossing(self, cmd: Command) -> Response:
        return Response(rg.crossing_response(), AlertLevel.HIGH)

    def _do_status(self, cmd: Command) -> Response:
        snap = self.state.snapshot()
        snap["ai"] = self.llm.status()
        return Response(rg.status_text(snap, {}, self.memory.objects.count() if self.memory.enabled else 0))

    def _do_help(self, cmd: Command) -> Response:
        return Response(HELP_TEXT)

    def _do_emergency(self, cmd: Command) -> Response:
        name = self.user_cfg.get("emergency_contact_name") or ""
        phone = self.user_cfg.get("emergency_contact_phone") or ""
        base = "I cannot call anyone or contact emergency services for you yet. "
        if phone:
            who = f" {name}," if name else ""
            return Response(base + f"Your saved emergency contact is{who} number {' '.join(phone)}. "
                            "If you are in danger, call your local emergency number now.", AlertLevel.CRITICAL)
        return Response(base + "No emergency contact is saved. Add one under 'user' in config/settings.yaml. "
                        "If you are in danger, call your local emergency number now.", AlertLevel.CRITICAL)

    # ------------------------------------------------------------------ navigation
    def _do_navigate(self, cmd: Command) -> Response:
        dest = cmd.target or ""
        try:
            loc = self.location.get_location()
            place = self.maps.geocode(dest, near=(loc.lat, loc.lon))
            route = self.maps.walking_route((loc.lat, loc.lon), place["name"], (place["lat"], place["lon"]))
        except (LocationError, MapError) as exc:
            return Response(str(exc))
        self.guidance = RouteGuidance(route)
        self.state.set(navigating=True)
        first = self.guidance.update(loc)
        return Response(self.guidance.summary() + (" " + first if first else ""))

    def navigation_tick(self) -> Optional[str]:
        """Called periodically by the orchestrator while navigating."""
        if self.guidance is None:
            return None
        try:
            msg = self.guidance.update(self.location.get_location())
        except LocationError:
            return None
        if self.guidance.arrived:
            self.guidance = None
            self.state.set(navigating=False)
        return msg

    def _do_where_am_i(self, cmd: Command) -> Response:
        try:
            loc = self.location.get_location()
        except LocationError as exc:
            return Response(str(exc))
        acc = f" accurate to about {int(loc.accuracy_m)} metres" if loc.accuracy_m else ""
        text = f"Your position is latitude {loc.lat:.5f}, longitude {loc.lon:.5f}{acc}."
        try:
            text += " " + self.landmarks.describe(loc.lat, loc.lon)
        except RuntimeError as exc:
            text += " " + str(exc)
        return Response(text)

    # ------------------------------------------------------------------ AI
    def _do_ask(self, cmd: Command) -> Response:
        question = cmd.target or cmd.raw
        recs = self.memory.objects.history(None, 8) if self.memory.enabled else []
        ctx = self.context.build(cmd, self.get_tracks(), self.alerts.history, self.memory.room,
                                 self.state.get("search_target"), recs, self._clock(),
                                 view_issues=self.state.get("view_quality_issues"),
                                 detector_ready=self.state.get("detector_ready"))
        try:
            answer = self.llm.ask(question, ctx.to_llm_payload())
            prefix = "Using the cloud assistant. " if self.llm.is_cloud else ""
            return Response(prefix + answer)
        except LLMUnavailable as exc:
            return Response(f"{exc} I can still describe your surroundings, find objects, "
                            "read text and recall where things were. Say 'help' for the list.")
