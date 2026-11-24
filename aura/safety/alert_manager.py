"""Decides which alerts are actually spoken: dedup, cooldowns, rate limit.

Rules:
  * Alerts below ``min_spoken_level`` are dropped (stay quiet).
  * The same key ("car:right") is not repeated inside its level's cooldown,
    UNLESS its severity increased (escalation always breaks through).
  * A global minimum gap prevents machine-gun warnings, except CRITICAL.
  * Only ONE alert is emitted per call (the most severe).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from aura.safety.safety_rules import level_from_name
from aura.types import Alert, AlertLevel
from aura.utils.helpers import now


class AlertManager:
    def __init__(self, cfg: Dict[str, Any], clock: Callable[[], float] = now) -> None:
        self.cfg = cfg
        self._clock = clock
        self.min_level = level_from_name(cfg.get("min_spoken_level", "LOW"))
        self.cooldowns = {AlertLevel[k]: float(v) for k, v in cfg["cooldown_seconds"].items()}
        self.global_gap = float(cfg.get("global_min_gap_seconds", 1.5))
        self._last: Dict[str, Tuple[float, AlertLevel]] = {}
        self._last_any = -1e9
        self.history: List[Alert] = []

    def reset(self) -> None:
        self._last.clear()
        self._last_any = -1e9

    def process(self, alerts: List[Alert]) -> Optional[Alert]:
        """Return the single alert to speak now, or None to stay silent."""
        t = self._clock()
        candidates = [a for a in alerts if a.level >= self.min_level]
        candidates.sort(key=lambda a: a.level, reverse=True)
        for alert in candidates:
            prev = self._last.get(alert.key)
            escalated = prev is not None and alert.level > prev[1]
            if prev is not None and not escalated:
                if (t - prev[0]) < self.cooldowns.get(alert.level, 10.0):
                    continue
            if alert.level < AlertLevel.CRITICAL and (t - self._last_any) < self.global_gap:
                continue
            alert.timestamp = t
            self._last[alert.key] = (t, alert.level)
            self._last_any = t
            self.history.append(alert)
            del self.history[:-50]
            return alert
        return None
