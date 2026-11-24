"""Optional conversational AI over an OpenAI-compatible HTTP API.

Works with local servers (Ollama, llama.cpp, LM Studio - e.g.
``http://127.0.0.1:11434/v1``) or a cloud provider. Rules enforced here:

* Disabled by default.
* A non-local endpoint is refused unless ``ai.allow_cloud: true``; the
  user is told when cloud processing is active.
* The API key is read from an environment variable only.
* Only the structured ``Context`` JSON is sent - never images or audio.
* Failures raise :class:`LLMUnavailable`; AURA keeps working without it.
"""

from __future__ import annotations

import ipaddress
import json
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from aura.config import get_secret
from aura.utils.logger import get_logger

log = get_logger("llm")

SYSTEM_PROMPT = (
    "You are AURA, a voice assistant for a blind or visually impaired person. "
    "Answer in one to three short, plain spoken sentences. "
    "Use ONLY the facts in the JSON the user provides. Never invent objects, positions, "
    "distances, text or places. If the facts do not contain the answer, say you do not know. "
    "Distances are approximate; say 'appears'. Remembered objects may be outdated; mention how long ago. "
    "NEVER say that a road, crossing, stairs, path or area is safe or clear, and never advise "
    "when to cross or walk. For such questions say you cannot judge safety and to use "
    "their cane, guide dog or established method."
)


class LLMUnavailable(Exception):
    """The AI backend is disabled, blocked, unreachable or returned junk."""


def is_local_endpoint(url: str) -> bool:
    """True for localhost and private-network addresses."""
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1", "host.docker.internal") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


class LLMClient:
    """Minimal chat-completions client."""

    def __init__(self, cfg: Dict[str, Any], session: Any = None) -> None:
        self.enabled = bool(cfg.get("enabled", False))
        self.endpoint = str(cfg.get("endpoint", "")).rstrip("/")
        self.model = str(cfg.get("model", ""))
        self.timeout = float(cfg.get("timeout_s", 15))
        self.api_key = get_secret(str(cfg.get("api_key_env", "AURA_LLM_API_KEY")))
        self.allow_cloud = bool(cfg.get("allow_cloud", False))
        self._session = session
        self.is_cloud = bool(self.endpoint) and not is_local_endpoint(self.endpoint)

    def status(self) -> str:
        """One-line status for the user (mentions cloud use explicitly)."""
        if not self.enabled:
            return "AI assistant is off. Basic commands still work."
        if self.is_cloud and not self.allow_cloud:
            return "AI assistant is blocked: the endpoint is a cloud service and cloud use is not allowed."
        where = "cloud service, your scene facts are sent over the internet" if self.is_cloud else "local"
        return f"AI assistant is on ({where})."

    def ask(self, question: str, facts: Dict[str, Any]) -> str:
        """Ask the model. Raises LLMUnavailable on any problem."""
        if not self.enabled:
            raise LLMUnavailable("AI assistant is disabled.")
        if not self.endpoint or not self.model:
            raise LLMUnavailable("AI endpoint or model is not configured.")
        if self.is_cloud and not self.allow_cloud:
            raise LLMUnavailable("Cloud AI is not allowed by your privacy settings.")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "temperature": 0.2, "max_tokens": 160,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}\n\nQUESTION: {question}"}]}
        try:
            if self._session is None:
                import requests
                self._session = requests
            resp = self._session.post(f"{self.endpoint}/chat/completions", headers=headers,
                                      json=body, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except LLMUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - network, HTTP, JSON shape: all mean "unavailable"
            log.warning("AI request failed: %s", type(exc).__name__)
            raise LLMUnavailable("The AI assistant could not be reached.") from exc
        if not text:
            raise LLMUnavailable("The AI assistant returned an empty answer.")
        return text
