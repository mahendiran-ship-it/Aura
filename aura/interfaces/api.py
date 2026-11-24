"""Small local HTTP API (stdlib only) so an Android app, a web page or a
wearable bridge can drive AURA.

Security defaults: binds to 127.0.0.1 only and REQUIRES a bearer token from
the environment variable named in ``api.token_env``. Without a token the
server refuses to start. It never returns camera images or audio.

  GET  /status                  JSON status
  POST /command  {"text": ".."} run a voice/text command, returns {"response": ".."}
"""

from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from aura.config import get_secret


class ApiConfigError(Exception):
    pass


def make_server(orchestrator: Any, host: str, port: int, token_env: str) -> ThreadingHTTPServer:
    token = get_secret(token_env)
    if not token:
        raise ApiConfigError(f"Set the environment variable {token_env} to a secret token to enable the API.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # keep logs free of user text
            pass

        def _authorised(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, f"Bearer {token}")

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorised():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/status":
                return self._send(200, orchestrator.state.snapshot())
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorised():
                return self._send(401, {"error": "unauthorized"})
            if self.path != "/command":
                return self._send(404, {"error": "not found"})
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 4096)
                text = str(json.loads(self.rfile.read(length) or b"{}").get("text", ""))
            except (ValueError, TypeError):
                return self._send(400, {"error": "invalid JSON"})
            resp = orchestrator.handle_text(text)
            self._send(200, {"response": resp.text})

    return ThreadingHTTPServer((host, port), Handler)


def start_api(orchestrator: Any, cfg: dict) -> Optional[ThreadingHTTPServer]:
    """Start the API in a background thread if enabled."""
    if not cfg.get("enabled"):
        return None
    server = make_server(orchestrator, cfg["host"], int(cfg["port"]), cfg["token_env"])
    threading.Thread(target=server.serve_forever, name="aura-api", daemon=True).start()
    return server
