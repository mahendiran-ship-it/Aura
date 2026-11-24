"""Logging setup.

Privacy: AURA logs events and decisions, not what the user says or sees,
unless ``logging.log_user_text`` is explicitly turned on.
"""

from __future__ import annotations

import logging
from typing import Optional

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=not _CONFIGURED,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``aura.<name>``)."""
    return logging.getLogger(name if name.startswith("aura") else f"aura.{name}")
