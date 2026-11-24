"""Entry point: ``python -m aura`` (or ``python -m aura.main``)."""

from __future__ import annotations

from aura.interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
