# Data

AURA's memory database (`aura_memory.sqlite3`) is created here at runtime.

- It stores **structured facts only** (object name, rough position, room name, time, confidence). **No photos, video or audio are ever saved.**
- Everything here is git-ignored. Do not commit it.
- Delete memories by voice ("forget my bag", "clear all memory") or by deleting this file.
- Old observations are purged automatically after `memory.retention_days` (default 30).
