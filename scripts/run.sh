#!/usr/bin/env bash
# Start AURA. Extra arguments are passed through, e.g.:  scripts/run.sh --camera 1 --mode LOW_POWER
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
[ -f .env ] && { set -a; . ./.env; set +a; }
exec python3 -m aura "$@"
