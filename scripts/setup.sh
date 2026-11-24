#!/usr/bin/env bash
# AURA setup for Linux (Debian/Ubuntu/Raspberry Pi OS) and Termux.
# Creates a virtualenv, installs only the required Python packages, and reports what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

IS_TERMUX=0
[ -n "${TERMUX_VERSION:-}" ] || [ -d /data/data/com.termux ] && IS_TERMUX=1

echo "== AURA setup =="
command -v python3 >/dev/null || { echo "python3 is required."; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
  || { echo "Python 3.9 or newer is required."; exit 1; }

if [ "$IS_TERMUX" = 1 ]; then
  echo "Termux detected. System packages (run once yourself if missing):"
  echo "  pkg install python opencv-python python-numpy tesseract termux-api"
  echo "  (onnxruntime is not in pkg; see docs/ANDROID.md for options)"
  PIP_FLAGS="--break-system-packages"
else
  PIP_FLAGS=""
  if command -v apt-get >/dev/null; then
    for pkg in tesseract-ocr espeak-ng; do
      dpkg -s "$pkg" >/dev/null 2>&1 || echo "Optional system package missing: sudo apt install $pkg"
    done
  fi
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

python3 -m pip install $PIP_FLAGS --upgrade pip >/dev/null
python3 -m pip install $PIP_FLAGS -r requirements.txt || {
  echo "Some packages failed to install (common on Termux: onnxruntime). See docs/ANDROID.md."; }

echo
python3 -m aura check || true
echo
echo "Next: python scripts/download_models.py   then   scripts/run.sh"
