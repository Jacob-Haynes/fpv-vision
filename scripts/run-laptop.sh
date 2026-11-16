#!/usr/bin/env bash
# Run fpv-vision on a laptop (primary: Apple Silicon Mac with MPS) in the foreground.
#
#   scripts/run-laptop.sh                    # serve http://0.0.0.0:8110
#   scripts/run-laptop.sh --replay ~/Movies/FPV/flight1.mp4
#   scripts/run-laptop.sh --setup-only       # create/refresh the venv and exit
#
# Config comes from the ENVIRONMENT only (the platform's laptop/start-mac.sh sources
# ~/.config/isr-kit/station.env first). Exits cleanly on SIGTERM/Ctrl-C.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv"
REQ="$REPO/requirements-desktop.txt"

pick_python() {
    for p in python3.12 python3.11 python3.13 python3; do
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
            echo "$p"; return 0
        fi
    done
    return 1
}

setup_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        echo "[fpv-vision] creating venv in $VENV"
        if command -v uv >/dev/null 2>&1; then
            uv venv -q -p 3.11 "$VENV"
        else
            PY="$(pick_python)" || { echo "Need Python 3.10-3.13 (brew install python@3.12, or install uv)" >&2; exit 1; }
            "$PY" -m venv "$VENV"
        fi
    fi
    # reinstall only when the requirements change (works offline once installed)
    local stamp
    stamp="$VENV/.req-$(cat requirements.txt "$REQ" | shasum | cut -c1-12)"
    if [ ! -f "$stamp" ]; then
        echo "[fpv-vision] installing dependencies (first run downloads torch; a few minutes)"
        if command -v uv >/dev/null 2>&1; then
            uv pip install -q -p "$VENV/bin/python" -r "$REQ"
        else
            "$VENV/bin/python" -m pip install -q --upgrade pip
            "$VENV/bin/python" -m pip install -q -r "$REQ"
        fi
        rm -f "$VENV"/.req-* && touch "$stamp"
    fi
}

if [ "${1:-}" = "--setup-only" ]; then setup_venv; echo "[fpv-vision] ready"; exit 0; fi
setup_venv

# ---- defaults for a Mac on the isr-kit hotspot (only applied when not already set) ----
export FPV_HOST="${FPV_HOST:-0.0.0.0}"
export FPV_PORT="${FPV_PORT:-8110}"
export FPV_DATA_DIR="${FPV_DATA_DIR:-$HOME/Library/Application Support/isr-kit/fpv-vision}"
export FPV_FOOTAGE_DIRS="${FPV_FOOTAGE_DIRS:-$HOME/Movies/FPV}"
export FPV_TARGET_FPS="${FPV_TARGET_FPS:-30}"
export FPV_STREAM_WIDTH="${FPV_STREAM_WIDTH:-1280}"
export FPV_DEVICE="${FPV_DEVICE:-auto}"
export TAK_HOST="${TAK_HOST:-isr-pi.local}"
export TAK_PORT="${TAK_PORT:-8087}"
export MQTT_HOST="${MQTT_HOST:-isr-pi.local}"
export MQTT_PORT="${MQTT_PORT:-1883}"
export MQTT_USER="${MQTT_USER:-isrkit}"
export VLM_URL="${VLM_URL:-http://isr-hq.local:11434}"
export VLM_MODEL="${VLM_MODEL:-qwen3-vl:8b}"
if [ -z "${MQTT_PASS:-}" ]; then
    echo "[fpv-vision] WARNING: MQTT_PASS not set; heartbeat and mesh texts will fail (guest only works on the Pi)" >&2
fi
mkdir -p "$FPV_DATA_DIR"
if [ -z "${FPV_SOURCE:-}" ]; then
    first_dir="${FPV_FOOTAGE_DIRS%%:*}"
    if [ -d "$first_dir" ] && [ -n "$(find "$first_dir" -maxdepth 2 -type f \( -iname '*.mp4' -o -iname '*.mov' \) -print -quit)" ]; then
        export FPV_SOURCE="replay:$first_dir"
    else
        export FPV_SOURCE="simulate"
    fi
fi

echo "[fpv-vision] source=$FPV_SOURCE  TAK=$TAK_HOST:$TAK_PORT  MQTT=$MQTT_USER@$MQTT_HOST:$MQTT_PORT  VLM=$VLM_URL ($VLM_MODEL)"
# exec: python receives SIGTERM directly and shuts down cleanly (publishes heartbeat 'stopped')
exec "$VENV/bin/python" -m fpv_vision "$@"
