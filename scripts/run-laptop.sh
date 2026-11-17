#!/usr/bin/env bash
# Run fpv-vision on a laptop (primary: Apple Silicon Mac with MPS) in the foreground.
#
#   scripts/run-laptop.sh                    # serve http://0.0.0.0:8110
#   scripts/run-laptop.sh --replay ~/Movies/FPV/flight1.mp4
#   scripts/run-laptop.sh --setup-only       # venv + sample clips + model weights, then exit (run online)
#
# Config comes from the ENVIRONMENT only (the platform's laptop/start-mac.sh sources
# ~/.config/isr-kit/station.env first). Exits cleanly on SIGTERM/Ctrl-C.
set -euo pipefail

# Started from a launcher (start-mac.sh, launchd) PATH can be bare: find uv and Homebrew's Python anyway.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VENV="${FPV_VENV:-$REPO/.venv}"
REQ="$REPO/requirements-desktop.txt"
UV="${FPV_UV:-$(command -v uv 2>/dev/null || true)}"

pick_python() {
    for p in python3.12 python3.11 python3.13 python3; do
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
            echo "$p"; return 0
        fi
    done
    return 1
}

pip_install() {  # pip_install <requirements file> [extra uv args...]
    local file="$1"; shift
    if [ -n "$UV" ]; then
        "$UV" pip install -q -p "$VENV/bin/python" "$@" -r "$file"
    elif "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        "$VENV/bin/python" -m pip install -q -r "$file"
    else
        echo "[fpv-vision] neither uv nor pip found (a uv-made venv has no pip): install uv, or run with Homebrew on PATH" >&2
        return 1
    fi
}

deps_ok() {  # what fpv-vision needs to start (the rest is optional and reported by the app)
    "$VENV/bin/python" - <<'PY'
import importlib.util, sys
need = ["cv2", "numpy", "fastapi", "uvicorn", "paho.mqtt.client"]
missing = [m for m in need if importlib.util.find_spec(m.split(".")[0]) is None]
optional = ["ultralytics", "torch", "lap", "onnxruntime", "mss", "google.genai"] + (["AVFoundation"] if sys.platform == "darwin" else [])
gone = [m for m in optional if importlib.util.find_spec(m.split(".")[0]) is None]
if gone:
    print("[fpv-vision] optional packages missing: " + ", ".join(gone), file=sys.stderr)
if missing:
    print("[fpv-vision] required packages missing: " + ", ".join(missing), file=sys.stderr)
sys.exit(1 if missing else 0)
PY
}

flat_reqs() {  # both requirement files as one list, without the -r line
    cat requirements.txt
    grep -v '^[[:space:]]*-r[[:space:]]' "$REQ"
}

setup_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        echo "[fpv-vision] creating venv in $VENV"
        mkdir -p "$(dirname "$VENV")"
        if [ -n "$UV" ]; then
            "$UV" venv -q -p 3.11 "$VENV"
        else
            PY="$(pick_python)" || { echo "Need Python 3.10-3.13 (brew install python@3.12, or install uv)" >&2; exit 1; }
            "$PY" -m venv "$VENV"
        fi
    fi
    # Install only when the requirements change, and then only the new lines. Offline-safe: demo day
    # has no internet, so a failed install never stops the tool when what it needs is already there.
    local stamp prev new ok
    stamp="$VENV/.req-$(cat requirements.txt "$REQ" | shasum | cut -c1-12)"
    prev="$VENV/.req.txt"   # the requirements last installed (both files, flattened)
    if [ -f "$stamp" ]; then
        [ -f "$prev" ] || flat_reqs > "$prev"   # venvs from before .req.txt: record what they have
        return 0
    fi
    new="$REQ"
    if [ -f "$prev" ] && ls "$VENV"/.req-* >/dev/null 2>&1; then
        new="$(mktemp "${TMPDIR:-/tmp}/fpv-req.XXXXXX")"
        flat_reqs | grep -vxF -f "$prev" | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$' > "$new" || true
    fi
    ok=0
    if [ "$new" != "$REQ" ] && [ ! -s "$new" ]; then
        echo "[fpv-vision] requirements changed, nothing new to install"
        ok=1
    else
        echo "[fpv-vision] installing dependencies (first run downloads torch; a few minutes)"
        if pip_install "$new"; then
            ok=1
        elif [ -n "$UV" ] && "$UV" pip install -q --offline -p "$VENV/bin/python" -r "$new"; then
            echo "[fpv-vision] installed from the local package cache (offline)"
            ok=1
        fi
    fi
    [ "$new" = "$REQ" ] || rm -f "$new"
    if [ "$ok" = 1 ]; then
        rm -f "$VENV"/.req-* && touch "$stamp"   # old stamps only: .req.txt has a dot, not a dash
        flat_reqs > "$prev"
    elif deps_ok; then
        echo "[fpv-vision] WARNING: couldn't install the updated requirements (no internet?): starting with what is installed." >&2
        echo "[fpv-vision]          It retries next start; or run scripts/run-laptop.sh --setup-only on a network." >&2
    else
        echo "[fpv-vision] ERROR: the packages fpv-vision needs aren't installed and couldn't be (no internet?)." >&2
        echo "[fpv-vision]        Connect to a network and run scripts/run-laptop.sh --setup-only." >&2
        exit 1
    fi
}

export FPV_DATA_DIR="${FPV_DATA_DIR:-$HOME/Library/Application Support/isr-kit/fpv-vision}"
if [ "${1:-}" = "--setup-only" ]; then
    setup_venv
    mkdir -p "$FPV_DATA_DIR"
    # demo day is offline: fetch the sample clips and model weights now (retries with backoff)
    if "$VENV/bin/python" -m fpv_vision --prefetch; then
        echo "[fpv-vision] ready (works offline)"
    else
        echo "[fpv-vision] setup done, but something is missing (see above); rerun while online" >&2
        exit 1
    fi
    exit 0
fi
setup_venv

# ---- defaults for a Mac on the isr-kit hotspot (only applied when not already set) ----
export FPV_HOST="${FPV_HOST:-0.0.0.0}"
export FPV_PORT="${FPV_PORT:-8110}"
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
export VLM_MODEL="${VLM_MODEL:-qwen3-vl:8b-instruct}"
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
