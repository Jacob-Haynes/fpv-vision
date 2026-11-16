#!/usr/bin/env bash
# Install fpv-vision on a Raspberry Pi 5 (Pi OS Bookworm 64-bit).
# Usage (from the repo checkout):  sudo bash deploy/install.sh
# Idempotent and non-interactive. Does NOT enable the service at boot — pi-profiles starts it.
set -euo pipefail

APP=fpv-vision
APP_DIR=/opt/$APP
DATA_DIR=/var/lib/$APP
ENV_FILE=/etc/$APP.env
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then echo "run with sudo" >&2; exit 1; fi
echo "=== $APP install ==="

echo "[1/7] system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip v4l-utils ffmpeg libgl1 rsync >/dev/null

echo "[2/7] service user"
if ! id -u "$APP" >/dev/null 2>&1; then
    useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin --user-group "$APP"
fi
usermod -a -G video "$APP"

echo "[3/7] copy code to $APP_DIR"
mkdir -p "$APP_DIR"
rsync -a --delete --exclude .venv --exclude .git --exclude data --exclude __pycache__ \
    "$SRC_DIR/fpv_vision" "$SRC_DIR/deploy" "$SRC_DIR/profiles.d" "$SRC_DIR/requirements.txt" "$APP_DIR/"
chmod +x "$APP_DIR/deploy/start.sh"

echo "[4/7] python venv + deps (first run downloads torch; takes a while)"
[ -x "$APP_DIR/.venv/bin/python" ] || python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
# export-time only
"$APP_DIR/.venv/bin/pip" install -q "onnx>=1.16" "onnxslim>=0.1.30" || true

echo "[5/7] data dir, config"
mkdir -p "$DATA_DIR"/{snapshots,clips,models,samples,.ultralytics}
if [ ! -f "$ENV_FILE" ]; then
    install -m 600 -o root -g root "$SRC_DIR/deploy/$APP.env.example" "$ENV_FILE"
    echo "    created $ENV_FILE (edit FPV_SOURCE / FPV_DEFAULT_POSITION / GEMINI_API_KEY)"
else
    echo "    keeping existing $ENV_FILE"
fi

echo "[6/7] model export + sample clip (needs internet once; skipped if already present)"
# read just the keys we need (the env file is systemd syntax, not safe to `source`)
env_get() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '"' || true; }
MODEL="$(env_get FPV_MODEL)"; MODEL="${MODEL:-yolo26n}"
IMGSZ="$(env_get FPV_IMGSZ)"; IMGSZ="${IMGSZ:-416}"
cd "$APP_DIR"
export YOLO_CONFIG_DIR="$DATA_DIR/.ultralytics"
"$APP_DIR/.venv/bin/python" -m fpv_vision.export --model "$MODEL" --format ncnn --imgsz "$IMGSZ" --data-dir "$DATA_DIR" \
    || echo "    WARNING: model export failed (offline?). Re-run install.sh with internet; until then the PyTorch weights or stream-only mode are used."
"$APP_DIR/.venv/bin/python" -m fpv_vision.sample "$DATA_DIR" \
    || echo "    WARNING: sample clip download failed; simulate mode will show a synthetic test pattern."
cd - >/dev/null
chown -R "$APP:$APP" "$DATA_DIR"

echo "[7/7] systemd unit + pi-profiles"
install -m 644 "$APP_DIR/deploy/$APP.service" /etc/systemd/system/$APP.service
systemctl daemon-reload
if [ -d /opt/pi-profiles/profiles.d ]; then
    cp "$APP_DIR"/profiles.d/*.conf /opt/pi-profiles/profiles.d/
    echo "    installed pi-profiles: fpv-vision, bft-fpv"
fi

cat <<MSG

=== install complete ===
  config : $ENV_FILE
  data   : $DATA_DIR  (snapshots, clips, models)
  web UI : http://<pi-ip>:8110

  sudo pi-profiles fpv-vision        # standalone
  sudo pi-profiles bft-fpv           # with OpenTAKServer + mesh bridge
  sudo systemctl start $APP          # or start it by hand
  journalctl -u $APP -f
  sudo -u $APP $APP_DIR/.venv/bin/python -m fpv_vision --benchmark 100 --data-dir $DATA_DIR   # measure FPS
MSG
