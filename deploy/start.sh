#!/usr/bin/env bash
# Launched by fpv-vision.service. Source and options come from /etc/fpv-vision.env.
set -euo pipefail
cd /opt/fpv-vision
# shellcheck disable=SC2086  # FPV_EXTRA_ARGS is intentionally word-split
exec /opt/fpv-vision/.venv/bin/python -m fpv_vision ${FPV_EXTRA_ARGS:-}
