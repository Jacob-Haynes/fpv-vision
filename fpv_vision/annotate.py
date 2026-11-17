"""Draw detections and a HUD on frames."""

from __future__ import annotations

import time

import cv2
import numpy as np

from fpv_vision.detector import Detection

def _bgr(hex_rgb: str) -> tuple[int, int, int]:
    v = int(hex_rgb.lstrip("#"), 16)
    return (v & 0xFF, (v >> 8) & 0xFF, v >> 16)


# Overlay palette: the ISTAR kit v2 colours (isr-ui.css), BGR for OpenCV. The only colours drawn
# into the video; change them here, nowhere else.
UNKNOWN = _bgr("#ffff80")  # APP-6 unknown: every detection box
HOSTILE = _bgr("#ff8080")  # APP-6 hostile: reserved for detections that raise an alert (none do yet)
TEXT = _bgr("#dcdcd4")  # off-white label text
TAG_BG = _bgr("#0e1012")  # graphite, drawn semi-transparent behind text
TAG_ALPHA = 0.7
SIM = _bgr("#a493c9")  # provenance badge: simulated
REC = _bgr("#c9a75e")  # provenance badge: recorded / replay (brass)
LIVE = _bgr("#8fb07a")  # provenance badge: live (sage)
FAULT = _bgr("#d4695c")  # recording indicator
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tag(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Semi-transparent graphite rectangle (clipped to the frame)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.addWeighted(roi, 1 - TAG_ALPHA, np.full_like(roi, TAG_BG), TAG_ALPHA, 0)


def _badge_colour(badge: str) -> tuple[int, int, int]:
    b = badge.upper()
    if b.startswith("SIM"):
        return SIM
    if b.startswith(("REPLAY", "REC")):
        return REC
    return LIVE


def scale_detections(dets: list[Detection], sx: float, sy: float) -> list[Detection]:
    if sx == 1 and sy == 1:
        return dets
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        out.append(Detection(d.cls_id, d.label, d.conf,
                             (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)), d.track_id))
    return out


def draw(frame: np.ndarray, dets: list[Detection], hud: list[str] | None = None,
         recording: bool = False, badge: str = "", alert_tracks: set[int] | None = None) -> np.ndarray:
    img = frame
    h, w = img.shape[:2]
    thick = 2 if w >= 1280 else 1  # 1-2 px boxes
    fs = 0.4 * max(1.0, w / 1280)  # small labels
    for d in dets:
        c = HOSTILE if alert_tracks and d.track_id in alert_tracks else UNKNOWN
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
        tag = f"{d.label}{f' {d.track_id}' if d.track_id is not None else ''} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, fs, 1)
        ty = max(th + 4, y1)
        _tag(img, x1, ty - th - 5, x1 + tw + 6, ty)
        cv2.putText(img, tag, (x1 + 3, ty - 3), FONT, fs, TEXT, 1, cv2.LINE_AA)

    lines = list(hud or [])
    lines.append(time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()))
    y = 8
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, fs, 1)
        _tag(img, 6, y, 12 + tw, y + th + 8)
        cv2.putText(img, line, (9, y + th + 3), FONT, fs, TEXT, 1, cv2.LINE_AA)
        y += th + 10
    if badge:
        (tw, th), _ = cv2.getTextSize(badge, FONT, fs, 1)
        x0 = w - tw - 14
        _tag(img, x0, 8, w - 6, 16 + th)
        cv2.rectangle(img, (x0, 8), (w - 6, 16 + th), _badge_colour(badge), 1)
        cv2.putText(img, badge, (x0 + 4, 12 + th), FONT, fs, TEXT, 1, cv2.LINE_AA)
    if recording and int(time.time() * 2) % 2 == 0:
        cv2.circle(img, (w - 20, 30 + int(th if badge else 10)), 6, FAULT, -1)
    return img


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if width <= 0 or w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
