"""Draw detections and a HUD on frames."""

from __future__ import annotations

import time

import cv2
import numpy as np

from fpv_vision.detector import Detection

# BGR
COLOURS = {"person": (40, 40, 235), "vehicle": (0, 165, 255), "other": (220, 200, 40)}


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
         recording: bool = False, badge: str = "") -> np.ndarray:
    img = frame
    h, w = img.shape[:2]
    thick = max(1, round(w / 640))
    fs = 0.45 * max(1.0, w / 960)
    for d in dets:
        c = COLOURS.get(d.category, COLOURS["other"])
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thick + (1 if d.category != "other" else 0))
        tag = f"{d.label}{f' #{d.track_id}' if d.track_id is not None else ''} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), c, -1)
        cv2.putText(img, tag, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    lines = list(hud or [])
    lines.append(time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()))
    y = 8
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(img, (6, y), (12 + tw, y + th + 8), (0, 0, 0), -1)
        cv2.putText(img, line, (9, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + 10
    if badge:
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        x0 = w - tw - 14
        colour = (0, 170, 0) if badge == "LIVE" else (0, 140, 230)
        cv2.rectangle(img, (x0, 8), (w - 6, 16 + th), (0, 0, 0), -1)
        cv2.rectangle(img, (x0, 8), (w - 6, 16 + th), colour, 1)
        cv2.putText(img, badge, (x0 + 4, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    if recording and int(time.time() * 2) % 2 == 0:
        cv2.circle(img, (w - 24, 44), 9, (0, 0, 255), -1)
        cv2.putText(img, "REC", (w - 70, 50), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if width <= 0 or w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
