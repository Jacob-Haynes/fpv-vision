"""Pluggable video sources.

Source spec strings (config ``FPV_SOURCE``, ``--source`` or the web UI):

  simulate | simulate:street   auto-downloaded public-domain sample clip, looped
                           (synthetic test pattern if it cannot be fetched)
  uvc:0 | uvc:/dev/video0  USB UVC capture card (analog FPV receiver, goggles HDMI-out, GoPro HDMI-out)
  rtsp://... rtmp://... http(s)://... udp://... srt://...   network streams (via FFmpeg)
  file:/path/clip.mp4      a video file, looped at its native frame rate (bare paths work too)
  gopro-usb | gopro-usb:172.2X.1YZ.51   GoPro (HERO9+) USB webcam mode, UDP MPEG-TS on port 8554

Every source runs its own capture thread and only keeps the *latest* frame, so a slow
consumer never builds up latency. Live sources reconnect with backoff; files loop.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "srt://", "tcp://")
GOPRO_UDP_URL = "udp://@0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000"



@dataclass
class SourceSpec:
    kind: str  # simulate | uvc | network | file | gopro-usb | synthetic
    target: str  # device index/path, URL, or file path
    raw: str

    @property
    def is_live(self) -> bool:
        return self.kind in ("uvc", "network", "gopro-usb")


def parse_spec(spec: str) -> SourceSpec:
    raw = (spec or "simulate").strip()
    low = raw.lower()
    if low in ("simulate", "sim", "sample") or low.startswith("simulate:"):
        return SourceSpec("simulate", raw.split(":", 1)[1] if ":" in raw else "", raw)
    if low == "synthetic":
        return SourceSpec("synthetic", "", raw)
    if low.startswith("uvc:") or low.startswith("/dev/video"):
        target = raw[4:] if low.startswith("uvc:") else raw
        return SourceSpec("uvc", target or "0", raw)
    if low.isdigit():
        return SourceSpec("uvc", raw, raw)
    if low.startswith("gopro-usb"):
        target = raw.split(":", 1)[1] if ":" in raw else ""
        return SourceSpec("gopro-usb", target, raw)
    if low.startswith(NETWORK_SCHEMES):
        return SourceSpec("network", raw, raw)
    if low.startswith("file:"):
        return SourceSpec("file", raw[5:], raw)
    return SourceSpec("file", raw, raw)


def list_uvc_devices() -> list[str]:
    """Video capture nodes on Linux. Capture dongles usually expose 2 nodes; the first works."""
    if not sys.platform.startswith("linux"):
        return []
    devs = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))
    out = []
    for d in devs:
        name_file = Path(f"/sys/class/video4linux/{Path(d).name}/name")
        name = name_file.read_text().strip() if name_file.exists() else ""
        # Pi 5 exposes many ISP/codec nodes (pispbe, rpi-hevc-dec) — not cameras.
        if any(k in name.lower() for k in ("pispbe", "hevc", "rp1-cfe", "bcm2835")):
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------- GoPro USB webcam


def find_gopro_ip() -> str | None:
    """A GoPro in USB webcam mode appears as a USB network interface on 172.2X.1YZ.0/24.

    The host gets .5x by DHCP and the camera is .51 (X,Y,Z = last 3 digits of the serial).
    """
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for m in re.finditer(r"inet (172\.2\d\.1\d\d)\.\d+/", out):
        return f"{m.group(1)}.51"
    return None


def gopro_start_webcam(ip: str, res: int = 12, fov: int = 0) -> bool:
    """Ask the camera to start streaming MPEG-TS to the host's UDP port 8554 (Open GoPro HTTP API).

    res: 4=480p, 7=720p, 12=1080p. Older HERO8 firmware used /gp/gpWebcam/START instead.
    """
    urls = [
        f"http://{ip}:8080/gopro/webcam/start?res={res}&fov={fov}",
        f"http://{ip}/gp/gpWebcam/START?res={res}",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                if r.status == 200:
                    log.info("GoPro webcam started via %s", url)
                    return True
        except OSError as e:
            log.debug("GoPro start via %s failed: %s", url, e)
    return False


def gopro_stop_webcam(ip: str) -> None:
    for url in (f"http://{ip}:8080/gopro/webcam/stop", f"http://{ip}/gp/gpWebcam/STOP"):
        try:
            urllib.request.urlopen(url, timeout=3).close()
            return
        except OSError:
            pass


# ---------------------------------------------------------------- synthetic frames


class SyntheticGenerator:
    """Moving shapes over a gradient 'terrain' — used when no sample clip is available
    (offline first run) and by the tests. YOLO will not detect anything in it."""

    def __init__(self, width: int = 960, height: int = 540):
        self.w, self.h = width, height
        self.t0 = time.time()
        ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        base = np.zeros((height, width, 3), np.uint8)
        base[..., 0] = (40 + 30 * ys).astype(np.uint8)
        base[..., 1] = (90 + 50 * ys).astype(np.uint8)
        base[..., 2] = (60 + 20 * ys).astype(np.uint8)
        self.base = base

    def frame(self) -> np.ndarray:
        t = time.time() - self.t0
        img = self.base.copy()
        for i in range(3):
            x = int((self.w * 0.15) + (self.w * 0.7) * (0.5 + 0.5 * np.sin(t * 0.4 + i * 2.1)))
            y = int(self.h * (0.3 + 0.2 * i))
            cv2.rectangle(img, (x - 30, y - 15), (x + 30, y + 15), (30, 30, 160 + 30 * i), -1)
        cv2.putText(img, "SYNTHETIC TEST PATTERN - no sample clip", (20, self.h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------- the source


class VideoSource:
    """Threaded capture holding only the latest frame."""

    def __init__(self, spec: str, *, width: int = 1280, height: int = 720, sample_path: Path | None = None,
                 loop_seconds: float = 0.0):
        self.spec = parse_spec(spec)
        self.loop_seconds = loop_seconds  # files: rewind after this many seconds (0 = whole file)
        self.width, self.height = width, height
        self.sample_path = sample_path
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.error = ""
        self.fps_in = 0.0
        self.resolved = ""  # what we actually opened
        self._gopro_ip: str | None = None

    # -- public API
    def start(self) -> "VideoSource":
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._gopro_ip:
            gopro_stop_webcam(self._gopro_ip)

    def latest(self) -> tuple[int, np.ndarray | None, float]:
        with self._lock:
            return self._frame_id, self._frame, self._frame_ts

    def status(self) -> dict:
        fid, frame, _ = self.latest()
        h, w = (frame.shape[:2] if frame is not None else (0, 0))
        return {
            "spec": self.spec.raw,
            "kind": self.spec.kind,
            "resolved": self.resolved,
            "connected": self.connected,
            "error": self.error,
            "width": w,
            "height": h,
            "fps_in": round(self.fps_in, 1),
            "frames": fid,
        }

    # -- internals
    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            self._frame_ts = time.time()

    def _open(self) -> tuple[cv2.VideoCapture | None, bool]:
        """Return (capture, loop_at_native_fps)."""
        kind, target = self.spec.kind, self.spec.target
        if kind == "simulate":
            if self.sample_path and self.sample_path.exists():
                self.resolved = str(self.sample_path)
                return cv2.VideoCapture(str(self.sample_path)), True
            self.resolved = "synthetic"
            return None, True
        if kind == "synthetic":
            self.resolved = "synthetic"
            return None, True
        if kind == "file":
            path = os.path.expanduser(target)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            self.resolved = path
            return cv2.VideoCapture(path), True
        if kind == "uvc":
            dev: int | str = int(target) if target.isdigit() else target
            backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
            cap = cv2.VideoCapture(dev, backend)
            # Most cheap HDMI/AV dongles only reach 30fps at 720p+ with MJPEG.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.resolved = str(dev)
            return cap, False
        if kind == "network":
            if target.lower().startswith("rtsp"):
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.resolved = target
            return cap, False
        if kind == "gopro-usb":
            ip = target or find_gopro_ip()
            if not ip:
                raise RuntimeError("no GoPro USB network interface (172.2x.1xx.x) found — is webcam mode on?")
            if not gopro_start_webcam(ip):
                raise RuntimeError(f"GoPro at {ip} did not accept the webcam start command")
            self._gopro_ip = ip
            self.resolved = f"{ip} -> {GOPRO_UDP_URL}"
            return cv2.VideoCapture(GOPRO_UDP_URL, cv2.CAP_FFMPEG), False
        raise ValueError(f"unknown source kind {kind}")

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                cap, paced = self._open()
            except Exception as e:  # noqa: BLE001 - keep retrying, show the error in the UI
                self.connected, self.error = False, str(e)
                log.warning("source %s: %s (retry in %.0fs)", self.spec.raw, e, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15)
                continue

            if cap is None:  # synthetic
                self._run_synthetic()
                return

            if not cap.isOpened():
                self.connected, self.error = False, f"could not open {self.resolved}"
                cap.release()
                log.warning("source %s: could not open (retry in %.0fs)", self.spec.raw, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15)
                continue

            self.connected, self.error = True, ""
            backoff = 1.0
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if not 1 <= native_fps <= 120:
                native_fps = 25.0
            period = 1.0 / native_fps
            log.info("source %s opened (%s, %.0f fps)", self.spec.raw, self.resolved, native_fps)
            n, t_window = 0, time.time()
            next_t = time.time()
            fails = 0
            loop_frames = int(self.loop_seconds * native_fps) if paced and self.loop_seconds else 0
            pos = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                pos += 1
                if loop_frames and pos > loop_frames:
                    ok, pos = False, 0
                if not ok or frame is None:
                    if paced:  # file / sample: loop
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        pos = 1
                        if not ok:
                            self.error = "file unreadable"
                            break
                    else:
                        fails += 1
                        if fails > 25:
                            self.error = "stream lost"
                            break
                        time.sleep(0.02)
                        continue
                fails = 0
                self._publish(frame)
                n += 1
                now = time.time()
                if now - t_window >= 2.0:
                    self.fps_in = n / (now - t_window)
                    n, t_window = 0, now
                if paced:
                    next_t += period
                    delay = next_t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.time()
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                log.warning("source %s: %s — reconnecting", self.spec.raw, self.error)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15)

    def _run_synthetic(self) -> None:
        gen = SyntheticGenerator()
        self.connected, self.error = True, ""
        self.fps_in = 25.0
        while not self._stop.is_set():
            self._publish(gen.frame())
            time.sleep(1 / 25)
