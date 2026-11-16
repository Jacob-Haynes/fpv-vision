"""Pluggable video sources.

Source spec strings (config ``FPV_SOURCE``, ``--source`` or the web UI):

  replay:/path/a.mp4[|/path/b.mp4|/dir]   footage replayed AS LIVE: wall-clock paced, frames
                                          dropped (never slowed) if we fall behind, playlist loops
  file:/path/clip.mp4 | /path/clip.mp4    same as replay with one file
  simulate | simulate:aerial              bundled public-domain sample clip, replayed (synthetic
                                          test pattern if it cannot be fetched)
  camera:<name substring>                 macOS/Windows camera by NAME (e.g. "camera:iPhone" for
                                          Continuity Camera, "camera:USB Video" for a capture card)
  uvc:0 | uvc:/dev/video0 | 0             camera by OpenCV index / Linux V4L2 node
  screen[:N]                              macOS/Linux screen capture of monitor N (default 1);
                                          use the crop control to cut it down to a mirrored phone
  rtsp://... rtmp://... http(s)://... udp://... srt://...   network streams (via FFmpeg)
  gopro-usb | gopro-usb:172.2X.1YZ.51     GoPro HERO9+ USB webcam mode (untested)

Every source runs its own capture thread and keeps only the *latest* frame, so a slow
consumer never builds up latency. Live sources reconnect with backoff.
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
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".ts", ".mts"}
REPLAY_KINDS = ("replay", "simulate")


@dataclass
class SourceSpec:
    kind: str  # replay | simulate | synthetic | camera | uvc | screen | network | gopro-usb
    target: str
    raw: str

    @property
    def is_live(self) -> bool:
        return self.kind in ("camera", "uvc", "screen", "network", "gopro-usb")


def parse_spec(spec: str) -> SourceSpec:
    raw = (spec or "simulate").strip()
    low = raw.lower()
    if low in ("simulate", "sim", "sample") or low.startswith("simulate:"):
        return SourceSpec("simulate", raw.split(":", 1)[1] if ":" in raw else "", raw)
    if low == "synthetic":
        return SourceSpec("synthetic", "", raw)
    if low.startswith("replay:"):
        return SourceSpec("replay", raw[7:], raw)
    if low.startswith("file:"):
        return SourceSpec("replay", raw[5:], raw)
    if low.startswith("camera:"):
        return SourceSpec("camera", raw[7:], raw)
    if low == "screen" or low.startswith("screen:"):
        return SourceSpec("screen", raw[7:] if ":" in raw else "1", raw)
    if low.startswith("uvc:") or low.startswith("/dev/video"):
        target = raw[4:] if low.startswith("uvc:") else raw
        return SourceSpec("uvc", target or "0", raw)
    if low.isdigit():
        return SourceSpec("uvc", raw, raw)
    if low.startswith("gopro-usb"):
        return SourceSpec("gopro-usb", raw.split(":", 1)[1] if ":" in raw else "", raw)
    if low.startswith(NETWORK_SCHEMES):
        return SourceSpec("network", raw, raw)
    return SourceSpec("replay", raw, raw)


def expand_playlist(target: str) -> list[str]:
    """'a.mp4|b.mp4|/dir' -> existing video files (dirs expanded, sorted)."""
    out: list[str] = []
    for part in [p.strip() for p in target.split("|") if p.strip()]:
        p = Path(os.path.expanduser(part))
        if p.is_dir():
            out += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS and not f.name.startswith("."))
        elif p.exists():
            out.append(str(p))
    return out


# ---------------------------------------------------------------- device listing


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


def list_cameras() -> list[dict]:
    """Cameras with names and the index OpenCV will use for them.

    macOS: same enumeration as OpenCV's AVFoundation backend (video + muxed devices sorted by
    uniqueID), so names line up with indexes. Includes Continuity Camera iPhones and USB
    capture cards. Linux: V4L2 nodes with their driver names.
    """
    cams: list[dict] = []
    if sys.platform == "darwin":
        try:
            import AVFoundation as AVF  # pyobjc-framework-AVFoundation

            devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)) + \
                list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeMuxed))
            devs.sort(key=lambda d: str(d.uniqueID()))
            for i, d in enumerate(devs):
                cams.append({"index": i, "name": str(d.localizedName()), "id": str(d.uniqueID())})
        except Exception as e:  # noqa: BLE001
            log.debug("AVFoundation listing unavailable: %s", e)
    elif sys.platform.startswith("linux"):
        for d in list_uvc_devices():
            name_file = Path(f"/sys/class/video4linux/{Path(d).name}/name")
            cams.append({"index": d, "name": name_file.read_text().strip() if name_file.exists() else d, "id": d})
    return cams


def camera_permission_status() -> str:
    """macOS camera (TCC) status for this process: authorized | denied | restricted | not_determined | n/a."""
    if sys.platform != "darwin":
        return "n/a"
    try:
        import AVFoundation as AVF

        st = int(AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo))
        return {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(st, str(st))
    except Exception:  # noqa: BLE001
        return "unknown"


def request_camera_permission(timeout: float = 30.0) -> str:
    """Ask macOS for camera access. MUST run on the main thread: OpenCV cannot raise the prompt
    from its capture thread ("can not spin main run loop from other thread")."""
    status = camera_permission_status()
    if status != "not_determined":
        return status
    try:
        import AVFoundation as AVF
        from Foundation import NSDate, NSRunLoop

        done = {"v": None}
        AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVF.AVMediaTypeVideo, lambda granted: done.__setitem__("v", bool(granted)))
        end = time.time() + timeout
        while done["v"] is None and time.time() < end:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
    except Exception as e:  # noqa: BLE001
        log.warning("camera permission request failed: %s", e)
    return camera_permission_status()


def resolve_camera(name: str) -> int | str | None:
    want = name.strip().lower()
    for c in list_cameras():
        if want in c["name"].lower():
            return c["index"]
    return int(want) if want.isdigit() else None


def camera_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


# ---------------------------------------------------------------- GoPro USB webcam


def find_gopro_ip() -> str | None:
    """A GoPro in USB webcam mode appears as a USB network interface on 172.2X.1YZ.0/24 (.51 = camera)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for m in re.finditer(r"inet (172\.2\d\.1\d\d)\.\d+/", out):
        return f"{m.group(1)}.51"
    return None


def gopro_start_webcam(ip: str, res: int = 12, fov: int = 0) -> bool:
    for url in (f"http://{ip}:8080/gopro/webcam/start?res={res}&fov={fov}", f"http://{ip}/gp/gpWebcam/START?res={res}"):
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                if r.status == 200:
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
    """Moving shapes over a gradient — used when no sample clip is available and by tests."""

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
    """Threaded capture holding only the latest frame (plus its capture timestamp)."""

    def __init__(self, spec: str, *, width: int = 1920, height: int = 1080, sample_path: Path | None = None,
                 loop_seconds: float = 0.0, fps: int = 60):
        self.spec = parse_spec(spec)
        self.width, self.height, self.req_fps = width, height, fps
        self.negotiated = ""  # what the camera/capture card actually agreed to, e.g. "1920x1080@60 MJPG"
        self.sample_path = sample_path
        self.loop_seconds = loop_seconds  # replay: rewind after this many seconds (0 = whole file)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.error = ""
        self.fps_in = 0.0
        self.resolved = ""
        self._gopro_ip: str | None = None
        self.crop: tuple[float, float, float, float] | None = None  # normalised x, y, w, h
        # replay state
        self.playlist: list[str] = []
        self.replay = {"index": 0, "file": "", "pos": 0.0, "duration": 0.0, "fps": 0.0, "dropped": 0,
                       "paused": False, "width": 0, "height": 0}
        self._seek_req: float | None = None
        self._skip_req = 0
        self._paused = False

    @property
    def is_replay(self) -> bool:
        return self.spec.kind in REPLAY_KINDS and self.resolved != "synthetic"

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

    def set_crop(self, crop: tuple[float, float, float, float] | None) -> None:
        if crop is not None:
            x, y, w, h = (min(max(float(v), 0.0), 1.0) for v in crop)
            if w < 0.02 or h < 0.02:
                crop = None
            else:
                crop = (x, y, min(w, 1 - x), min(h, 1 - y))
        self.crop = crop

    # replay controls (ignored for live sources)
    def seek(self, seconds: float) -> None:
        self._seek_req = max(0.0, float(seconds))

    def next_file(self, step: int = 1) -> None:
        self._skip_req = step

    def pause(self, paused: bool) -> None:
        self._paused = bool(paused)

    def status(self) -> dict:
        fid, frame, _ = self.latest()
        h, w = (frame.shape[:2] if frame is not None else (0, 0))
        st = {
            "spec": self.spec.raw,
            "kind": self.spec.kind,
            "live": self.spec.is_live,
            "replay": self.is_replay,
            "resolved": self.resolved,
            "connected": self.connected,
            "error": self.error,
            "width": w,
            "height": h,
            "fps_in": round(self.fps_in, 1),
            "frames": fid,
            "crop": self.crop,
            "negotiated": self.negotiated,
        }
        if self.is_replay:
            st["replay_state"] = {**self.replay, "files": [Path(p).name for p in self.playlist],
                                  "pos": round(self.replay["pos"], 1), "duration": round(self.replay["duration"], 1)}
        return st

    # -- internals
    def _publish(self, frame: np.ndarray, ts: float | None = None) -> None:
        if self.crop:
            h, w = frame.shape[:2]
            x, y, cw, ch = self.crop
            x0, y0 = int(x * w), int(y * h)
            frame = frame[y0:y0 + max(2, int(ch * h)), x0:x0 + max(2, int(cw * w))]
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            self._frame_ts = ts or time.time()

    def _wait_or_stop(self, seconds: float) -> bool:
        return self._stop.wait(seconds)

    def _run(self) -> None:
        kind = self.spec.kind
        try:
            if kind in ("synthetic",) or (kind == "simulate" and not (self.sample_path and self.sample_path.exists())):
                self.resolved = "synthetic"
                return self._run_synthetic()
            if kind in REPLAY_KINDS:
                return self._run_replay()
            if kind == "screen":
                return self._run_screen()
            return self._run_live()
        except Exception as e:  # noqa: BLE001 - never let the capture thread die silently
            log.exception("source %s crashed: %s", self.spec.raw, e)
            self.connected, self.error = False, f"crashed: {e}"

    # ---- replay-as-live
    def _run_replay(self) -> None:
        if self.spec.kind == "simulate":
            self.playlist = [str(self.sample_path)]
        else:
            self.playlist = expand_playlist(self.spec.target)
        if not self.playlist:
            self.connected, self.error = False, f"no video files found: {self.spec.target}"
            log.warning("source %s: %s", self.spec.raw, self.error)
            return
        self.resolved = self.playlist[0] if len(self.playlist) == 1 else f"{len(self.playlist)} files"
        idx = 0
        fails = 0
        while not self._stop.is_set():
            path = self.playlist[idx % len(self.playlist)]
            ok = self._play_file(path, idx % len(self.playlist))
            fails = 0 if ok else fails + 1
            if fails >= len(self.playlist):
                self.connected, self.error = False, "no playable files in playlist"
                if self._wait_or_stop(5):
                    return
                fails = 0
            step = self._skip_req or 1
            self._skip_req = 0
            idx = (idx + step) % len(self.playlist)

    def _play_file(self, path: str, index: int) -> bool:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.error = f"cannot open {Path(path).name}"
            log.warning("replay: %s", self.error)
            cap.release()
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not 1 <= fps <= 240:
            fps = 30.0
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = nframes / fps if nframes > 0 else 0.0
        if self.loop_seconds and (duration == 0 or duration > self.loop_seconds):
            duration = self.loop_seconds
        self.replay.update(index=index, file=Path(path).name, duration=duration, fps=round(fps, 2), pos=0.0,
                           width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.connected, self.error = True, ""
        log.info("replay %s (%.0f fps, %.0fs)", path, fps, duration)

        frame_idx = 0  # index of the next frame read() will return
        wall0 = time.time()
        n, t_window = 0, time.time()
        while not self._stop.is_set():
            if self._skip_req:
                break
            if self._seek_req is not None:
                target = min(self._seek_req, max(0.0, duration - 0.5)) if duration else self._seek_req
                self._seek_req = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target * fps))
                frame_idx = int(target * fps)
                wall0 = time.time() - target
            if self._paused:
                self.replay["paused"] = True
                time.sleep(0.05)
                wall0 += 0.05
                continue
            self.replay["paused"] = False

            due = int((time.time() - wall0) * fps)  # frame that should be on screen now
            behind = due - frame_idx
            if behind > fps * 2:  # badly behind (slow decode): jump ahead instead of crawling
                cap.set(cv2.CAP_PROP_POS_FRAMES, due)
                self.replay["dropped"] += behind
                frame_idx = due
            else:
                while behind > 0:  # drop frames: grab() skips colour conversion/copy
                    if not cap.grab():
                        break
                    frame_idx += 1
                    behind -= 1
                    self.replay["dropped"] += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break  # end of file -> next in playlist (loops)
            ts = time.time()
            frame_idx += 1
            pos = frame_idx / fps
            self.replay["pos"] = pos
            if self.loop_seconds and pos > self.loop_seconds:
                break
            self._publish(frame, ts)
            n += 1
            if ts - t_window >= 2.0:
                self.fps_in = n / (ts - t_window)
                n, t_window = 0, ts
            delay = wall0 + frame_idx / fps - time.time()
            if delay > 0:
                time.sleep(min(delay, 0.5))
        cap.release()
        return True

    # ---- screen capture
    def _run_screen(self) -> None:
        try:
            import mss
        except ImportError:
            self.connected, self.error = False, "screen capture needs the 'mss' package"
            return
        mon = int(self.spec.target or 1) if (self.spec.target or "1").isdigit() else 1
        factory = getattr(mss, "MSS", None) or mss.mss
        with factory() as sct:
            if mon >= len(sct.monitors):
                mon = 1
            region = sct.monitors[mon]
            self.resolved = f"monitor {mon} {region['width']}x{region['height']}"
            period = 1 / 30
            n, t_window = 0, time.time()
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    img = np.asarray(sct.grab(region))
                except Exception as e:  # noqa: BLE001
                    self.connected, self.error = False, f"screen grab failed: {e} (grant Screen Recording permission)"
                    if self._wait_or_stop(2):
                        return
                    continue
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if not self.connected and frame.mean() < 1.0:
                    # macOS returns black frames when Screen Recording permission is missing
                    self.error = "screen is all black: grant Screen Recording permission to your terminal app"
                else:
                    self.error = ""
                self.connected = True
                self._publish(frame, t0)
                n += 1
                if t0 - t_window >= 2.0:
                    self.fps_in = n / (t0 - t_window)
                    n, t_window = 0, t0
                rest = period - (time.time() - t0)
                if rest > 0:
                    time.sleep(rest)

    # ---- cameras / network / GoPro
    def _open_live(self) -> cv2.VideoCapture:
        kind, target = self.spec.kind, self.spec.target
        if kind in ("camera", "uvc"):
            if kind == "camera":
                dev = resolve_camera(target)
                if dev is None:
                    raise RuntimeError(f"no camera matching '{target}' (check it is connected / unlocked)")
            else:
                dev = int(target) if target.isdigit() else target
            cap = cv2.VideoCapture(dev, camera_backend())
            if sys.platform != "darwin":
                # UVC capture cards (e.g. MS2109/MS2130 HDMI->USB) only reach 1080p30/60 as MJPEG;
                # raw YUY2 at 1080p needs more USB bandwidth and more CPU. On macOS AVFoundation
                # picks the device format itself (and decodes MJPEG in hardware), so we don't force it.
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.req_fps:
                cap.set(cv2.CAP_PROP_FPS, self.req_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.resolved = f"{target} -> device {dev}" if kind == "camera" else str(dev)
            return cap
        if kind == "network":
            if target.lower().startswith("rtsp"):
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.resolved = target
            return cap
        if kind == "gopro-usb":
            ip = target or find_gopro_ip()
            if not ip:
                raise RuntimeError("no GoPro USB network interface (172.2x.1xx.x) found — is webcam mode on?")
            if not gopro_start_webcam(ip):
                raise RuntimeError(f"GoPro at {ip} did not accept the webcam start command")
            self._gopro_ip = ip
            self.resolved = f"{ip} -> {GOPRO_UDP_URL}"
            return cv2.VideoCapture(GOPRO_UDP_URL, cv2.CAP_FFMPEG)
        raise ValueError(f"unknown source kind {kind}")

    def _run_live(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                cap = self._open_live()
            except Exception as e:  # noqa: BLE001 - keep retrying, show the error in the UI
                self.connected, self.error = False, str(e)
                log.warning("source %s: %s (retry in %.0fs)", self.spec.raw, e, backoff)
                if self._wait_or_stop(backoff):
                    return
                backoff = min(backoff * 2, 15)
                continue
            if not cap.isOpened():
                self.connected = False
                self.error = f"could not open {self.resolved}" + (
                    " (macOS: allow Camera access for your terminal app)" if sys.platform == "darwin" else "")
                cap.release()
                if self._wait_or_stop(backoff):
                    return
                backoff = min(backoff * 2, 15)
                continue
            self.connected, self.error = True, ""
            backoff = 1.0
            self.negotiated = self._describe_mode(cap)
            log.info("source %s opened (%s) %s", self.spec.raw, self.resolved, self.negotiated)
            n, t_window, fails = 0, time.time(), 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 50:
                        self.error = "stream lost"
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                ts = time.time()
                self._publish(frame, ts)
                n += 1
                if ts - t_window >= 2.0:
                    self.fps_in = n / (ts - t_window)
                    n, t_window = 0, ts
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                log.warning("source %s: %s — reconnecting", self.spec.raw, self.error)
                if self._wait_or_stop(backoff):
                    return
                backoff = min(backoff * 2, 15)

    @staticmethod
    def _describe_mode(cap: cv2.VideoCapture) -> str:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") if code else ""
        fourcc = fourcc if fourcc.isprintable() else ""
        return f"{w}x{h}" + (f"@{fps:.0f}" if fps else "") + (f" {fourcc}" if fourcc else "")

    def _run_synthetic(self) -> None:
        gen = SyntheticGenerator()
        self.connected, self.error = True, ""
        self.fps_in = 25.0
        while not self._stop.is_set():
            self._publish(gen.frame())
            time.sleep(1 / 25)
