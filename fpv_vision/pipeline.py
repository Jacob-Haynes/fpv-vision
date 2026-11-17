"""Capture -> detect/track -> annotate -> MJPEG, plus snapshots and clip recording.

Two worker threads keep the overlay smooth on a CPU-only Pi:

* the **detect** thread always works on the newest captured frame and skips whatever
  arrived while it was busy (adaptive frame-skip). If the render loop starts missing its
  target rate, the detector backs off with a small gap between inferences to free CPU;
  when there is headroom the gap shrinks back to zero.
* the **render** thread draws the latest boxes on every new frame at up to
  ``target_fps`` and encodes the JPEG served to browsers.

Boxes can therefore lag the video by one inference (~100-250 ms on a Pi 5).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from fpv_vision import annotate, summary
from fpv_vision.detector import Detection, Detector
from fpv_vision.sources import VideoSource

log = logging.getLogger(__name__)

MIN_TRACK_HITS = 3  # a track must be seen this many times before it is counted


RARE_CLASSES = {"bus", "train"}  # easily confused with a van/lorry from the air or through glass
RARE_MIN_CONF = 0.6  # below this voted confidence, a bus/train is reported as a truck
FOLD_TO = (7, "truck")


def _fold(label: str, cls_id: int, conf: float) -> tuple[int, str]:
    if label in RARE_CLASSES and conf < RARE_MIN_CONF:
        return FOLD_TO
    return cls_id, label


class TrackCounter:
    """Unique-object counts using tracker IDs, so one person walking through is counted once.

    Each track gets ONE class for its whole life: a confidence-weighted majority vote over every
    detection of it (a white van flickering truck/bus/train stays one truck). Bus and train are only
    kept when their voted mean confidence is at least ``RARE_MIN_CONF``; otherwise they fold into
    truck. ``totals`` counts unique track IDs by voted class, so class flips never add to "seen".
    """

    def __init__(self, min_hits: int = MIN_TRACK_HITS):
        self.min_hits = min_hits
        self._hits: dict[int, int] = defaultdict(int)
        self._votes: dict[int, dict[str, list]] = {}  # track -> label -> [conf sum, n, cls_id]
        self._counted: set[int] = set()
        self.untracked_peak: dict[str, int] = defaultdict(int)

    def voted(self, track_id: int) -> tuple[int, str, float]:
        """(cls_id, label, mean confidence) of a track's winning class, after the bus/train fold."""
        votes = self._votes[track_id]
        label, (total, n, cls_id) = max(list(votes.items()), key=lambda kv: kv[1][0])
        mean = total / n
        cls_id, label = _fold(label, cls_id, mean)
        return cls_id, label, mean

    def update(self, dets: list[Detection]) -> list[Detection]:
        """Record this frame's detections; return them relabelled with each track's voted class."""
        untracked: dict[str, int] = defaultdict(int)
        out = []
        for d in dets:
            if d.track_id is None:
                cls_id, label = _fold(d.label, d.cls_id, d.conf)
                untracked[label] += 1
                out.append(Detection(cls_id, label, d.conf, d.xyxy, None) if label != d.label else d)
                continue
            self._hits[d.track_id] += 1
            v = self._votes.setdefault(d.track_id, {}).setdefault(d.label, [0.0, 0, d.cls_id])
            v[0] += d.conf
            v[1] += 1
            if d.track_id not in self._counted and self._hits[d.track_id] >= self.min_hits:
                self._counted.add(d.track_id)
            cls_id, label, _ = self.voted(d.track_id)
            out.append(d if label == d.label else Detection(cls_id, label, d.conf, d.xyxy, d.track_id))
        for k, n in untracked.items():
            self.untracked_peak[k] = max(self.untracked_peak[k], n)
        return out

    @property
    def totals(self) -> dict[str, int]:
        t: dict[str, int] = defaultdict(int)
        for tid in list(self._counted):  # read from the web thread while detect updates
            t[self.voted(tid)[1]] += 1
        return dict(t)

    def reset(self) -> None:
        self.__init__(self.min_hits)

    def snapshot(self) -> dict[str, int]:
        return self.totals


@dataclass
class PipelineStats:
    render_fps: float = 0.0
    detect_fps: float = 0.0
    detect_ms: float = 0.0
    detect_gap_ms: float = 0.0
    skip_ratio: float = 0.0  # captured frames per detection
    # latency (ms, exponential moving averages)
    pipeline_ms: float = 0.0  # frame captured/decoded -> annotated JPEG ready to serve
    box_lag_ms: float = 0.0  # age of the boxes drawn on a frame relative to that frame
    encode_ms: float = 0.0
    extra: dict = field(default_factory=dict)


def _ema(old: float, new: float, a: float = 0.15) -> float:
    return round(new if not old else old + a * (new - old), 1)


class Recorder:
    def __init__(self, clips_dir: Path, max_seconds: int = 300):
        self.clips_dir = clips_dir
        self.max_seconds = max_seconds
        self._writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.started = 0.0
        self.fps = 10.0
        self.size: tuple[int, int] | None = None
        self._frames = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.path is not None

    def start(self, fps: float) -> Path:
        with self._lock:
            if self._writer is not None and self.path:
                return self.path
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.clips_dir / time.strftime("clip_%Y%m%d_%H%M%SZ.mp4", time.gmtime())
            self.fps = max(5.0, min(30.0, round(fps or 10.0)))
            self.started = time.time()
            self._frames = 0
            self.size = None
            return self.path

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self.path is None:
                return
            if self._writer is None:
                h, w = frame.shape[:2]
                self.size = (w, h)
                # mp4v plays in VLC/most players; browsers may need download-then-play.
                self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size)
            # keep real-time duration: write as many frames as wall-clock says we owe
            owed = int((time.time() - self.started) * self.fps) + 1 - self._frames
            for _ in range(max(0, min(owed, 5))):
                self._writer.write(frame)
                self._frames += 1
        if time.time() - self.started > self.max_seconds:
            self.stop()

    def stop(self) -> Path | None:
        with self._lock:
            p = self.path
            if self._writer is not None:
                self._writer.release()
            self._writer, self.path = None, None
            return p


class Pipeline:
    def __init__(self, settings, detector: Detector, source: VideoSource):
        self.settings = settings
        self.detector = detector
        self.source = source
        self.counter = TrackCounter()
        self.running = True  # False = Stop pressed: no detection, stream holds the last frame
        self.stats = PipelineStats()
        self.recorder = Recorder(settings.clips_dir, settings.max_clip_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._dets: list[Detection] = []
        self._dets_frame_shape: tuple[int, int] = (0, 0)
        self._dets_ts = 0.0
        self._jpeg = b""
        self._jpeg_seq = 0
        self._raw_frame: np.ndarray | None = None
        self._annotated: np.ndarray | None = None
        self._threads: list[threading.Thread] = []
        self.hud_mode = "LOCAL"

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "Pipeline":
        self.source.start()
        for target, name in ((self._detect_loop, "detect"), (self._render_loop, "render")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()
        self.recorder.stop()
        self.source.stop()
        for t in self._threads:
            t.join(timeout=3)

    def switch_source(self, source: VideoSource) -> None:
        old = self.source
        self.source = source
        source.start()
        old.stop()
        with self._lock:
            self._dets = []
        self.detector.reset_tracks()
        log.info("switched source to %s", source.spec.raw)

    # ------------------------------------------------------------ accessors
    def latest_jpeg(self) -> tuple[int, bytes]:
        with self._lock:
            return self._jpeg_seq, self._jpeg

    def latest_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            raw = None if self._raw_frame is None else self._raw_frame.copy()
            ann = None if self._annotated is None else self._annotated.copy()
            return raw, ann

    def current_detections(self) -> tuple[list[Detection], tuple[int, int]]:
        with self._lock:
            return list(self._dets), self._dets_frame_shape

    def current_counts(self) -> dict[str, int]:
        dets, _ = self.current_detections()
        return summary.count_by_label(dets)

    # ------------------------------------------------------------ workers
    def _detect_loop(self) -> None:
        last_id = -1
        gap = 0.0
        n, t_window, frames_at_window = 0, time.time(), 0
        while not self._stop.is_set():
            if not self.running:
                time.sleep(0.05)
                continue
            src = self.source
            fid, frame, fts = src.latest()
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            t0 = time.perf_counter()
            try:
                dets = self.detector.detect(frame)
            except Exception as e:  # noqa: BLE001
                log.exception("detector error: %s", e)
                dets = []
                time.sleep(0.5)
            dt = time.perf_counter() - t0
            if src is not self.source:  # source switched mid-inference
                continue
            dets = self.counter.update(dets)  # one voted class per track
            with self._lock:
                self._dets = dets
                self._dets_frame_shape = frame.shape[:2]
                self._dets_ts = fts  # capture time of the frame these boxes belong to
            self.stats.detect_ms = round(dt * 1000, 1)
            n += 1
            now = time.time()
            if now - t_window >= 2.0:
                self.stats.detect_fps = round(n / (now - t_window), 1)
                captured = src.status()["frames"] - frames_at_window
                self.stats.skip_ratio = round(captured / n, 1) if n else 0.0
                frames_at_window = src.status()["frames"]
                n, t_window = 0, now
                # adaptive back-off: protect the overlay frame rate first
                target = self.settings.target_fps
                src_fps = (src.fps_unique if src.spec.is_live else src.fps_in) or target
                if self.stats.render_fps and self.stats.render_fps < 0.85 * min(target, src_fps):
                    gap = min(1.0, gap + 0.02)
                elif gap > 0:
                    gap = max(0.0, gap - 0.02)
                self.stats.detect_gap_ms = round(gap * 1000)
            if self.detector.name == "none":
                time.sleep(0.2)
            elif gap:
                self._stop.wait(gap)

    def _render_loop(self) -> None:
        last_id = -1
        n, t_window = 0, time.time()
        next_t = time.time()
        while not self._stop.is_set():
            period = 1.0 / max(1.0, self.settings.target_fps)
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            if not self.running:
                time.sleep(0.05)
                continue
            src = self.source
            fid, frame, fts = src.latest()
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            with self._lock:
                dets = list(self._dets)
                dshape = self._dets_frame_shape
                dets_ts = self._dets_ts
            out = annotate.resize_to_width(frame, self.settings.stream_width)
            if dshape[0] and dshape[1]:
                dets = annotate.scale_detections(dets, out.shape[1] / dshape[1], out.shape[0] / dshape[0])
            out = out.copy() if out is frame else out
            counts = summary.count_by_label(dets)
            accel = getattr(self.detector, "accelerator", "cpu").upper()
            hud = [
                f"FPV-VISION  {self.hud_mode}  {self.detector.backend.upper()}/{accel}",
                f"video {self.stats.render_fps:.0f}fps  det {self.stats.detect_fps:.1f}fps  "
                f"lat {self.stats.pipeline_ms + self.stats.box_lag_ms:.0f}ms",
            ]
            if counts:
                hud.append("  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
            annotate.draw(out, dets, hud if self.settings.show_hud else None, recording=self.recorder.active,
                          badge=self._badge(src))
            t_enc = time.time()
            jpeg = annotate.encode_jpeg(out, self.settings.jpeg_quality)
            done = time.time()
            self.stats.encode_ms = _ema(self.stats.encode_ms, (done - t_enc) * 1000)
            if fts:
                self.stats.pipeline_ms = _ema(self.stats.pipeline_ms, (done - fts) * 1000)
            if dets_ts and fts and dets:
                self.stats.box_lag_ms = _ema(self.stats.box_lag_ms, max(0.0, fts - dets_ts) * 1000)
            if self.recorder.active:
                self.recorder.write(out)
            with self._lock:
                self._raw_frame = frame
                self._annotated = out
                self._jpeg = jpeg
                self._jpeg_seq += 1
            n += 1
            now = time.time()
            if now - t_window >= 2.0:
                self.stats.render_fps = round(n / (now - t_window), 1)
                n, t_window = 0, now
            # deadline-based pacing: hold target_fps without accumulating drift
            next_t = max(next_t + period, time.time() - period)

    @staticmethod
    def _badge(src) -> str:
        """Discreet but honest provenance label, drawn top-right on every frame."""
        kind = src.spec.kind
        if kind == "simulate" or src.resolved == "synthetic":
            return "SIMULATED"
        if kind == "replay":
            r = src.replay
            pos = int(r.get("pos", 0))
            return f"REPLAY {r.get('file', '')[:28]} {pos // 60:02d}:{pos % 60:02d}"
        if kind == "screen":
            return "SCREEN MIRROR"
        return "LIVE"

    # ------------------------------------------------------------ snapshots
    def take_snapshot(self, note: str = "") -> dict | None:
        raw, ann = self.latest_frames()
        if raw is None or ann is None:
            return None
        dets, shape = self.current_detections()
        extra = {}
        if self.source.is_replay:
            extra = {"video_file": self.source.replay.get("file"), "video_time": round(self.source.replay.get("pos", 0), 1)}
        kind = self.source.spec.kind
        provenance = "simulated" if (kind in ("simulate", "synthetic") or self.source.resolved == "synthetic") else \
            "recorded" if kind == "replay" else "live"
        return self.save_snapshot(raw, ann, dets, shape[1] or raw.shape[1], note=note,
                                  source=self.source.spec.raw, extra={**extra, "provenance": provenance})

    def save_snapshot(self, raw: np.ndarray, ann: np.ndarray, dets: list[Detection], frame_width: int, *,
                      note: str = "", source: str = "", extra: dict | None = None, prefix: str = "snap") -> dict:
        snap_dir = self.settings.snapshots_dir
        snap_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime()) + f"_{int(time.time() * 1000) % 1000:03d}"
        base = f"snap_{stamp}" if prefix == "snap" else f"snap_{prefix}_{stamp}"
        n = 1
        while (snap_dir / f"{base}.json").exists():
            n += 1
            base = f"{base.rsplit('-', 1)[0]}-{n}"
        cv2.imwrite(str(snap_dir / f"{base}.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(snap_dir / f"{base}_raw.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 90])
        meta = {
            "name": base,
            "ts": time.time(),
            "image": f"{base}.jpg",
            "raw_image": f"{base}_raw.jpg",
            "source": source,
            "detections": [d.to_dict() for d in dets],
            "counts": summary.count_by_label(dets),
            "local_summary": summary.describe(dets, frame_width),
            "summary": None,
            "summary_mode": None,
            "note": note,
            **(extra or {}),
        }
        self.save_snapshot_meta(meta)
        return meta

    def save_snapshot_meta(self, meta: dict) -> None:
        path = self.settings.snapshots_dir / f"{meta['name']}.json"
        path.write_text(json.dumps(meta, indent=2))

    def load_snapshot(self, name: str) -> dict | None:
        if not name or "/" in name or ".." in name:
            return None
        path = self.settings.snapshots_dir / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_snapshots(self, limit: int = 60) -> list[dict]:
        d = self.settings.snapshots_dir
        if not d.exists():
            return []
        items = []
        for p in sorted(d.glob("snap_*.json"), reverse=True)[:limit]:
            try:
                items.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return items

    def list_clips(self) -> list[dict]:
        d = self.settings.clips_dir
        if not d.exists():
            return []
        active = self.recorder.path
        return [
            {"name": p.name, "bytes": p.stat().st_size, "ts": p.stat().st_mtime, "recording": p == active}
            for p in sorted(d.glob("clip_*.mp4"), reverse=True)
        ]
