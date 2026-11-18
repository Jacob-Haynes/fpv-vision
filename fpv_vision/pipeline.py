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
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from fpv_vision import annotate, summary
from fpv_vision.detector import Detection, Detector, category
from fpv_vision.sources import VideoSource
from fpv_vision.stable import CAR_REF_MIN, CAR_REF_SAMPLES, StableConfig, iou

log = logging.getLogger(__name__)
AIR_COCO = {4, 14, 33}  # COCO airplane, bird, kite: replaced by Air watch classes when it is on

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

    With a :class:`~fpv_vision.stable.StableConfig` (the ground picture) it also filters and holds
    tracks so boxes neither lie nor flicker: implausible vehicle boxes are dropped, a track is
    shown and counted only once confirmed (consecutive detections plus the class confidence
    floor), a confirmed track keeps one label and coasts on its last box for a moment when
    missed. Without one (Air watch, and ``FPV_STABLE=0``) it relabels and counts only.
    """

    def __init__(self, min_hits: int = MIN_TRACK_HITS, stable: StableConfig | None = None):
        self.min_hits = min_hits
        self.stable = stable
        self._hits: dict[int, int] = defaultdict(int)
        self._votes: dict[int, dict[str, list]] = {}  # track -> label -> [conf sum, n, cls_id]
        self._counted: set[int] = set()
        self._override: dict[int, tuple[int, str]] = {}  # track -> (cls_id, label) set by the vision model
        self.just_counted: list[int] = []  # tracks that reached min_hits in the last update()
        self.untracked_peak: dict[str, int] = defaultdict(int)
        self._streak: dict[int, int] = {}  # stable: consecutive passes an unconfirmed track was seen
        self._shown: dict[int, tuple[int, str]] = {}  # stable: the one label a confirmed track shows
        self._last: dict[int, tuple[Detection, float]] = {}  # stable: last box + time, for coasting
        self._alias: dict[int, int] = {}  # stable: tracker id -> the held track it took over
        self._car_areas: deque[int] = deque(maxlen=CAR_REF_SAMPLES)  # stable: recent car box sizes
        self._search = False  # stable: this pass came from Search (its own class floors)

    def voted(self, track_id: int) -> tuple[int, str, float]:
        """(cls_id, label, mean confidence) of a track's winning class, after the bus/train fold.

        A class set by the vision model (:meth:`set_override`) wins over the vote."""
        votes = self._votes[track_id]
        if track_id in self._override:
            cls_id, label = self._override[track_id]
            total, n = sum(v[0] for v in votes.values()), sum(v[1] for v in votes.values())
            return cls_id, label, total / max(1, n)
        label, (total, n, cls_id) = max(list(votes.items()), key=lambda kv: kv[1][0])
        mean = total / n
        cls_id, label = _fold(label, cls_id, mean)
        return cls_id, label, mean

    def _vote(self, d: Detection) -> None:
        self._hits[d.track_id] += 1
        v = self._votes.setdefault(d.track_id, {}).setdefault(d.label, [0.0, 0, d.cls_id])
        v[0] += d.conf
        v[1] += 1

    def update(self, dets: list[Detection], frame_shape: tuple[int, int] | None = None, now: float | None = None,
               prior_hits: int = 0, search: bool = False) -> list[Detection]:
        """Record this frame's detections; return them relabelled with each track's voted class.

        With ``stable`` set, return only confirmed tracks (plus coasting ones). ``frame_shape``
        (h, w) is for the vehicle size check, ``now`` the frame's time, and ``prior_hits`` the
        consecutive passes a detector has already required before it reports a track (Search);
        ``search``: these came from Search, so its class floors apply."""
        if self.stable is not None:
            self._search = search
            return self._update_stable(dets, frame_shape, time.time() if now is None else now, prior_hits)
        untracked: dict[str, int] = defaultdict(int)
        out = []
        self.just_counted = []
        for d in dets:
            if d.track_id is None:
                cls_id, label = _fold(d.label, d.cls_id, d.conf)
                untracked[label] += 1
                out.append(Detection(cls_id, label, d.conf, d.xyxy, None) if label != d.label else d)
                continue
            self._vote(d)
            if d.track_id not in self._counted and self._hits[d.track_id] >= self.min_hits:
                self._counted.add(d.track_id)
                self.just_counted.append(d.track_id)
            cls_id, label, _ = self.voted(d.track_id)
            out.append(d if label == d.label else Detection(cls_id, label, d.conf, d.xyxy, d.track_id))
        for k, n in untracked.items():
            self.untracked_peak[k] = max(self.untracked_peak[k], n)
        return out

    def _update_stable(self, dets: list[Detection], frame_shape, now: float, prior_hits: int) -> list[Detection]:
        cfg = self.stable
        self.just_counted = []
        seen: dict[int, Detection] = {}
        fresh = []
        for d in dets:
            if d.label == "car":
                self._car_areas.append((d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]))
        car_area = float(np.median(self._car_areas)) if len(self._car_areas) >= CAR_REF_MIN else 0.0
        held: dict[int, list[Detection]] = defaultdict(list)
        for d in dets:
            # no track id: nothing to confirm it against (ByteTrack only reports activated tracks)
            if d.track_id is None or not cfg.plausible(d, frame_shape, car_area):
                continue
            tid = self._alias.get(d.track_id, d.track_id)
            (held[tid] if tid in self._counted else fresh).append(d)
        for tid, ds in held.items():
            # the track's own id first; an id that took it over while it was missed, and now
            # turns out to be a different object beside it, becomes its own track (already shown,
            # so confirmed at once, with the same label) rather than lose its box
            ds.sort(key=lambda d: d.track_id != tid)
            seen[tid] = replace(ds[0], track_id=tid)
            for d in ds[1:]:
                if d.track_id in seen:
                    continue
                self._alias.pop(d.track_id, None)
                self._counted.add(d.track_id)
                self.just_counted.append(d.track_id)
                self._shown[d.track_id] = self._shown.get(tid, (d.cls_id, d.label))
                seen[d.track_id] = d
        for d in fresh:
            # the tracker gave a held object a new id (a blur, a pan): the new id takes over the old
            # track, its label and its count, instead of being a new object that has to confirm again
            old = self._handover(d, seen)
            if old is not None:
                self._alias[d.track_id] = old
                d = replace(d, track_id=old)
            if d.track_id not in seen:
                seen[d.track_id] = d
        for tid, d in seen.items():
            self._vote(d)
            if tid not in self._counted:
                self._streak[tid] = self._streak.get(tid, max(0, prior_hits)) + 1
        for tid in [t for t in self._streak if t not in seen]:
            del self._streak[tid]  # a miss before confirmation starts the count again
        out = []
        for tid, d in seen.items():
            cls_id, label, mean = self.voted(tid)
            if tid not in self._counted:
                if self._streak[tid] < cfg.needed(cls_id) or mean < cfg.floor(label, self._search):
                    continue
                self._counted.add(tid)
                self.just_counted.append(tid)
                self._streak.pop(tid, None)
            cls_id, label = self._stable_label(tid, cls_id, label)
            shown = Detection(cls_id, label, round(mean, 3), d.xyxy, tid, d.note, d.named_by)
            self._last[tid] = (shown, now)
            out.append(shown)
        for tid, (last, t) in list(self._last.items()):
            if tid in seen:
                continue
            if now - t <= cfg.coast_s:
                out.append(replace(last, coast=True))
            else:
                del self._last[tid]
        return out

    def _handover(self, d: Detection, seen: dict[int, Detection]) -> int | None:
        """The coasting track (confirmed, missed this pass, same category) this new detection
        overlaps most, if the overlap is at least ``handover_iou``."""
        best, best_iou = None, self.stable.handover_iou
        if best_iou <= 0:
            return None
        for tid, (last, _) in self._last.items():
            if tid in seen or category(last.cls_id) != category(d.cls_id):
                continue
            o = iou(last.xyxy, d.xyxy)
            if o >= best_iou:
                best, best_iou = tid, o
        return best

    def _stable_label(self, tid: int, cls_id: int, label: str) -> tuple[int, str]:
        """The label a confirmed track keeps: it moves to the vote's winner only when that class's
        confidence total is ``label_switch`` times the current label's (bus/train folded as voted)."""
        cur = self._shown.get(tid)
        if cur is None or cur[1] == label or tid in self._override:
            self._shown[tid] = (cls_id, label)
            return self._shown[tid]
        sums: dict[str, float] = defaultdict(float)
        for lab, (total, n, c) in self._votes[tid].items():
            sums[_fold(lab, c, total / n)[1]] += total
        # ... and only to a class whose own floor it meets (a car never drifts into a weak "truck")
        n = sum(v[1] for lab, v in self._votes[tid].items() if _fold(lab, v[2], v[0] / v[1])[1] == label)
        if sums[label] >= self.stable.label_switch * sums[cur[1]] and sums[label] / max(1, n) >= self.stable.floor(label, self._search):
            self._shown[tid] = (cls_id, label)
        return self._shown[tid]

    def label_of(self, tid: int) -> str:
        shown = self._shown.get(tid)
        return shown[1] if shown else self.voted(tid)[1]

    def forget_live(self) -> None:
        """New tracker ids (source switched, Search toggled): drop held boxes and part-confirmed tracks."""
        self._streak.clear()
        self._last.clear()
        self._alias.clear()
        self._car_areas.clear()

    def set_override(self, track_id: int, cls_id: int, label: str) -> None:
        self._override[track_id] = (cls_id, label)

    @property
    def totals(self) -> dict[str, int]:
        t: dict[str, int] = defaultdict(int)
        for tid in list(self._counted):  # read from the web thread while detect updates
            t[self.label_of(tid)] += 1
        return dict(t)

    def reset(self) -> None:
        self.__init__(self.min_hits, self.stable)

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
        self.counter = TrackCounter(stable=StableConfig.from_settings(settings))  # the ground picture
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
        # Air watch (fpv_vision.air): a second, open-vocabulary detector on its own thread
        self.air = None  # AirDetector once loaded
        self.air_on = False
        self.air_counter = TrackCounter()
        self.air_hostile: set[int] = set()  # air track ids the operator marked hostile
        self.air_stats = {"fps": 0.0, "ms": 0.0}
        self.on_air_track = None  # callback(Detection) when an air track is first counted
        self.air_namer = None  # fpv_vision.air_namer.AirNamer: the vision model names air tracks
        self._air_dets: list[Detection] = []
        # Search (fpv_vision.search): the ground detector on tiles, in place of the normal pass
        self.search = None  # SearchDetector once loaded
        self.search_on = False

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "Pipeline":
        self.source.start()
        for target, name in ((self._detect_loop, "detect"), (self._render_loop, "render"), (self._air_loop, "air")):
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
            self._air_dets = []
        self.detector.reset_tracks()
        self.counter.forget_live()
        if self.search is not None:
            self.search.reset_tracks()
        if self.air is not None:
            self.air.reset_tracks()
        self.air_counter.reset()
        self.air_hostile.clear()
        if self.air_namer is not None:
            self.air_namer.reset()
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
            return list(self._dets) + (list(self._air_dets) if self.air_on else []), self._dets_frame_shape

    def air_detections(self) -> list[Detection]:
        with self._lock:
            return list(self._air_dets) if self.air_on else []

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
            if frame is None and self._dets:  # source disconnected: no stale boxes or counts
                with self._lock:
                    self._dets = []
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            t0 = time.perf_counter()
            det = self.search if (self.search_on and self.search is not None) else self.detector
            try:
                dets = det.detect(frame)
            except Exception as e:  # noqa: BLE001
                log.exception("detector error: %s", e)
                dets = []
                time.sleep(0.5)
            dt = time.perf_counter() - t0
            if src is not self.source or det is not (self.search if self.search_on else self.detector):
                continue  # source switched or Search toggled mid-inference
            if self.air_on:  # the air model owns aircraft and birds; avoid double boxes
                dets = [d for d in dets if d.cls_id not in AIR_COCO]
            # one voted class per track; with FPV_STABLE (default) confirmed tracks only, plus brief coasting
            prior = max(0, getattr(det, "hits", 1) - 1) if det is self.search else 0  # Search's own gate
            dets = self.counter.update(dets, frame.shape[:2], fts or time.time(), prior_hits=prior,
                                       search=det is self.search)
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

    def _air_loop(self) -> None:
        """Air watch at up to settings.air_fps on the newest frame; never blocks the main detector."""
        last_id, n, t_window = -1, 0, time.time()
        while not self._stop.is_set():
            air, src = self.air, self.source
            if not (self.air_on and air is not None and self.running):
                with self._lock:
                    self._air_dets = []
                self._stop.wait(0.2)
                continue
            fid, frame, _ = src.latest()
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            t0 = time.perf_counter()
            try:
                dets = air.detect(frame)
            except Exception as e:  # noqa: BLE001
                log.exception("air watch error: %s", e)
                self._stop.wait(1.0)
                continue
            dt = time.perf_counter() - t0
            if src is not self.source:
                continue
            dets = self.air_counter.update(dets)
            namer = self.air_namer
            if namer is not None:
                namer.observe(dets, frame)  # cheap: queues crops; naming runs on its own thread
                dets = namer.apply(dets)
            with self._lock:
                self._air_dets = dets
            if self.on_air_track:
                by_id = {d.track_id: d for d in dets}
                for tid in self.air_counter.just_counted:
                    if tid in by_id:
                        try:
                            self.on_air_track(by_id[tid])
                        except Exception as e:  # noqa: BLE001
                            log.warning("air track callback failed: %s", e)
            self.air_stats["ms"] = round(dt * 1000, 1)
            n += 1
            if time.time() - t_window >= 2.0:
                self.air_stats["fps"] = round(n / (time.time() - t_window), 1)
                n, t_window = 0, time.time()
            rate = self.settings.air_fps
            if self.search_on:  # Search and Air watch share the GPU: the ground picture comes first
                rate = min(rate, self.settings.search_air_fps)
            period = 1.0 / max(0.5, rate)
            if dt < period:
                self._stop.wait(period - dt)

    def set_air(self, on: bool) -> None:
        self.air_on = bool(on) and self.air is not None
        if not self.air_on:
            with self._lock:
                self._air_dets = []

    def set_search(self, on: bool) -> None:
        on = bool(on) and self.search is not None
        if on != self.search_on:
            (self.search if on else self.detector).reset_tracks()  # start the incoming tracker clean
            self.counter.forget_live()
            with self._lock:
                self._dets = []
        self.search_on = on

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
                dets = list(self._dets) + (list(self._air_dets) if self.air_on else [])
                dshape = self._dets_frame_shape
                dets_ts = self._dets_ts
            out = annotate.resize_to_width(frame, self.settings.stream_width)
            if dshape[0] and dshape[1]:
                dets = annotate.scale_detections(dets, out.shape[1] / dshape[1], out.shape[0] / dshape[0])
            out = out.copy() if out is frame else out
            counts = summary.count_by_label(dets)
            accel = getattr(self.detector, "accelerator", "cpu").upper()
            hud = [
                f"FPV-VISION  {self.hud_mode}  {self.detector.backend.upper()}/{accel}"
                + ("  SEARCH" if self.search_on else ""),
                f"video {self.stats.render_fps:.0f}fps  det {self.stats.detect_fps:.1f}fps  "
                f"lat {self.stats.pipeline_ms + self.stats.box_lag_ms:.0f}ms",
            ]
            if counts:
                hud.append("  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
            annotate.draw(out, dets, hud if self.settings.show_hud else None, recording=self.recorder.active,
                          badge=self._badge(src), alert_tracks=self.air_hostile)
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
