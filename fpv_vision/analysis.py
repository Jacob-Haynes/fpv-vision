"""Post-flight batch analysis: run the detector over a whole recording as fast as possible.

Fly, land, pull the SD card, point this at the clip (or folder): it samples frames at
``sample_fps``, runs detection + ByteTrack, and produces
  * a per-second timeline of people / vehicles / other in view,
  * unique tracked counts,
  * the best frames (most people/vehicles, highest confidence, spread over the clip) saved
    as normal gallery snapshots so they can be described and spot-reported.

Runs in its own thread with its own detector instance, so the live/replay view keeps going.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

import cv2

from fpv_vision import annotate, summary
from fpv_vision.pipeline import TrackCounter
from fpv_vision.sources import expand_playlist

log = logging.getLogger(__name__)

MAX_BEST = 12
MIN_GAP_S = 3.0


def frame_score(dets) -> float:
    return sum((2.0 if d.category == "person" else 1.0 if d.category == "vehicle" else 0.3) * d.conf for d in dets)


def pick_best(candidates: list[dict], k: int = MAX_BEST, min_gap: float = MIN_GAP_S) -> list[dict]:
    """Highest-scoring frames, at least ``min_gap`` seconds apart (per file)."""
    chosen: list[dict] = []
    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if c["score"] <= 0:
            break
        if all(c["file"] != o["file"] or abs(c["t"] - o["t"]) >= min_gap for o in chosen):
            chosen.append(c)
        if len(chosen) >= k:
            break
    return sorted(chosen, key=lambda c: (c["file"], c["t"]))


class AnalysisJob:
    def __init__(self, target: str, pipeline, detector_factory, sample_fps: float = 5.0):
        self.id = uuid.uuid4().hex[:8]
        self.target = target
        self.files = expand_playlist(target)
        self.pipeline = pipeline
        self.detector_factory = detector_factory
        self.sample_fps = max(0.5, sample_fps)
        self.state = "queued"
        self.error = ""
        self.progress = 0.0
        self.processed_s = 0.0
        self.total_s = 0.0
        self.started = 0.0
        self.finished = 0.0
        self.speed = 0.0  # video seconds processed per wall second
        self.timeline: list[dict] = []
        self.totals: dict[str, int] = {}
        self.best: list[dict] = []
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "AnalysisJob":
        self._thread = threading.Thread(target=self._run, name=f"analysis-{self.id}", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def to_dict(self, full: bool = True) -> dict:
        d = {
            "id": self.id, "target": self.target, "files": [Path(f).name for f in self.files], "paths": self.files, "state": self.state,
            "error": self.error, "progress": round(self.progress, 3), "processed_s": round(self.processed_s, 1),
            "total_s": round(self.total_s, 1), "speed": round(self.speed, 2), "sample_fps": self.sample_fps,
            "elapsed_s": round((self.finished or time.time()) - self.started, 1) if self.started else 0,
            "totals": self.totals, "best": self.best,
        }
        if full:
            d["timeline"] = self.timeline
        return d

    def _run(self) -> None:
        self.started = time.time()
        if not self.files:
            self.state, self.error = "error", f"no video files found: {self.target}"
            return
        self.state = "running"
        try:
            for f in self.files:
                cap = cv2.VideoCapture(f, cv2.CAP_FFMPEG)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                self.total_s += n / fps if n > 0 and 1 <= fps <= 240 else 0
                cap.release()
            detector = self.detector_factory()
            counter = TrackCounter()
            candidates: list[dict] = []
            offset = 0.0
            for f in self.files:
                if self._cancel.is_set():
                    break
                offset = self._process_file(f, detector, counter, candidates, offset)
            self.totals = counter.snapshot()
            best = pick_best(candidates)
            self.best = [self._save_best(c) for c in best]
            self.state = "cancelled" if self._cancel.is_set() else "done"
            self.progress = 1.0 if self.state == "done" else self.progress
        except Exception as e:  # noqa: BLE001
            log.exception("analysis failed")
            self.state, self.error = "error", str(e)
        finally:
            self.finished = time.time()
            self._persist()

    def _process_file(self, path: str, detector, counter: TrackCounter, candidates: list[dict], offset: float) -> float:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.warning("analysis: cannot open %s", path)
            return offset
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not 1 <= fps <= 240:
            fps = 30.0
        step = max(1, round(fps / self.sample_fps))
        detector.reset_tracks()
        idx = 0
        bucket: dict[int, dict] = {}
        best_in_sec: dict[int, dict] = {}
        name = Path(path).name
        while not self._cancel.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            dets = detector.detect(frame)
            dets = counter.update(dets)
            sec = int(t)
            b = bucket.setdefault(sec, {"t": round(offset + sec, 1), "file": name, "file_t": sec,
                                        "person": 0, "vehicle": 0, "other": 0})
            for cat in ("person", "vehicle", "other"):
                b[cat] = max(b[cat], sum(1 for d in dets if d.category == cat))
            score = frame_score(dets)
            best_here = best_in_sec.get(sec)
            if score > 0 and (best_here is None or score > best_here["score"]):
                # one candidate per second, kept as an in-memory JPEG to bound memory
                small = annotate.resize_to_width(frame, 1920)
                best_in_sec[sec] = {"file": name, "path": path, "t": round(t, 2), "score": round(score, 3),
                                    "jpeg": annotate.encode_jpeg(small, 90), "dets": dets, "orig_w": frame.shape[1]}
            # skip ahead to the next sample (grab = decode without colour conversion)
            for _ in range(step - 1):
                if not cap.grab():
                    break
            idx += step
            self.processed_s = offset + idx / fps
            if self.total_s:
                self.progress = min(0.999, self.processed_s / self.total_s)
            elapsed = time.time() - self.started
            self.speed = self.processed_s / elapsed if elapsed > 0 else 0.0
            if (idx // step) % 25 == 0:
                self.timeline = self.timeline_from(bucket, self.timeline)
        cap.release()
        candidates.extend(best_in_sec.values())
        self.timeline = self.timeline_from(bucket, self.timeline)
        return offset + idx / fps

    @staticmethod
    def timeline_from(bucket: dict, existing: list[dict]) -> list[dict]:
        keep = {(e["file"], e["file_t"]): e for e in existing}
        for b in bucket.values():
            keep[(b["file"], b["file_t"])] = b
        return sorted(keep.values(), key=lambda e: e["t"])

    def _save_best(self, c: dict) -> dict:
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(c["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        sx = frame.shape[1] / c["orig_w"]
        dets = annotate.scale_detections(c["dets"], sx, sx)
        ann = annotate.draw(frame.copy(), dets, [f"POST-FLIGHT {c['file']} t={c['t']:.1f}s"], badge="REPLAY ANALYSIS")
        meta = self.pipeline.save_snapshot(frame, ann, dets, frame.shape[1], note="post-flight", source=c["path"],
                                           extra={"video_file": c["file"], "video_time": c["t"],
                                                  "analysis": self.id, "score": c["score"],
                                                  "provenance": self._provenance(c["path"])}, prefix="pf")
        meta["summary"], meta["summary_mode"] = meta["local_summary"], "local"
        self.pipeline.save_snapshot_meta(meta)
        return {"name": meta["name"], "image": meta["image"], "file": c["file"], "t": c["t"],
                "counts": meta["counts"], "summary": meta["local_summary"]}

    def _provenance(self, path: str) -> str:
        """Post-flight footage is 'recorded'; the bundled sample clips are 'simulated'."""
        try:
            samples = (self.pipeline.settings.data_dir / "samples").resolve()
            if Path(path).resolve().is_relative_to(samples):
                return "simulated"
        except (OSError, ValueError):
            pass
        return "recorded"

    def _persist(self) -> None:
        try:
            d = self.pipeline.settings.data_dir / "analysis"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"analysis_{time.strftime('%Y%m%d_%H%M%SZ', time.gmtime())}_{self.id}.json").write_text(
                json.dumps(self.to_dict(), indent=1))
        except OSError as e:
            log.warning("could not save analysis result: %s", e)
