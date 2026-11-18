"""Search: the ground detector on tiles, for small, distant people and vehicles.

The normal detector squeezes the whole frame into one 640 px input, so a figure far off across
open ground is a handful of pixels and is missed or flickers. Search runs the same YOLO weights
on the full frame plus overlapping tiles (2×2, or 3×3 for 1440p and up) as one batch, merges the
boxes (``fpv_vision.tiling``), tracks them with ByteTrack, and only shows a track after it has
been seen on ``hits`` consecutive passes. Confidence is a little lower than normal (0.25) to
reach the faint ones; the hit gate keeps the one-off false positives off the screen.

It costs roughly one inference per tile, so the detection rate drops (see the README table).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from fpv_vision.detector import COCO_NAMES, Detection, accelerator_of
from fpv_vision.tiling import HitGate, byte_tracker, predict_tiles, track

log = logging.getLogger(__name__)

TRACK_ID_OFFSET = 2_000_000  # apart from the main detector's ids (and Air watch's 1,000,000)

# NMS groups: boxes only suppress boxes in the same group. One van called car in one tile and
# truck in another is one box; a person beside it, or riding a motorcycle, is kept.
GROUPS = {0: 0, 2: 1, 5: 1, 6: 1, 7: 1, 1: 2, 3: 2, 4: 3, 8: 4}


def grid_for(height: int, width: int, setting: str | int = "auto") -> int:
    """Tile grid for a frame: 3×3 when the short side is 1440 px or more, else 2×2 (or as set)."""
    s = str(setting).strip().lower()
    if s.isdigit():
        return max(1, int(s))
    return 3 if min(height, width) >= 1440 else 2


class SearchDetector:
    name = "search"
    backend = "pytorch"

    def __init__(self, model_path: str, *, device: str = "cpu", imgsz: int = 640, conf: float = 0.25,
                 classes: list[int] | None = None, grid: str | int = "auto", hits: int = 2,
                 names: dict[int, str] | None = None, model=None, fast_track: bool = True):
        if model is None:
            from ultralytics import YOLO  # its own instance: the main one carries tracker callbacks

            model = YOLO(model_path, task="detect")
        self._model = model
        self.model_path, self.device, self.imgsz, self.conf = model_path, device, imgsz, conf
        self.classes, self.grid_setting, self.hits = classes, grid, hits
        self.names = dict(names or COCO_NAMES)
        self.accelerator = accelerator_of(device)
        self.grid = 0  # set from the first frame
        self._lock = threading.Lock()
        self.gate = HitGate(hits)
        self.fast_track = fast_track
        self._tracker = byte_tracker(conf, fast_track)
        log.info("search loaded: %s on %s, conf %.2f, %d-hit gate", Path(model_path).name, device, conf, hits)

    @classmethod
    def from_detector(cls, det, settings) -> "SearchDetector":
        """Search with the same weights, classes and device as the main YoloDetector."""
        return cls(det.model_path, device=det.device, imgsz=det.imgsz, conf=settings.search_conf,
                   classes=det.classes, grid=settings.search_grid, hits=settings.search_hits, names=det.names,
                   fast_track=getattr(det, "fast_track", True))

    def reset_tracks(self) -> None:
        with self._lock:
            self._tracker = byte_tracker(self.conf, self.fast_track)
            self.gate.reset()

    def raw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.grid = grid_for(h, w, self.grid_setting)
        return predict_tiles(self._model, frame, self.grid, imgsz=self.imgsz, conf=self.conf, device=self.device,
                             classes=self.classes, groups=GROUPS)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        with self._lock:
            tracked = track(self._tracker, self.raw(frame), frame)
            shown = self.gate.update(int(t[4]) for t in tracked)
        out = []
        for t in tracked:
            x1, y1, x2, y2, tid, conf, c = t[:7]
            if int(tid) not in shown:
                continue
            c = int(c)
            out.append(Detection(c, self.names.get(c, str(c)), float(conf), (int(x1), int(y1), int(x2), int(y2)),
                                 int(tid) + TRACK_ID_OFFSET))
        return out

    def info(self) -> dict:
        g = self.grid or grid_for(0, 0, self.grid_setting)
        return {"name": "search", "backend": self.backend, "model": self.model_path, "imgsz": self.imgsz,
                "conf": self.conf, "tiles": f"{g}×{g} + full frame", "hits": self.hits, "tracker": "ByteTrack",
                "device": self.device, "accelerator": self.accelerator}
