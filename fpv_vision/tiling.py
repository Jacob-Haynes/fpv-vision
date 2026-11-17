"""Tiled detection for small, distant objects: shared by Air watch and Search.

A 4K frame squeezed into a 640 px model input turns a person 30 px tall into 5 px. Cutting the
frame into overlapping tiles (as in SAHI) and running them as one batch beside the full frame
keeps small objects big enough to detect. Boxes are mapped back to frame coordinates, tile
boxes cut by an inner tile edge are dropped (the full-frame pass sees those objects whole), and
the rest are merged with greedy NMS before tracking.
"""

from __future__ import annotations

import numpy as np


def tiles(w: int, h: int, grid: int = 2, overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    """``grid``×``grid`` overlapping tiles (x0, y0, x1, y1) covering the frame, plus the full frame first."""
    out = [(0, 0, w, h)]
    if grid <= 1:
        return out
    tw, th = w / (grid - (grid - 1) * overlap), h / (grid - (grid - 1) * overlap)
    for r in range(grid):
        for c in range(grid):
            x0, y0 = int(round(c * tw * (1 - overlap))), int(round(r * th * (1 - overlap)))
            out.append((x0, y0, min(w, int(round(x0 + tw))), min(h, int(round(y0 + th)))))
    return out


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def _area(z: np.ndarray) -> np.ndarray:
    return (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = _inter(a, b)
    return inter / np.maximum(_area(a) + _area(b) - inter, 1e-9)


def _ios(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box: catches a tile's half-box inside the full-frame box."""
    return _inter(a, b) / np.maximum(np.minimum(_area(a), _area(b)), 1e-9)


def drop_cut(rows: np.ndarray, rect: tuple[int, int, int, int], w: int, h: int, margin: int = 4) -> np.ndarray:
    """Drop tile boxes cut by an *inner* tile edge: those objects are whole in the full-frame pass."""
    x0, y0, x1, y1 = rect
    cut = np.zeros(len(rows), bool)
    if x0 > 0:
        cut |= rows[:, 0] <= x0 + margin
    if y0 > 0:
        cut |= rows[:, 1] <= y0 + margin
    if x1 < w:
        cut |= rows[:, 2] >= x1 - margin
    if y1 < h:
        cut |= rows[:, 3] >= y1 - margin
    return rows[~cut]


def merge(boxes: np.ndarray, iou: float = 0.5, ios: float = 0.8, groups: dict[int, int] | None = None) -> np.ndarray:
    """Greedy NMS over rows [x1, y1, x2, y2, conf, cls]; highest confidence wins.

    ``groups`` maps a class id to a group: boxes only suppress boxes in the same group, so the
    same van called "car" in one tile and "truck" in another ends up as ONE box, but a person
    standing beside it is kept. No ``groups`` = class-agnostic (Air watch: drone vs bird). A box
    mostly inside a kept box is dropped too, which removes the partial boxes tiles produce.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 6)
    order = np.argsort(-boxes[:, 4])
    b = boxes[order]
    grp = np.array([groups.get(int(c), int(c)) for c in b[:, 5]]) if groups else np.zeros(len(b), int)
    keep = []
    alive = np.ones(len(b), bool)
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.where(alive)[0]
        rest = rest[(rest > i) & (grp[rest] == grp[i])]
        if len(rest):
            drop = (_iou(b[i], b[rest]) > iou) | (_ios(b[i], b[rest]) > ios)
            alive[rest[drop]] = False
    return b[keep]


def predict_tiles(model, frame: np.ndarray, grid: int, *, imgsz: int, conf: float, device: str,
                  classes: list[int] | None = None, overlap: float = 0.15,
                  groups: dict[int, int] | None = None) -> np.ndarray:
    """Run ``model`` (an Ultralytics predictor) on the full frame + tiles as one batch.

    Returns merged rows [x1, y1, x2, y2, conf, model_cls] in frame coordinates.
    """
    h, w = frame.shape[:2]
    rects = tiles(w, h, grid, overlap)
    crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
    results = model.predict(crops, imgsz=imgsz, conf=conf, device=device, classes=classes, verbose=False)
    rows = []
    for rect, r in zip(rects, results):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        xyxy[:, [0, 2]] += rect[0]
        xyxy[:, [1, 3]] += rect[1]
        row = np.column_stack([xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy()])
        rows.append(row if rect == rects[0] else drop_cut(row, rect, w, h))
    return merge(np.concatenate(rows), groups=groups) if rows else np.zeros((0, 6))


def byte_tracker(conf: float):
    """A fresh Ultralytics BYTETracker whose high/new-track thresholds follow ``conf``."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    cfg.update(track_high_thresh=conf, track_low_thresh=0.1, new_track_thresh=conf)
    return BYTETracker(IterableSimpleNamespace(**cfg))


def track(tracker, rows: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Feed merged rows to ``tracker``; returns rows [x1, y1, x2, y2, track_id, conf, cls, idx]."""
    from ultralytics.engine.results import Boxes

    rows = rows.astype(np.float32) if len(rows) else np.zeros((0, 6), np.float32)
    return tracker.update(Boxes(rows, frame.shape[:2]).cpu().numpy(), frame)


class HitGate:
    """Show a track only after ``hits`` consecutive passes with it (cuts one-frame flicker).

    A track missed for a pass starts again from zero. ``update`` takes this pass's track ids and
    returns the set that may be shown.
    """

    def __init__(self, hits: int = 2):
        self.hits = max(1, int(hits))
        self._run: dict[int, int] = {}

    def update(self, track_ids) -> set[int]:
        seen = set(track_ids)
        self._run = {t: self._run.get(t, 0) + 1 for t in seen}  # absent tracks drop out: not consecutive
        return {t for t, n in self._run.items() if n >= self.hits}

    def reset(self) -> None:
        self._run = {}
