"""Air watch: open-vocabulary detection of drones, helicopters, aeroplanes and birds.

COCO has "airplane" but no drone or helicopter, and small drones come out as "bird" or "kite".
Air watch runs an open-vocabulary Ultralytics YOLOE model, prompted with text, beside the normal
detector. Small targets: each frame is cut into 2×2 overlapping tiles plus the full frame (as in
SAHI), the tiles are run as one batch, boxes are mapped back and merged with NMS, and then tracked
with ByteTrack. Each track gets one class by confidence-weighted vote (``TrackCounter``).

Offline: YOLOE needs a text encoder (MobileCLIP, ~250 MB) only to turn the prompts into
embeddings. ``prefetch()`` (run by ``--setup-only``) downloads the weights, computes the
embeddings once and caches them in the models dir; at run time only the weights and that
small cache file are loaded, with no network and no text encoder.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

import numpy as np

from fpv_vision.detector import Detection

log = logging.getLogger(__name__)

AIR_MODEL = "yoloe-26s-seg.pt"
PROMPTS = ["drone", "quadcopter", "helicopter", "aeroplane", "bird"]

# class ids >= AIR_BASE never collide with COCO (0-79); detector.category() calls them "air"
AIR_BASE = 100
DRONE, HELICOPTER, AEROPLANE, BIRD = AIR_BASE, AIR_BASE + 1, AIR_BASE + 2, AIR_BASE + 3
AIR_CONTACT = AIR_BASE + 4  # the vision model looked but could not say (low confidence or "other")
# prompt -> (cls_id, label shown on screen)
PROMPT_CLASS = {
    "drone": (DRONE, "drone"),
    "quadcopter": (DRONE, "drone"),  # two prompts, one class
    "helicopter": (HELICOPTER, "helicopter"),
    "aeroplane": (AEROPLANE, "aeroplane"),
    "bird": (BIRD, "bird"),
}
LABELS = {DRONE: "drone", HELICOPTER: "helicopter", AEROPLANE: "aeroplane", BIRD: "bird", AIR_CONTACT: "air contact"}

# CoT types (MIL-STD-2525 / APP-6 style): unknown UAV, unknown rotary wing, neutral civil fixed wing
COT_TYPE = {"drone": "a-u-A-M-F-Q", "helicopter": "a-u-A-M-H", "aeroplane": "a-n-A-C", "air contact": "a-u-A"}

TRACK_ID_OFFSET = 1_000_000  # keeps air track ids apart from the main detector's


def map_prompt(prompt: str) -> tuple[int, str]:
    """Prompt name from the model -> (cls_id, label). Unknown prompts fall back to drone."""
    return PROMPT_CLASS.get(prompt.strip().lower(), (DRONE, "drone"))


def cot_type(label: str, hostile: bool = False) -> str | None:
    """CoT type for an air label (None for birds: they are shown, never reported).

    Marked hostile by the operator, the affiliation becomes h (a-h-A-M-F-Q for a drone). Any
    non-bird track can be marked, because the model sometimes names a drone "helicopter".
    """
    t = COT_TYPE.get(label)
    return f"a-h-{t[4:]}" if t and hostile else t


# ---------------------------------------------------------------- tiling + NMS
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


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def _ios(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box: catches a tile's half-box inside the full-frame box."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])  # noqa: E731
    return inter / np.maximum(np.minimum(area(a), area(b)), 1e-9)


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


def merge(boxes: np.ndarray, iou: float = 0.5, ios: float = 0.8) -> np.ndarray:
    """Class-agnostic greedy NMS over rows [x1, y1, x2, y2, conf, cls]; highest confidence wins.

    Class-agnostic on purpose: the same object called "drone" in one tile and "bird" in another
    must end up as ONE box (the more confident call). A box mostly inside a kept box is dropped
    too, which removes the partial boxes that tiles produce at their edges.
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 6)
    order = np.argsort(-boxes[:, 4])
    b = boxes[order]
    keep = []
    alive = np.ones(len(b), bool)
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.where(alive)[0]
        rest = rest[rest > i]
        if len(rest):
            drop = (_iou(b[i], b[rest]) > iou) | (_ios(b[i], b[rest]) > ios)
            alive[rest[drop]] = False
    return b[keep]


# ---------------------------------------------------------------- model files
def _pe_path(models_dir: Path, model: str, prompts: list[str]) -> Path:
    key = hashlib.sha1(("|".join(prompts) + "@" + model).encode()).hexdigest()[:10]
    return Path(models_dir) / f"{Path(model).stem}.prompts-{key}.pt"


def prefetch(models_dir: Path, model: str = AIR_MODEL, prompts: list[str] | None = None) -> Path:
    """Download the YOLOE weights and cache the prompt embeddings (needs internet once)."""
    import torch
    from ultralytics import YOLOE

    from fpv_vision.detector import ensure_weights

    prompts = prompts or PROMPTS
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights = ensure_weights(str(models_dir / model))
    pe_file = _pe_path(models_dir, model, prompts)
    if pe_file.exists():
        return pe_file
    cwd = os.getcwd()
    os.chdir(models_dir)  # the text encoder downloads into the working directory
    try:
        m = YOLOE(weights)
        pe = m.get_text_pe(prompts)
    finally:
        os.chdir(cwd)
    torch.save(pe.cpu(), pe_file)
    log.info("air watch prompts cached: %s", pe_file)
    for enc in models_dir.glob("mobileclip*.ts"):  # ~250 MB, only needed to change the prompts
        enc.unlink(missing_ok=True)
    return pe_file


def available(models_dir: Path, model: str = AIR_MODEL, prompts: list[str] | None = None) -> bool:
    return (Path(models_dir) / model).exists() and _pe_path(models_dir, model, prompts or PROMPTS).exists()


# ---------------------------------------------------------------- detector
class AirDetector:
    name = "air"

    def __init__(self, models_dir: Path, device: str = "cpu", *, model: str = AIR_MODEL, imgsz: int = 640,
                 conf: float = 0.2, grid: int = 2, prompts: list[str] | None = None):
        import torch
        from ultralytics import YOLOE

        self.prompts = prompts or PROMPTS
        weights = Path(models_dir) / model
        pe_file = _pe_path(models_dir, model, self.prompts)
        if not weights.exists() or not pe_file.exists():
            raise FileNotFoundError("air watch model not downloaded: run scripts/run-laptop.sh --setup-only "
                                    "(or python -m fpv_vision --prefetch) while online")
        self.model_path, self.device, self.imgsz, self.conf, self.grid = str(weights), device, imgsz, conf, grid
        self._model = YOLOE(str(weights))
        self._model.set_classes(self.prompts, torch.load(pe_file, map_location="cpu"))
        self._names = {i: p for i, p in enumerate(self.prompts)}
        self._lock = threading.Lock()
        self._tracker = None
        self.reset_tracks()
        log.info("air watch loaded: %s on %s, %d×%d tiles + full frame", weights.name, device, grid, grid)

    def reset_tracks(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = YAML.load(check_yaml("bytetrack.yaml"))
        cfg.update(track_high_thresh=self.conf, track_low_thresh=0.1, new_track_thresh=self.conf)
        self._tracker = BYTETracker(IterableSimpleNamespace(**cfg))

    def raw(self, frame: np.ndarray) -> np.ndarray:
        """Merged (untracked) rows [x1, y1, x2, y2, conf, model_cls] in frame coordinates."""
        h, w = frame.shape[:2]
        rects = tiles(w, h, self.grid)
        crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in rects]
        with self._lock:
            results = self._model.predict(crops, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
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
        return merge(np.concatenate(rows)) if rows else np.zeros((0, 6))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from ultralytics.engine.results import Boxes

        rows = self.raw(frame)
        tracked = self._tracker.update(Boxes(rows.astype(np.float32), frame.shape[:2]).cpu().numpy(), frame) \
            if len(rows) else self._tracker.update(Boxes(np.zeros((0, 6), np.float32), frame.shape[:2]), frame)
        out = []
        for t in tracked:
            x1, y1, x2, y2, tid, conf, cls = t[:7]
            cls_id, label = map_prompt(self._names.get(int(cls), "drone"))
            out.append(Detection(cls_id, label, float(conf), (int(x1), int(y1), int(x2), int(y2)),
                                 int(tid) + TRACK_ID_OFFSET))
        return out

    def info(self) -> dict:
        return {"model": Path(self.model_path).name, "device": self.device, "imgsz": self.imgsz, "conf": self.conf,
                "tiles": f"{self.grid}×{self.grid} + full frame", "prompts": self.prompts}
