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
from fpv_vision.tiling import byte_tracker, drop_cut, merge, predict_tiles, tiles, track  # noqa: F401

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


# tiling + NMS live in fpv_vision.tiling (shared with Search); re-exported for callers


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
        self._tracker = byte_tracker(self.conf)

    def raw(self, frame: np.ndarray) -> np.ndarray:
        """Merged (untracked) rows [x1, y1, x2, y2, conf, model_cls] in frame coordinates."""
        with self._lock:
            return predict_tiles(self._model, frame, self.grid, imgsz=self.imgsz, conf=self.conf, device=self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        out = []
        for t in track(self._tracker, self.raw(frame), frame):
            x1, y1, x2, y2, tid, conf, cls = t[:7]
            cls_id, label = map_prompt(self._names.get(int(cls), "drone"))
            out.append(Detection(cls_id, label, float(conf), (int(x1), int(y1), int(x2), int(y2)),
                                 int(tid) + TRACK_ID_OFFSET))
        return out

    def info(self) -> dict:
        return {"model": Path(self.model_path).name, "device": self.device, "imgsz": self.imgsz, "conf": self.conf,
                "tiles": f"{self.grid}×{self.grid} + full frame", "prompts": self.prompts}
