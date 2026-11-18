"""Air watch: open-vocabulary detection of drones, helicopters, aeroplanes and birds.

COCO has "airplane" but no drone or helicopter, and small drones come out as "bird" or "kite".
Air watch runs an open-vocabulary Ultralytics YOLO-World v2 model, prompted with text, beside the normal
detector. Small targets: each frame is cut into 2×2 overlapping tiles plus the full frame (as in
SAHI), the tiles are run as one batch, boxes are mapped back and merged with NMS, and then tracked
with ByteTrack. Each track gets one class by confidence-weighted vote (``TrackCounter``).

Licence: YOLO-World v2 turns the prompts into embeddings with OpenAI's CLIP ViT-B/32 (MIT). It never
uses YOLOE, whose text encoder (Apple MobileCLIP, ``mobileclip*.ts``) is under a research-only licence
that this kit must not use; YOLOE model names are refused.

Offline: CLIP (~340 MB, in ``~/.cache/clip``) is needed only to turn the prompts into embeddings.
``prefetch()`` (``--setup-only`` with ``FPV_AIR=1``) downloads the weights, sets the prompts once and
saves a model with them baked in, in the models dir; at run time only that file is loaded, with no
network and no text encoder.
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

AIR_MODEL = "yolov8s-worldv2.pt"  # YOLO-World v2 (Ultralytics, AGPL-3.0) + OpenAI CLIP text encoder (MIT)
LICENCE_NOTE = ("Air watch (FPV_AIR=1) downloads YOLO-World v2 weights (Ultralytics, AGPL-3.0) and, once, "
                "OpenAI CLIP ViT-B/32 (MIT, ~340 MB) for its text prompts. It never uses YOLOE or Apple "
                "MobileCLIP (research-only licence). Air detection is OP sentry's job.")
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
def _check_licence(model: str) -> None:
    if "yoloe" in Path(model).name.lower() or "mobileclip" in Path(model).name.lower():
        raise ValueError(f"{model}: YOLOE needs Apple MobileCLIP (research-only licence), which this kit must "
                         "not use; Air watch uses YOLO-World v2 (OpenAI CLIP, MIT)")


def _baked_path(models_dir: Path, model: str, prompts: list[str]) -> Path:
    key = hashlib.sha1(("|".join(prompts) + "@" + model).encode()).hexdigest()[:10]
    return Path(models_dir) / f"{Path(model).stem}.air-{key}.pt"


def prefetch(models_dir: Path, model: str = AIR_MODEL, prompts: list[str] | None = None) -> Path:
    """Download YOLO-World v2 and save it with the prompts baked in (needs internet once)."""
    _check_licence(model)
    from ultralytics import YOLOWorld

    from fpv_vision.detector import ensure_weights

    prompts = prompts or PROMPTS
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    baked = _baked_path(models_dir, model, prompts)
    if baked.exists():
        return baked
    weights = ensure_weights(str(models_dir / model))
    cwd = os.getcwd()
    os.chdir(models_dir)  # Ultralytics downloads into the working directory
    try:
        m = YOLOWorld(weights)
        m.set_classes(prompts)  # OpenAI CLIP ViT-B/32
        m.save(str(baked))
    finally:
        os.chdir(cwd)
    log.info("air watch prompts baked in: %s", baked)
    return baked


def available(models_dir: Path, model: str = AIR_MODEL, prompts: list[str] | None = None) -> bool:
    return _baked_path(models_dir, model, prompts or PROMPTS).exists()


# ---------------------------------------------------------------- detector
class AirDetector:
    name = "air"

    def __init__(self, models_dir: Path, device: str = "cpu", *, model: str = AIR_MODEL, imgsz: int = 640,
                 conf: float = 0.2, grid: int = 2, prompts: list[str] | None = None):
        _check_licence(model)
        from ultralytics import YOLOWorld

        self.prompts = prompts or PROMPTS
        weights = _baked_path(models_dir, model, self.prompts)
        if not weights.exists():
            raise FileNotFoundError("air watch model not downloaded: run FPV_AIR=1 scripts/run-laptop.sh "
                                    "--setup-only (or FPV_AIR=1 python -m fpv_vision --prefetch) while online")
        log.warning(LICENCE_NOTE)
        self.model_path, self.device, self.imgsz, self.conf, self.grid = str(weights), device, imgsz, conf, grid
        self._model = YOLOWorld(str(weights))  # prompts baked in: no text encoder, no network
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
