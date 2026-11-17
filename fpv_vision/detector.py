"""Object detection + tracking on the CPU.

``YoloDetector`` wraps Ultralytics YOLO (nano) with ByteTrack. It prefers an exported
NCNN model (fastest on the Pi 5's Cortex-A76), then ONNX, then the PyTorch weights.
Export with ``python -m fpv_vision.export`` (``deploy/install.sh`` does this for you).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# COCO class ids
PERSON = {0}
VEHICLES = {1, 2, 3, 5, 6, 7}  # bicycle, car, motorcycle, bus, train, truck
WATERCRAFT = {8}  # boat (no aircraft: FPV looks at the ground; air detection is OP sentry's)
ISR_CLASSES = sorted(PERSON | VEHICLES | WATERCRAFT)

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "aeroplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat",
}


def category(cls_id: int) -> str:
    if cls_id >= 100:  # Air watch classes (fpv_vision.air)
        return "air"
    if cls_id in PERSON:
        return "person"
    if cls_id in VEHICLES:
        return "vehicle"
    return "other"


@dataclass
class Detection:
    cls_id: int
    label: str
    conf: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None
    note: str = ""  # extra detail for the label, e.g. "hexacopter" from the vision model
    named_by: str = ""  # "ai" when the class came from the vision model (fpv_vision.air_namer)

    @property
    def category(self) -> str:
        return category(self.cls_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category
        return d


class Detector:
    name = "none"
    backend = "none"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []

    def reset_tracks(self) -> None:  # pragma: no cover - trivial
        pass

    def info(self) -> dict:
        return {"name": self.name, "backend": self.backend}


class NullDetector(Detector):
    """Stream-only mode (no model available or detection disabled)."""

    def __init__(self, reason: str = "disabled"):
        self.reason = reason

    def info(self) -> dict:
        return {"name": "none", "backend": "none", "reason": self.reason}


def select_device(pref: str = "auto") -> str:
    """auto -> CUDA > Apple MPS > CPU. Explicit values (cpu, mps, cuda, 0) are passed through."""
    pref = (pref or "auto").strip().lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - torch missing/broken -> CPU
        pass
    return "cpu"


def accelerator_of(device: str) -> str:
    d = device.lower()
    if d.startswith("cuda") or d.isdigit():
        return "cuda"
    return "mps" if d == "mps" else "cpu"


def choose_model(settings, device: str) -> tuple[str, int]:
    """Model + inference size: bigger model on a GPU, nano on CPU (the Pi)."""
    gpu = accelerator_of(device) != "cpu"
    model = settings.model
    if model in ("", "auto"):
        model = settings.gpu_model if gpu else settings.cpu_model
    imgsz = settings.imgsz or (settings.gpu_imgsz if gpu else settings.cpu_imgsz)
    return model, imgsz


def resolve_model(model: str, models_dir: Path, prefer_pytorch: bool = False) -> tuple[str, str]:
    """Return (path_or_name, backend) for a model name like ``yolo26n``.

    Explicit paths are used as given. Otherwise look in ``models_dir`` for an exported
    NCNN dir, then ONNX, then .pt; finally fall back to downloading the .pt weights.
    """
    p = Path(model).expanduser()
    if p.exists():
        if p.is_dir() and p.name.endswith("_ncnn_model"):
            return str(p), "ncnn"
        return str(p), {".onnx": "onnx", ".pt": "pytorch"}.get(p.suffix, "unknown")
    stem = p.stem if p.suffix else model
    order = [(models_dir / f"{stem}_ncnn_model", "ncnn"), (models_dir / f"{stem}.onnx", "onnx"),
             (models_dir / f"{stem}.pt", "pytorch")]
    if prefer_pytorch:  # GPU (CUDA/MPS) runs the PyTorch weights; NCNN/ONNX here are CPU paths
        order = order[2:]
    for candidate, backend in order:
        if candidate.exists():
            return str(candidate), backend
    return str(models_dir / f"{stem}.pt"), "pytorch"


def ensure_weights(pt_path: str) -> str:
    """Download official Ultralytics weights into our models dir if missing (needs internet once)."""
    path = Path(pt_path)
    if path.exists():
        return str(path)
    from ultralytics.utils.downloads import attempt_download_asset

    path.parent.mkdir(parents=True, exist_ok=True)
    return str(attempt_download_asset(path))


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, model: str, models_dir: Path, *, imgsz: int = 416, conf: float = 0.35,
                 classes: list[int] | None = None, device: str = "cpu", tracker: str = "bytetrack.yaml",
                 infer_max_width: int = 0):
        from ultralytics import YOLO  # heavy import; keep it lazy

        self.accelerator = accelerator_of(device)
        path, backend = resolve_model(model, models_dir, prefer_pytorch=self.accelerator != "cpu")
        if backend == "pytorch":
            path = ensure_weights(path)
        self.model_path, self.backend = path, backend
        # exported models are fixed-size; export.py records the size next to them
        size_file = models_dir / f"{Path(path).name.removesuffix('_ncnn_model').removesuffix('.onnx')}.imgsz"
        if backend in ("ncnn", "onnx") and size_file.exists():
            imgsz = int(size_file.read_text().strip() or imgsz)
        self.imgsz, self.conf, self.classes, self.device, self.tracker = imgsz, conf, classes, device, tracker
        self.infer_max_width = infer_max_width
        self._model = YOLO(path, task="detect")
        self._lock = threading.Lock()
        self.names: dict[int, str] = dict(getattr(self._model, "names", None) or COCO_NAMES)
        self.names[4] = "aeroplane"  # British English on screen
        log.info("YOLO loaded: %s (%s on %s) imgsz=%d", path, backend, device, imgsz)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # 4K/2.7K footage: downscale once here (cheap) rather than inside Ultralytics; boxes are
        # mapped back to the original frame's coordinates.
        scale = 1.0
        h, w = frame.shape[:2]
        if self.infer_max_width and w > self.infer_max_width:
            import cv2

            scale = w / self.infer_max_width
            frame = cv2.resize(frame, (self.infer_max_width, int(h / scale)), interpolation=cv2.INTER_AREA)
        with self._lock:
            results = self._model.track(
                frame, persist=True, tracker=self.tracker, imgsz=self.imgsz, conf=self.conf,
                classes=self.classes, device=self.device, verbose=False,
            )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = (boxes.xyxy.cpu().numpy() * scale).astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for b, c, cf, tid in zip(xyxy, cls, confs, ids):
            out.append(Detection(int(c), self.names.get(int(c), str(c)), float(cf),
                                 (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                                 int(tid) if tid is not None else None))
        return out

    def reset_tracks(self) -> None:
        with self._lock:
            pred = getattr(self._model, "predictor", None)
            for t in getattr(pred, "trackers", None) or []:
                t.reset()

    def info(self) -> dict:
        return {"name": "yolo", "backend": self.backend, "model": self.model_path, "imgsz": self.imgsz,
                "conf": self.conf, "tracker": "ByteTrack", "device": self.device, "accelerator": self.accelerator}


def build_detector(settings) -> Detector:
    if settings.detector == "none":
        return NullDetector("disabled in config")
    try:
        classes = None if settings.all_classes else ISR_CLASSES
        device = select_device(settings.device)
        model, imgsz = choose_model(settings, device)
        return YoloDetector(model, settings.models_dir, imgsz=imgsz, conf=settings.conf, classes=classes,
                            device=device, infer_max_width=settings.infer_max_width)
    except Exception as e:  # noqa: BLE001 - offline with no weights must still stream
        log.error("detector unavailable, streaming without detection: %s", e)
        return NullDetector(f"model unavailable: {e}")


def benchmark(detector: Detector, frames: list[np.ndarray], warmup: int = 5) -> dict:
    """Mean per-frame detect+track latency over ``frames``."""
    for f in frames[:warmup]:
        detector.detect(f)
    t0 = time.perf_counter()
    for f in frames:
        detector.detect(f)
    dt = (time.perf_counter() - t0) / max(len(frames), 1)
    return {"ms_per_frame": round(dt * 1000, 1), "fps": round(1 / dt, 1) if dt else 0.0}
