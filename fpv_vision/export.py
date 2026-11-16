"""Download YOLO nano weights and export them for fast CPU inference.

    python -m fpv_vision.export                    # yolo26n -> NCNN at imgsz 416 into <data>/models
    python -m fpv_vision.export --format onnx --model yolo11n --imgsz 320

NCNN is the fastest CPU backend on the Pi 5 in Ultralytics' own Raspberry Pi benchmarks;
ONNX Runtime is a good second choice. The exported model is fixed to ``--imgsz``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fpv_vision.config import Settings
from fpv_vision.detector import ensure_weights


def export(model: str, fmt: str, imgsz: int, models_dir: Path, force: bool = False) -> Path:
    from ultralytics import YOLO

    stem = Path(model).stem
    target = models_dir / (f"{stem}_ncnn_model" if fmt == "ncnn" else f"{stem}.onnx")
    size_file = models_dir / f"{stem}.imgsz"
    same_size = size_file.exists() and size_file.read_text().strip() == str(imgsz)
    if target.exists() and same_size and not force:
        print(f"already exported: {target}")
        return target
    pt = ensure_weights(str(models_dir / f"{stem}.pt"))
    kwargs = {"simplify": True} if fmt == "onnx" else {}
    out = YOLO(pt).export(format=fmt, imgsz=imgsz, half=False, **kwargs)
    out_path = Path(out)
    if out_path.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(out_path), target)
    size_file.write_text(str(imgsz))
    print(f"exported {target}")
    return target


def main(argv=None) -> int:
    s = Settings.from_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=s.model)
    p.add_argument("--format", choices=["ncnn", "onnx"], default="ncnn")
    p.add_argument("--imgsz", type=int, default=s.imgsz)
    p.add_argument("--data-dir", type=Path, default=s.data_dir)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    models_dir = a.data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    export(a.model, a.format, a.imgsz, models_dir, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
