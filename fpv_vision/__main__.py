"""Entry point: ``python -m fpv_vision [--simulate] [--source SPEC] ...``"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from fpv_vision.config import Settings

log = logging.getLogger("fpv_vision")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="fpv-vision", description="Live object detection on drone/handheld video (ISR demo)")
    p.add_argument("--simulate", action="store_true", help="use the public-domain sample clip (no hardware)")
    p.add_argument("--source", help="video source spec: uvc:0, rtsp://..., file:clip.mp4, gopro-usb, simulate")
    p.add_argument("--port", type=int, help="web UI port (default 8110)")
    p.add_argument("--host", help="bind address (default 0.0.0.0)")
    p.add_argument("--model", help="model name/path, e.g. yolo26n, yolo11n, /path/yolo26n_ncnn_model")
    p.add_argument("--imgsz", type=int, help="inference size (default 416)")
    p.add_argument("--no-detect", action="store_true", help="stream only, no detection")
    p.add_argument("--data-dir", type=Path, help="snapshots/clips/models dir (default /var/lib/fpv-vision or ./data)")
    p.add_argument("--no-tak", action="store_true", help="disable CoT output")
    p.add_argument("--no-mesh", action="store_true", help="disable meshbus output")
    p.add_argument("--benchmark", type=int, metavar="N", help="time N frames of detection on the source and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def build_settings(args) -> Settings:
    s = Settings.from_env()
    if args.simulate:
        s.source = "simulate"
    if args.source:
        s.source = args.source
    if args.port:
        s.port = args.port
    if args.host:
        s.host = args.host
    if args.model:
        s.model = args.model
    if args.imgsz:
        s.imgsz = args.imgsz
    if args.no_detect:
        s.detector = "none"
    if args.data_dir:
        s.data_dir = args.data_dir
    if args.no_tak:
        s.tak_enabled = False
    if args.no_mesh:
        s.mesh_enabled = False
    return s


def run_benchmark(settings: Settings, n: int) -> int:
    import time

    from fpv_vision.detector import benchmark, build_detector
    from fpv_vision.service import make_source

    det = build_detector(settings)
    if det.name == "none":
        print("detector unavailable:", det.info())
        return 1
    src = make_source(settings.source, settings).start()
    frames, last = [], -1
    deadline = time.time() + 60
    while len(frames) < n and time.time() < deadline:
        fid, f, _ = src.latest()
        if f is not None and fid != last:
            frames.append(f)
            last = fid
        time.sleep(0.01)
    src.stop()
    if not frames:
        print("no frames from source", settings.source)
        return 1
    res = benchmark(det, frames)
    h, w = frames[0].shape[:2]
    print(f"{det.info()['model']} [{det.backend}] imgsz={settings.imgsz} on {w}x{h}: "
          f"{res['ms_per_frame']} ms/frame = {res['fps']} fps (detect+ByteTrack, {len(frames)} frames)")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    settings = build_settings(args)
    settings.ensure_dirs()

    if args.benchmark:
        return run_benchmark(settings, args.benchmark)

    import uvicorn

    from fpv_vision.cot import CotSender
    from fpv_vision.detector import build_detector
    from fpv_vision.gemini import GeminiDescriber
    from fpv_vision.mesh import MeshBus
    from fpv_vision.pipeline import Pipeline
    from fpv_vision.service import Service, make_source
    from fpv_vision.web.app import create_app

    detector = build_detector(settings)
    source = make_source(settings.source, settings)
    pipeline = Pipeline(settings, detector, source).start()
    gemini = GeminiDescriber(settings.gemini_api_key, settings.gemini_model, settings.gemini_min_interval)
    tak = CotSender(settings.tak_host, settings.tak_port, settings.tak_enabled).start()
    mesh = MeshBus(settings.mqtt_host, settings.mqtt_port, settings.mqtt_user, settings.mqtt_pass,
                   settings.mesh_enabled).start()
    service = Service(settings, pipeline, gemini, tak, mesh).start()

    log.info("fpv-vision: source=%s detector=%s mode=%s data=%s",
             settings.source, detector.info().get("backend"), "local+gemini" if gemini.enabled else "local (offline)",
             settings.data_dir)
    if gemini.enabled:
        log.warning("Gemini cloud mode ON (%s): snapshots are sent to Google — demo footage only", settings.gemini_model)
    log.info("web UI on http://%s:%d", settings.host, settings.port)

    def shutdown(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    try:
        uvicorn.run(create_app(service), host=settings.host, port=settings.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        pipeline.stop()
        tak.stop()
        mesh.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
