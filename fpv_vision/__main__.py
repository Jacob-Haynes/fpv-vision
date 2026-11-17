"""Entry point: ``python -m fpv_vision [--simulate] [--source SPEC] ...``"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from fpv_vision.config import Settings

log = logging.getLogger("fpv_vision")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="fpv-vision", description="Live object detection on drone/handheld video (ISTAR demo)")
    p.add_argument("--simulate", action="store_true", help="use the public-domain sample clip (no hardware)")
    p.add_argument("--source", help="video source spec: replay:/path/clip.mp4, camera:iPhone, uvc:0, screen:1, "
                                    "rtsp://..., simulate")
    p.add_argument("--replay", metavar="PATH", nargs="+", help="replay footage file(s)/folder(s) as live, looping")
    p.add_argument("--list-cameras", action="store_true", help="print cameras (name -> index) and exit")
    p.add_argument("--probe-camera", metavar="SPEC", help="open a camera (e.g. 'camera:USB3 Video'), print the "
                   "negotiated mode, received fps and frame-to-frame difference stats (for FPV_DEDUP_THRESHOLD), exit")
    p.add_argument("--device", help="auto (CUDA > MPS > CPU), cpu, mps, cuda")
    p.add_argument("--tak-host")
    p.add_argument("--tak-port", type=int)
    p.add_argument("--mqtt-host")
    p.add_argument("--mqtt-port", type=int)
    p.add_argument("--mqtt-user")
    p.add_argument("--mqtt-pass")
    p.add_argument("--vlm-url", help="bare base URL of an OpenAI-compatible VLM, e.g. http://isr-hq.local:11434")
    p.add_argument("--vlm-backend", choices=["auto", "openai_compat", "gemini", "off"])
    p.add_argument("--port", type=int, help="web UI port (default 8110)")
    p.add_argument("--host", help="bind address (default 0.0.0.0)")
    p.add_argument("--model", help="auto, or a model name/path e.g. yolo26n, yolo26s, /path/yolo26n_ncnn_model")
    p.add_argument("--imgsz", type=int, help="inference size (default 416)")
    p.add_argument("--no-detect", action="store_true", help="stream only, no detection")
    p.add_argument("--data-dir", type=Path, help="snapshots/clips/models dir (default /var/lib/fpv-vision or ./data)")
    p.add_argument("--no-tak", action="store_true", help="disable CoT output")
    p.add_argument("--no-mesh", action="store_true", help="disable meshbus output")
    p.add_argument("--benchmark", type=int, metavar="N", help="time N frames of detection on the source and exit")
    p.add_argument("--prefetch", action="store_true",
                   help="download the sample clips and model weights for offline use, then exit (run while online)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def build_settings(args) -> Settings:
    s = Settings.from_env()
    if args.simulate:
        s.source = "simulate"
    if args.source:
        s.source = args.source
    if args.replay:
        s.source = "replay:" + "|".join(args.replay)
    for attr in ("device", "tak_host", "tak_port", "mqtt_host", "mqtt_port", "mqtt_user", "vlm_url", "vlm_backend"):
        if getattr(args, attr, None):
            setattr(s, attr, getattr(args, attr))
    if args.mqtt_pass:
        s.mqtt_pass = args.mqtt_pass
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
    info = det.info()
    print(f"{info['model']} [{det.backend}/{info.get('device')}] imgsz={info.get('imgsz')} on {w}x{h}: "
          f"{res['ms_per_frame']} ms/frame = {res['fps']} fps (detect+ByteTrack, {len(frames)} frames)")
    return 0


def probe_camera(settings: Settings, spec: str, n: int = 180) -> int:
    """Calibrate duplicate-frame dropping on real hardware (run from Terminal.app on macOS)."""
    import time

    import cv2

    from fpv_vision import sources

    if sys.platform == "darwin":
        print("camera permission:", sources.request_camera_permission())
    src = sources.VideoSource(spec, width=settings.capture_width, height=settings.capture_height,
                              fps=settings.capture_fps)
    cap = src._open_live()
    if not cap.isOpened():
        print("could not open", spec, src.resolved)
        return 1
    print("device:", src.resolved, "| negotiated:", src._describe_mode(cap))
    diffs, prev, t0, got = [], None, None, 0
    while got < n:
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = t0 or time.time()
        got += 1
        sample = frame[::8, ::8]
        if prev is not None:
            diffs.append(float(cv2.absdiff(sample, prev).mean()))
        prev = sample.copy()
    dt = time.time() - t0
    cap.release()
    diffs.sort()
    pct = lambda q: diffs[min(len(diffs) - 1, int(q * len(diffs)))]  # noqa: E731
    thr = float(os.environ.get("FPV_DEDUP_THRESHOLD", "0.5") or 0.5)
    dup = sum(d < thr for d in diffs)
    print(f"received {got} frames in {dt:.1f}s = {got / dt:.1f} fps; frame shape {frame.shape[1]}x{frame.shape[0]}")
    print("mean-abs-diff between consecutive frames (stride-8 sample):")
    print("  " + "  ".join(f"p{int(q * 100)}={pct(q):.2f}" for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)))
    print(f"  {dup}/{len(diffs)} below FPV_DEDUP_THRESHOLD={thr} -> unique ~{(got - dup) / dt:.1f} fps")
    print("Move something in front of the camera while probing; padded duplicates show as a cluster near 0.")
    return 0


def prefetch(settings: Settings) -> int:
    """Everything fpv-vision downloads on first use, fetched now so demo day can be offline."""
    from fpv_vision import sample
    from fpv_vision.detector import choose_model, ensure_weights, resolve_model, select_device

    ok = True
    for name, path in sample.prefetch(settings.data_dir).items():
        print(f"sample {name}: {path or 'MISSING (copy it into ' + str(settings.data_dir / 'samples') + ')'}")
        ok = ok and path is not None
    device = select_device(settings.device)
    model, _ = choose_model(settings, device)
    path, backend = resolve_model(model, settings.models_dir, prefer_pytorch=device != "cpu")
    if backend == "pytorch":
        try:
            path = ensure_weights(path)
        except Exception as e:  # noqa: BLE001
            print(f"model {model}: MISSING ({e})")
            return 1
    print(f"model {model}: {path} ({backend}, {device})")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    settings = build_settings(args)
    settings.ensure_dirs()

    if args.list_cameras:
        from fpv_vision.sources import list_cameras

        cams = list_cameras()
        for c in cams:
            print(f"{c['index']}: {c['name']}    -> --source 'camera:{c['name']}'")
        if not cams:
            print("no cameras found (Linux: v4l2-ctl --list-devices)")
        return 0

    if args.prefetch:
        return prefetch(settings)

    if args.benchmark:
        return run_benchmark(settings, args.benchmark)

    if args.probe_camera:
        return probe_camera(settings, args.probe_camera)

    import uvicorn

    from fpv_vision import __version__
    from fpv_vision.tak_client import TakClient
    from fpv_vision.detector import build_detector
    from fpv_vision.gemini import GeminiDescriber
    from fpv_vision.mesh import MeshBus
    from fpv_vision.pipeline import Pipeline
    from fpv_vision.service import Service, make_source
    from fpv_vision.vlm import OpenAICompatDescriber, SceneDescriber
    from fpv_vision.web.app import create_app

    if sys.platform == "darwin" and settings.request_camera:
        # Camera permission must be requested from the main thread (see sources.request_camera_permission).
        import os

        os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
        from fpv_vision.sources import request_camera_permission

        cam = request_camera_permission()
        if cam != "authorized":
            log.warning("macOS camera access is '%s': live cameras (incl. iPhone Continuity Camera) won't open. "
                        "System Settings > Privacy & Security > Camera: allow your terminal app, then restart.", cam)

    detector = build_detector(settings)
    source = make_source(settings.source, settings)
    pipeline = Pipeline(settings, detector, source).start()
    gemini = GeminiDescriber(settings.gemini_api_key, settings.gemini_model, settings.gemini_min_interval)
    local_vlm = OpenAICompatDescriber(settings.vlm_url, settings.vlm_model, settings.vlm_api_key,
                                      settings.vlm_min_interval) if settings.vlm_url else None
    describer = SceneDescriber(settings.vlm_backend, local_vlm, gemini)
    from fpv_vision.service import short_hostname

    # Contract C5: one long-lived, registered TAK connection, opened at startup
    tak = TakClient(settings.tak_host, settings.tak_port, settings.tak_enabled, tool="fpv-vision", label="FPV",
                    station=short_hostname(), lat=settings.sensor_lat, lon=settings.sensor_lon,
                    version=__version__).start()
    mesh = MeshBus(settings.mqtt_host, settings.mqtt_port, settings.mqtt_user, settings.mqtt_pass,
                   enabled=settings.mqtt_enabled, mesh_enabled=settings.mesh_enabled)
    service = Service(settings, pipeline, describer, tak, mesh, detector_factory=lambda: build_detector(settings))
    mesh.heartbeat_fn = service.heartbeat
    mesh.start()
    service.start()
    pipeline.hud_mode = {"openai_compat": "LOCAL+VLM", "gemini": "LOCAL+GEMINI"}.get(describer.selected, "LOCAL")

    log.info("fpv-vision: source=%s detector=%s/%s scene=%s data=%s", settings.source,
             detector.info().get("backend"), detector.info().get("device", "-"), describer.selected, settings.data_dir)
    if gemini.enabled and describer.selected in ("gemini", "openai_compat"):
        log.warning("Gemini cloud mode available (%s): snapshots may be sent to Google — demo footage only",
                    settings.gemini_model)
    log.info("web UI on http://%s:%d  (heartbeat url %s)", settings.host, settings.port, service.public_url())

    def _sigterm(*_):
        raise KeyboardInterrupt

    # uvicorn captures SIGINT/SIGTERM during serve(), then restores this handler and re-raises
    # the signal on exit: without a Python handler the default action would kill us before the
    # cleanup below (heartbeat 'stopped', camera release) runs.
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        # timeout_graceful_shutdown: open MJPEG streams never finish on their own, so don't wait
        # for them on SIGTERM (pi-profiles allows 15 s; we aim for ~3 s).
        uvicorn.run(create_app(service), host=settings.host, port=settings.port, log_level="warning",
                    timeout_graceful_shutdown=2)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("shutting down")
        service.stop()
        mesh.stop()  # publishes retained status "stopped"
        pipeline.stop()
        tak.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
