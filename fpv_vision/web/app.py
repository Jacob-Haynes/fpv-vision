"""FastAPI web UI: MJPEG annotated stream, counts, snapshot gallery, clips, describe, spot report."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
BOUNDARY = "fpvframe"


def _safe_file(base: Path, name: str) -> Path:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(404)
    p = (base / name).resolve()
    if p.parent != base.resolve() or not p.is_file():
        raise HTTPException(404)
    return p


def create_app(service) -> FastAPI:
    app = FastAPI(title="fpv-vision")
    settings = service.settings

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / "index.html").read_text()

    # shared ISR kit look (vendored unchanged from ui-kit/, with fonts/)
    @app.get("/isr-ui.css")
    async def isr_ui_css():
        return FileResponse(STATIC_DIR / "isr-ui.css", media_type="text/css")

    @app.get("/isr-ui.js")
    async def isr_ui_js():
        return FileResponse(STATIC_DIR / "isr-ui.js", media_type="text/javascript")

    @app.get("/fonts/{name}")
    async def isr_ui_font(name: str):
        f = _safe_file(STATIC_DIR / "fonts", name)
        return FileResponse(f, media_type="font/woff2" if f.suffix == ".woff2" else "text/plain")

    @app.get("/stream.mjpg")
    async def stream():
        async def gen():
            last = -1
            while True:
                seq, jpeg = service.pipeline.latest_jpeg()
                if seq != last and jpeg:
                    last = seq
                    yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                           ).encode() + jpeg + b"\r\n"
                await asyncio.sleep(1 / max(2.0, settings.target_fps * 1.5))

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/frame.jpg")
    async def frame():
        _, jpeg = service.pipeline.latest_jpeg()
        if not jpeg:
            raise HTTPException(503, "no frame yet")
        return StreamingResponse(iter([jpeg]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def status():
        return service.status()

    @app.get("/api/sources")
    async def list_sources():
        return {"current": service.pipeline.source.spec.raw, "sources": service.available_sources()}

    @app.post("/api/source")
    async def set_source(body: dict = Body(...)):
        spec = str(body.get("spec", "")).strip()
        if not spec:
            raise HTTPException(400, "spec required")
        return await run_in_threadpool(service.switch_source, spec)

    @app.get("/api/cameras")
    async def cameras():
        from fpv_vision.sources import list_cameras

        return await run_in_threadpool(list_cameras)

    @app.get("/api/footage")
    async def footage():
        return {"dirs": [str(d) for d in service.footage_dirs()], "files": await run_in_threadpool(service.list_footage)}

    @app.post("/api/replay")
    async def replay(body: dict = Body(...)):
        res = service.replay_control(str(body.get("action", "")), body.get("value"))
        if not res.get("ok"):
            raise HTTPException(400, res.get("error"))
        return res

    @app.post("/api/crop")
    async def crop(body: dict = Body(default={})):
        c = body.get("crop")
        if c is not None and (not isinstance(c, list) or len(c) != 4):
            raise HTTPException(400, "crop must be [x, y, w, h] normalised 0-1, or null")
        try:
            return service.set_crop([float(v) for v in c] if c else None)
        except (TypeError, ValueError):
            raise HTTPException(400, "crop values must be numbers")

    @app.post("/api/analysis")
    async def analysis_start(body: dict = Body(...)):
        target = str(body.get("target", "")).strip()
        try:
            fps = float(body.get("sample_fps", 5))
        except (TypeError, ValueError):
            raise HTTPException(400, "sample_fps must be a number")
        res = await run_in_threadpool(service.start_analysis, target, fps)
        if not res.get("ok"):
            raise HTTPException(400, res.get("error"))
        return res

    @app.get("/api/analysis/{job_id}")
    async def analysis_get(job_id: str):
        job = service.jobs.get(job_id)
        if job is None:
            raise HTTPException(404)
        return job.to_dict()

    @app.post("/api/analysis/{job_id}/cancel")
    async def analysis_cancel(job_id: str):
        job = service.jobs.get(job_id)
        if job is None:
            raise HTTPException(404)
        job.cancel()
        return {"ok": True}

    @app.post("/api/snapshots/{name}/describe")
    async def describe_snapshot(name: str):
        res = await run_in_threadpool(service.describe_snapshot, name)
        if not res.get("ok"):
            raise HTTPException(404, res.get("error"))
        return res

    @app.post("/api/counts/reset")
    async def reset_counts():
        service.pipeline.counter.reset()
        service.pipeline.air_counter.reset()
        return {"ok": True}

    @app.post("/api/snapshot")
    async def snapshot():
        snap = await run_in_threadpool(service.pipeline.take_snapshot)
        if snap is None:
            raise HTTPException(503, "no frame yet")
        return snap

    @app.get("/api/snapshots")
    async def snapshots():
        return service.pipeline.list_snapshots()

    @app.get("/media/snapshots/{name}")
    async def snapshot_file(name: str):
        return FileResponse(_safe_file(settings.snapshots_dir, name))

    @app.post("/api/describe")
    async def describe(body: dict = Body(default={})):
        mode = str(body.get("mode", "auto"))
        if mode not in ("auto", "local", "gemini", "vlm"):
            raise HTTPException(400, "mode must be auto|local|vlm|gemini")
        return await run_in_threadpool(service.describe, mode)

    @app.post("/api/periodic")
    async def periodic(body: dict = Body(...)):
        try:
            seconds = float(body.get("seconds", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "seconds must be a number")
        return {"periodic_seconds": service.set_periodic(seconds), "enabled": service.describer.enabled}

    @app.post("/api/run")
    async def run(body: dict = Body(...)):
        return service.set_running(bool(body.get("on", True)))

    @app.post("/api/air")
    async def air_toggle(body: dict = Body(...)):
        res = await run_in_threadpool(service.set_air, bool(body.get("on", False)))
        if not res.get("ok"):
            raise HTTPException(409, res.get("error", "Air watch unavailable"))
        return res

    @app.post("/api/air/mark")
    async def air_mark(body: dict = Body(...)):
        try:
            track = int(body.get("track"))
        except (TypeError, ValueError):
            raise HTTPException(400, "track must be a number")
        return service.mark_air(track, bool(body.get("hostile", True)))

    @app.post("/api/hud")
    async def hud(body: dict = Body(...)):
        settings.show_hud = bool(body.get("on", False))
        return {"hud": settings.show_hud}

    @app.post("/api/record/start")
    async def record_start():
        path = service.pipeline.recorder.start(service.pipeline.stats.render_fps or settings.target_fps)
        return {"ok": True, "file": path.name}

    @app.post("/api/record/stop")
    async def record_stop():
        path = service.pipeline.recorder.stop()
        return {"ok": True, "file": path.name if path else None}

    @app.get("/api/clips")
    async def clips():
        return service.pipeline.list_clips()

    @app.get("/media/clips/{name}")
    async def clip_file(name: str):
        return FileResponse(_safe_file(settings.clips_dir, name), media_type="video/mp4", filename=name)

    @app.post("/api/spot-report")
    async def spot_report(body: dict = Body(...)):
        res = await run_in_threadpool(
            service.spot_report,
            position=str(body.get("position", "")),
            activity=str(body.get("activity", ""))[:300],
            snapshot=body.get("snapshot") or None,
            unit=str(body.get("unit", "unk"))[:40] or "unk",
            send_cot=bool(body.get("send_cot", True)),
            send_mesh=bool(body.get("send_mesh", True)),
        )
        if not res.get("ok"):
            raise HTTPException(400, res.get("error", "failed"))
        return res

    return app
