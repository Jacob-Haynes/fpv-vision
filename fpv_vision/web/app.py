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

    @app.post("/api/counts/reset")
    async def reset_counts():
        service.pipeline.counter.reset()
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
        if mode not in ("auto", "local", "gemini"):
            raise HTTPException(400, "mode must be auto|local|gemini")
        return await run_in_threadpool(service.describe, mode)

    @app.post("/api/periodic")
    async def periodic(body: dict = Body(...)):
        try:
            seconds = float(body.get("seconds", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "seconds must be a number")
        return {"periodic_seconds": service.set_periodic(seconds), "gemini": service.gemini.enabled}

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
