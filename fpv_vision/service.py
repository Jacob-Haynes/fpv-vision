"""Glue between pipeline, scene describer (VLM/Gemini), TAK, mesh and analysis — what the web API calls."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from pathlib import Path

from fpv_vision import cot, sources, spotrep, summary
from fpv_vision.detector import Detection
from fpv_vision.gemini import RateLimited
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline
from fpv_vision.vlm import SceneDescriber

log = logging.getLogger(__name__)

BACKEND_LABELS = {"openai_compat": "VLM", "gemini": "Gemini", "local": "local", "off": "local"}


def make_source(spec: str, settings) -> sources.VideoSource:
    """Build a VideoSource, fetching the sample clip for ``simulate[:aerial|street]`` specs.

    ``auto`` picks the first USB capture device on Linux (the Pi) if one is plugged in, else
    the sample clip.
    """
    if (spec or "").strip().lower() == "auto":
        devs = sources.list_uvc_devices()
        spec = f"uvc:{devs[0]}" if devs else "simulate"
        log.info("source auto -> %s", spec)
    parsed = sources.parse_spec(spec)
    sample_path, loop = None, 0.0
    if parsed.kind == "simulate":
        from fpv_vision import sample

        info = sample.sample_info(parsed.target or None)
        sample_path = sample.ensure_sample(settings.data_dir, parsed.target or None, url=settings.sample_url)
        loop = float(info.get("loop_seconds") or 0)
    return sources.VideoSource(spec, width=settings.capture_width, height=settings.capture_height,
                               sample_path=sample_path, loop_seconds=loop, fps=settings.capture_fps)


def short_hostname() -> str:
    """Contract C4.1: env ISR_HOST if set, else the lowercased short socket.gethostname()."""
    env = os.environ.get("ISR_HOST", "").strip()
    if env:
        return env.lower()
    return socket.gethostname().split(".")[0].lower()


class Service:
    def __init__(self, settings, pipeline: Pipeline, describer: SceneDescriber, tak: cot.CotSender, mesh: MeshBus,
                 detector_factory=None):
        self.settings = settings
        self.pipeline = pipeline
        self.describer = describer
        self.tak = tak
        self.mesh = mesh
        self.detector_factory = detector_factory
        self.periodic_seconds = settings.gemini_periodic_seconds if describer.enabled else 0.0
        self.last_summary: dict | None = None
        self.reports: list[dict] = []
        self.jobs: dict = {}
        self._stop = threading.Event()
        self._periodic_thread: threading.Thread | None = None
        self.started = time.time()

    def start(self) -> "Service":
        self._periodic_thread = threading.Thread(target=self._periodic_loop, name="periodic", daemon=True)
        self._periodic_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for j in self.jobs.values():
            j.cancel()

    # ------------------------------------------------------------ describe
    def describe(self, mode: str = "auto", note: str = "") -> dict:
        """Snapshot + summary. mode: auto (VLM chain if configured, else local) | local | vlm/gemini (= auto)."""
        snap = self.pipeline.take_snapshot(note=note)
        if snap is None:
            return {"ok": False, "error": "no video frame yet"}
        snap["summary"], snap["summary_mode"] = snap["local_summary"], "local"
        result = {"ok": True, "snapshot": snap}
        if mode != "local" and self.describer.enabled:
            img_path = self.settings.snapshots_dir / snap["raw_image"]
            try:
                text, backend = self.describer.describe(img_path.read_bytes(), context=snap["local_summary"])
                snap["summary"], snap["summary_mode"] = text, backend
                snap["summary_model"] = self.describer.status()["model"]
            except RateLimited as e:
                result["warning"] = f"Scene model rate limit: retry in {e.retry_in:.0f}s (local summary shown)"
            except RuntimeError as e:
                result["warning"] = f"Scene model unavailable, local summary shown ({e})"
        elif mode not in ("auto", "local"):
            result["warning"] = "No scene model configured (VLM_URL / GEMINI_API_KEY); local summary shown"
        self.pipeline.save_snapshot_meta(snap)
        self.last_summary = {"text": snap["summary"], "mode": snap["summary_mode"], "ts": snap["ts"],
                             "snapshot": snap["name"], "model": snap.get("summary_model")}
        return result

    def describe_snapshot(self, name: str) -> dict:
        """Run the scene model on an existing snapshot (e.g. a post-flight best frame)."""
        snap = self.pipeline.load_snapshot(name)
        if snap is None:
            return {"ok": False, "error": f"snapshot {name} not found"}
        if not self.describer.enabled:
            return {"ok": True, "snapshot": snap, "warning": "No scene model configured; local summary kept"}
        try:
            text, backend = self.describer.describe((self.settings.snapshots_dir / snap["raw_image"]).read_bytes(),
                                                    context=snap.get("local_summary", ""))
            snap["summary"], snap["summary_mode"] = text, backend
            snap["summary_model"] = self.describer.status()["model"]
            self.pipeline.save_snapshot_meta(snap)
            return {"ok": True, "snapshot": snap}
        except (RateLimited, RuntimeError) as e:
            return {"ok": True, "snapshot": snap, "warning": f"Scene model unavailable: {e}"}

    def set_periodic(self, seconds: float) -> float:
        if not self.describer.enabled:
            self.periodic_seconds = 0.0
            return 0.0
        self.periodic_seconds = 0.0 if seconds <= 0 else max(seconds, self.describer.min_interval)
        return self.periodic_seconds

    def _periodic_loop(self) -> None:
        last = 0.0
        while not self._stop.wait(1.0):
            if self.periodic_seconds <= 0 or not self.describer.enabled:
                continue
            if time.time() - last < self.periodic_seconds:
                continue
            dets, _ = self.pipeline.current_detections()
            if not dets:  # don't spend model calls on empty scenes
                continue
            last = time.time()
            try:
                self.describe("auto", note="periodic")
            except Exception as e:  # noqa: BLE001
                log.warning("periodic summary failed: %s", e)

    # ------------------------------------------------------------ spot report
    def spot_report(self, *, position: str, activity: str = "", snapshot: str | None = None, unit: str = "unk",
                    send_cot: bool = True, send_mesh: bool = True) -> dict:
        position = (position or "").strip() or self.default_position()
        if not position:
            return {"ok": False, "error": "no position: enter a grid / lat,lon (or set FPV_DEFAULT_POSITION or "
                                          "SENSOR_LAT/SENSOR_LON)"}
        if snapshot:
            snap = self.pipeline.load_snapshot(snapshot)
            if snap is None:
                return {"ok": False, "error": f"snapshot {snapshot} not found"}
        else:
            snap = self.pipeline.take_snapshot(note="spot report")
            if snap is None:
                return {"ok": False, "error": "no video frame yet"}
            snap["summary"], snap["summary_mode"] = snap["local_summary"], "local"
        dets = [Detection(d["cls_id"], d["label"], d["conf"], tuple(d["xyxy"]), d.get("track_id"))
                for d in snap.get("detections", [])]
        try:
            rep = spotrep.build(dets=dets, position=position, activity=activity,
                                summary_text=snap.get("summary") or snap.get("local_summary", ""),
                                callsign=self.settings.callsign, unit=unit, provenance=self.provenance_of(snap))
        except ValueError as e:
            return {"ok": False, "error": f"position: {e}"}
        if snap.get("video_file"):  # be explicit that this came from recorded footage
            rep["remarks"] += f"\nFrom recorded footage: {snap['video_file']} t={snap.get('video_time', 0)}s"
        out = {"ok": True, "report": rep, "snapshot": snap["name"], "cot": None, "mesh": None}
        if send_cot:
            xml = cot.spot_report_cot(lat=rep["lat"], lon=rep["lon"], callsign=rep["callsign"],
                                      remarks=rep["remarks"], sender=self.settings.callsign,
                                      sender_uid=getattr(self.tak, "uid", None),
                                      sender_callsign=getattr(self.tak, "callsign", None))
            queued = self.tak.send(xml)
            out["cot"] = {"queued": queued, "connected": self.tak.connected,
                          "registered": getattr(self.tak, "registered", False),
                          "error": None if queued else self.tak.last_error}
        if send_mesh:
            sent = self.mesh.send(rep["mesh_text"])
            out["mesh"] = {"sent": sent, "text": rep["mesh_text"], "error": None if sent else self.mesh.last_error}
        snap.setdefault("reports", []).append({"ts": time.time(), **{k: rep[k] for k in ("grid", "dtg", "mesh_text")}})
        self.pipeline.save_snapshot_meta(snap)
        self.reports.insert(0, {"ts": time.time(), **out})
        del self.reports[20:]
        return out

    def default_position(self) -> str:
        """Operator default (FPV_DEFAULT_POSITION), else shared SENSOR_LAT/SENSOR_LON. Never 0,0."""
        if self.settings.default_position:
            return self.settings.default_position
        lat, lon = self.settings.sensor_lat, self.settings.sensor_lon
        if lat is not None and lon is not None and not (lat == 0 and lon == 0):
            return f"{lat:.6f}, {lon:.6f}"
        return ""

    @staticmethod
    def provenance_of(snap: dict) -> str:
        if snap.get("provenance") in ("live", "simulated", "recorded"):
            return snap["provenance"]
        src = str(snap.get("source", "")).lower()  # snapshots from before provenance was recorded
        if src.startswith(("simulate", "synthetic")) or "/samples/" in src:
            return "simulated"
        if snap.get("video_file") or src.startswith(("replay:", "file:")) or src.endswith(tuple(sources.VIDEO_EXTS)):
            return "recorded"
        return "live"

    # ------------------------------------------------------------ footage library
    def footage_dirs(self) -> list[Path]:
        dirs = [Path(os.path.expanduser(p)) for p in self.settings.footage_dirs.split(os.pathsep) if p.strip()]
        dirs.append(self.settings.data_dir / "footage")
        dirs.append(self.settings.clips_dir)
        seen, out = set(), []
        for d in dirs:
            if d.is_dir() and d.resolve() not in seen:
                seen.add(d.resolve())
                out.append(d)
        return out

    def list_footage(self, limit: int = 300) -> list[dict]:
        items = []
        for d in self.footage_dirs():
            for f in sources.expand_playlist(str(d))[:limit]:
                p = Path(f)
                try:
                    st = p.stat()
                except OSError:
                    continue
                items.append({"path": str(p), "name": p.name, "dir": str(p.parent), "bytes": st.st_size,
                              "mtime": st.st_mtime})
        return items[:limit]

    # ------------------------------------------------------------ sources
    def available_sources(self) -> list[dict]:
        items = [{"spec": "simulate", "label": "Simulate: street crossing, handheld (CC0)", "group": "simulate"},
                 {"spec": "simulate:aerial", "label": "Simulate: NASA drone pass (public domain)", "group": "simulate"}]
        cams = sources.list_cameras()
        for c in cams:
            spec = f"camera:{c['name']}" if not str(c["index"]).startswith("/dev/") else f"uvc:{c['index']}"
            items.append({"spec": spec, "label": f"Camera: {c['name']}", "group": "camera"})
        if not cams:
            items.append({"spec": "uvc:0", "label": "Camera #0", "group": "camera"})
        if self.settings.allow_screen:
            items.append({"spec": "screen:1", "label": "Screen capture (mirrored phone / DJI Fly) - main display",
                          "group": "screen"})
        cur = self.pipeline.source.spec.raw
        if not any(i["spec"] == cur for i in items):
            items.insert(0, {"spec": cur, "label": cur, "group": "current"})
        return items

    def switch_source(self, spec: str) -> dict:
        s = sources.parse_spec(spec)
        if s.kind == "replay" and not sources.expand_playlist(s.target):
            return {"ok": False, "error": f"no video files found: {s.target}"}
        if s.kind == "screen" and not self.settings.allow_screen:
            return {"ok": False, "error": "screen capture disabled (FPV_ALLOW_SCREEN=0)"}
        src = make_source(spec, self.settings)
        self.pipeline.switch_source(src)
        self.pipeline.counter.reset()
        return {"ok": True, "source": src.spec.raw}

    def replay_control(self, action: str, value=None) -> dict:
        src = self.pipeline.source
        if not src.is_replay:
            return {"ok": False, "error": "current source is not a replay"}
        if action == "seek":
            src.seek(float(value or 0))
        elif action == "pause":
            src.pause(bool(value))
        elif action == "next":
            src.next_file(1)
        elif action == "prev":
            src.next_file(-1)
        else:
            return {"ok": False, "error": f"unknown action {action}"}
        return {"ok": True}

    def set_crop(self, crop) -> dict:
        self.pipeline.source.set_crop(tuple(crop) if crop else None)
        self.pipeline.detector.reset_tracks()
        return {"ok": True, "crop": self.pipeline.source.crop}

    # ------------------------------------------------------------ post-flight analysis
    def start_analysis(self, target: str, sample_fps: float = 5.0) -> dict:
        from fpv_vision.analysis import AnalysisJob

        if self.detector_factory is None:
            return {"ok": False, "error": "no detector available for analysis"}
        if any(j.state in ("queued", "running") for j in self.jobs.values()):
            return {"ok": False, "error": "an analysis is already running"}
        if not sources.expand_playlist(target):
            return {"ok": False, "error": f"no video files found: {target}"}
        job = AnalysisJob(target, self.pipeline, self.detector_factory, sample_fps=sample_fps).start()
        self.jobs[job.id] = job
        for old in list(self.jobs)[:-5]:
            self.jobs.pop(old)
        return {"ok": True, "job": job.to_dict(full=False)}

    # ------------------------------------------------------------ heartbeat / status
    def mode(self) -> str:
        kind = self.pipeline.source.spec.kind
        if kind in ("simulate", "synthetic") or self.pipeline.source.resolved == "synthetic":
            return "simulate"
        if kind == "replay":
            return "replay"
        return getattr(self.pipeline.detector, "accelerator", "cpu")

    def health(self) -> str:
        src = self.pipeline.source
        if src.error and not src.connected:
            return "error" if time.time() - self.started > 30 else "degraded"
        if self.pipeline.detector.name == "none" or not src.connected:
            return "degraded"
        return "ok"

    def public_url(self) -> str:
        if self.settings.public_url:
            return self.settings.public_url
        return f"http://{short_hostname()}.local:{self.settings.port}/"

    def detail(self) -> str:
        """Human summary for the launcher, e.g. 'yolo26s mps 1080p 42fps, VLM qwen3-vl:8b-instruct @ isr-hq'."""
        p = self.pipeline
        info = p.detector.info()
        st = p.source.status()
        parts = []
        if info.get("name") == "none":
            parts.append("no detector")
        else:
            model = Path(str(info.get("model", ""))).name.replace("_ncnn_model", "").replace(".pt", "")
            parts.append(f"{model} {info.get('accelerator', 'cpu')}")
        if st.get("height"):
            parts.append(f"{st['height']}p")
        if p.stats.detect_fps:
            parts.append(f"{p.stats.detect_fps:.0f}fps")
        head = " ".join(parts)
        vlm = self.describer.status()
        if vlm["active"] == "openai_compat" and self.describer.openai:
            from urllib.parse import urlparse

            host = (urlparse(self.describer.openai.base).hostname or "").removesuffix(".local")
            return f"{head}, VLM {vlm['model']} @ {host}"
        if vlm["active"] == "gemini":
            return f"{head}, Gemini {vlm['model']}"
        return f"{head}, local summary"

    def heartbeat(self) -> dict:
        return {"tool": "fpv-vision", "host": short_hostname(),
                "url": self.public_url(), "mode": self.mode(), "status": self.health(), "ts": int(time.time()),
                "detail": self.detail()}

    def status(self) -> dict:
        p = self.pipeline
        dets, shape = p.current_detections()
        rec = p.recorder
        vlm = self.describer.status()
        return {
            "mode": self.mode(),
            "health": self.health(),
            "scene_backend": vlm["active"],
            "source": p.source.status(),
            "detector": p.detector.info(),
            "stats": {
                "render_fps": p.stats.render_fps, "detect_fps": p.stats.detect_fps,
                "detect_ms": p.stats.detect_ms, "detect_gap_ms": p.stats.detect_gap_ms,
                "frames_per_detection": p.stats.skip_ratio, "pipeline_ms": p.stats.pipeline_ms,
                "box_lag_ms": p.stats.box_lag_ms, "encode_ms": p.stats.encode_ms,
                "latency_ms": round(p.stats.pipeline_ms + p.stats.box_lag_ms, 1),
            },
            "counts": summary.count_by_label(dets),
            "totals": p.counter.snapshot(),
            "live_summary": summary.describe(dets, shape[1] or 1) if shape[1] else "",
            "last_summary": self.last_summary,
            "recording": {"active": rec.active, "file": rec.path.name if rec.path else None,
                          "seconds": round(time.time() - rec.started) if rec.active else 0,
                          "max_seconds": rec.max_seconds},
            "vlm": {**vlm, "periodic_seconds": self.periodic_seconds},
            "tak": self.tak.status(),
            "mesh": self.mesh.status(),
            "heartbeat": self.heartbeat(),
            "camera_permission": sources.camera_permission_status(),
            "default_position": self.default_position(),
            "callsign": self.settings.callsign,
            "reports": self.reports[:5],
            "jobs": [j.to_dict(full=False) for j in list(self.jobs.values())[-3:]],
        }
