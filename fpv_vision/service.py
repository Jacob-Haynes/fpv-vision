"""Glue between pipeline, Gemini, TAK and mesh — what the web API calls."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from fpv_vision import cot, sources, spotrep, summary
from fpv_vision.detector import Detection
from fpv_vision.gemini import GeminiDescriber, RateLimited
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline

log = logging.getLogger(__name__)


def make_source(spec: str, settings) -> sources.VideoSource:
    """Build a VideoSource, fetching the sample clip for ``simulate[:aerial|street]`` specs.

    ``auto`` picks the first USB capture device if one is plugged in, else the sample clip.
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
                               sample_path=sample_path, loop_seconds=loop)


class Service:
    def __init__(self, settings, pipeline: Pipeline, gemini: GeminiDescriber, tak: cot.CotSender, mesh: MeshBus):
        self.settings = settings
        self.pipeline = pipeline
        self.gemini = gemini
        self.tak = tak
        self.mesh = mesh
        self.periodic_seconds = settings.gemini_periodic_seconds if gemini.enabled else 0.0
        self.last_summary: dict | None = None
        self.reports: list[dict] = []
        self._stop = threading.Event()
        self._periodic_thread: threading.Thread | None = None
        pipeline.hud_mode = "LOCAL+GEMINI" if gemini.enabled else "LOCAL"

    def start(self) -> "Service":
        self._periodic_thread = threading.Thread(target=self._periodic_loop, name="periodic", daemon=True)
        self._periodic_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ describe
    def describe(self, mode: str = "auto", note: str = "") -> dict:
        """Snapshot + summary. mode: auto (Gemini if configured, else local) | local | gemini."""
        snap = self.pipeline.take_snapshot(note=note)
        if snap is None:
            return {"ok": False, "error": "no video frame yet"}
        snap["summary"], snap["summary_mode"] = snap["local_summary"], "local"
        result = {"ok": True, "snapshot": snap}
        if mode in ("auto", "gemini") and self.gemini.enabled:
            img_path = self.settings.snapshots_dir / snap["raw_image"]
            try:
                text = self.gemini.describe(img_path.read_bytes(), context=snap["local_summary"])
                snap["summary"], snap["summary_mode"] = text, "gemini"
            except RateLimited as e:
                result["warning"] = f"Gemini rate limit: retry in {e.retry_in:.0f}s (showing local summary)"
            except RuntimeError as e:
                result["warning"] = f"Gemini failed, local summary shown: {e}"
        elif mode == "gemini":
            result["warning"] = "Gemini not configured (set GEMINI_API_KEY); local summary shown"
        self.pipeline.save_snapshot_meta(snap)
        self.last_summary = {"text": snap["summary"], "mode": snap["summary_mode"], "ts": snap["ts"],
                             "snapshot": snap["name"]}
        return result

    def set_periodic(self, seconds: float) -> float:
        if not self.gemini.enabled:
            self.periodic_seconds = 0.0
            return 0.0
        self.periodic_seconds = 0.0 if seconds <= 0 else max(seconds, self.gemini.min_interval)
        return self.periodic_seconds

    def _periodic_loop(self) -> None:
        last = 0.0
        while not self._stop.wait(1.0):
            if self.periodic_seconds <= 0 or not self.gemini.enabled:
                continue
            if time.time() - last < self.periodic_seconds:
                continue
            dets, _ = self.pipeline.current_detections()
            if not dets:  # don't spend API calls on empty scenes
                continue
            last = time.time()
            try:
                self.describe("gemini", note="periodic")
            except Exception as e:  # noqa: BLE001
                log.warning("periodic summary failed: %s", e)

    # ------------------------------------------------------------ spot report
    def spot_report(self, *, position: str, activity: str = "", snapshot: str | None = None, unit: str = "unk",
                    send_cot: bool = True, send_mesh: bool = True) -> dict:
        position = (position or "").strip() or self.settings.default_position
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
                                callsign=self.settings.callsign, unit=unit)
        except ValueError as e:
            return {"ok": False, "error": f"position: {e}"}
        out = {"ok": True, "report": rep, "snapshot": snap["name"], "cot": None, "mesh": None}
        if send_cot:
            xml = cot.spot_report_cot(lat=rep["lat"], lon=rep["lon"], callsign=rep["callsign"],
                                      remarks=rep["remarks"], sender=self.settings.callsign)
            queued = self.tak.send(xml)
            out["cot"] = {"queued": queued, "connected": self.tak.connected}
        if send_mesh:
            out["mesh"] = {"sent": self.mesh.send(rep["mesh_text"]), "text": rep["mesh_text"]}
        snap.setdefault("reports", []).append({"ts": time.time(), **{k: rep[k] for k in ("grid", "dtg", "mesh_text")}})
        self.pipeline.save_snapshot_meta(snap)
        self.reports.insert(0, {"ts": time.time(), **out})
        del self.reports[20:]
        return out

    # ------------------------------------------------------------ sources
    def available_sources(self) -> list[dict]:
        items = [{"spec": "simulate", "label": "Simulate: street crossing, handheld (CC0)"},
                 {"spec": "simulate:aerial", "label": "Simulate: NASA drone pass (public domain)"}]
        for d in sources.list_uvc_devices():
            items.append({"spec": f"uvc:{d}", "label": f"USB capture {d}"})
        if not sources.list_uvc_devices():
            items.append({"spec": "uvc:0", "label": "USB capture #0"})
        items.append({"spec": "gopro-usb", "label": "GoPro USB webcam mode"})
        cur = self.pipeline.source.spec.raw
        if not any(i["spec"] == cur for i in items):
            items.insert(0, {"spec": cur, "label": cur})
        return items

    def switch_source(self, spec: str) -> dict:
        s = sources.parse_spec(spec)
        if s.kind == "file" and not Path(s.target).expanduser().exists():
            return {"ok": False, "error": f"file not found: {s.target}"}
        src = make_source(spec, self.settings)
        self.pipeline.switch_source(src)
        self.pipeline.counter.reset()
        return {"ok": True, "source": src.spec.raw}

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        p = self.pipeline
        dets, shape = p.current_detections()
        rec = p.recorder
        return {
            "mode": "local+gemini" if self.gemini.enabled else "local",
            "source": p.source.status(),
            "detector": p.detector.info(),
            "stats": {
                "render_fps": p.stats.render_fps, "detect_fps": p.stats.detect_fps,
                "detect_ms": p.stats.detect_ms, "detect_gap_ms": p.stats.detect_gap_ms,
                "frames_per_detection": p.stats.skip_ratio,
            },
            "counts": summary.count_by_label(dets),
            "totals": p.counter.snapshot(),
            "live_summary": summary.describe(dets, shape[1] or 1) if shape[1] else "",
            "last_summary": self.last_summary,
            "recording": {"active": rec.active, "file": rec.path.name if rec.path else None,
                          "seconds": round(time.time() - rec.started) if rec.active else 0,
                          "max_seconds": rec.max_seconds},
            "gemini": {**self.gemini.status(), "periodic_seconds": self.periodic_seconds},
            "tak": self.tak.status(),
            "mesh": self.mesh.status(),
            "default_position": self.settings.default_position,
            "callsign": self.settings.callsign,
            "reports": self.reports[:5],
        }
