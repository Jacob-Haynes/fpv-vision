"""Glue between pipeline, scene describer (VLM/Gemini), TAK, mesh and analysis — what the web API calls."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path

from fpv_vision import cot, geo, sources, spotrep, summary
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
                 detector_factory=None, air_factory=None, namer_client=None,
                 search_factory=None):
        self.settings = settings
        self.pipeline = pipeline
        self.describer = describer
        self.tak = tak
        self.mesh = mesh
        self.detector_factory = detector_factory
        self.periodic_seconds = settings.gemini_periodic_seconds if describer.enabled else 0.0
        self.last_summary: dict | None = None
        self.reports: list[dict] = []
        self._markers: dict[str, dict] = {}  # uid -> spot_report_cot kwargs, for POST /api/demo/clear
        self.jobs: dict = {}
        self._stop = threading.Event()
        self._periodic_thread: threading.Thread | None = None
        self.started = time.time()
        # Air watch
        self.air_factory = air_factory or self._default_air_factory
        self.search_factory = search_factory or self._default_search_factory
        self.search_error = ""
        self.air_error = ""
        self.air_events: list[dict] = []  # newest first: what was reported for each new air track
        pipeline.on_air_track = self._on_air_track
        self.namer_client = namer_client  # tests inject a fake; default: VLM_URL / VLM_MODEL

    def start(self) -> "Service":
        self._periodic_thread = threading.Thread(target=self._periodic_loop, name="periodic", daemon=True)
        self._periodic_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for j in self.jobs.values():
            j.cancel()

    # ------------------------------------------------------------ Search
    def _default_search_factory(self):
        from fpv_vision.search import SearchDetector

        det = self.pipeline.detector
        if not getattr(det, "model_path", None):
            raise RuntimeError("no detector model loaded")
        return SearchDetector.from_detector(det, self.settings)

    def set_search(self, on: bool) -> dict:
        """Turn Search (tiled ground detection) on/off. Loads its own copy of the weights on first use."""
        if on and self.pipeline.search is None:
            try:
                self.pipeline.search = self.search_factory()
                self.search_error = ""
            except Exception as e:  # noqa: BLE001
                self.search_error = str(e)
                log.warning("search unavailable: %s", e)
                return {"ok": False, "search": False, "error": self.search_error}
        self.pipeline.set_search(on)
        return {"ok": True, "search": self.pipeline.search_on}

    def search_status(self) -> dict:
        p = self.pipeline
        s = self.settings
        return {"on": p.search_on, "loaded": p.search is not None, "error": self.search_error,
                "fps": p.stats.detect_fps if p.search_on else 0.0, "ms": p.stats.detect_ms if p.search_on else 0.0,
                "model": p.search.info() if p.search is not None else None,
                "conf": s.search_conf, "hits": s.search_hits, "air_fps_cap": s.search_air_fps}

    # ------------------------------------------------------------ Air watch
    def _default_air_factory(self):
        from fpv_vision import air
        from fpv_vision.detector import select_device

        return air.AirDetector(self.settings.models_dir, select_device(self.settings.device))

    def set_air(self, on: bool) -> dict:
        """Turn Air watch on/off. Only with FPV_AIR=1: air detection belongs to OP sentry, so FPV
        vision never offers it in the UI and never loads YOLOE otherwise. The model loads on first use."""
        if on and not self.settings.air:
            self.air_error = "Air watch is off in FPV vision (air detection is OP sentry's); set FPV_AIR=1"
            return {"ok": False, "air": False, "error": self.air_error}
        if on and self.pipeline.air is None:
            try:
                self.pipeline.air = self.air_factory()
                self.air_error = ""
            except Exception as e:  # noqa: BLE001 - missing weights offline is expected
                self.air_error = str(e)
                log.warning("air watch unavailable: %s", e)
                return {"ok": False, "air": False, "error": self.air_error}
        if on and self.pipeline.air_namer is None:
            self._start_namer()
        self.pipeline.set_air(on)
        return {"ok": True, "air": self.pipeline.air_on}

    def _start_namer(self) -> None:
        """The local vision model names air tracks (fpv_vision.air_namer); off when no VLM_URL."""
        from fpv_vision.air_namer import AirNamer, VlmClient

        client = self.namer_client
        if client is None:
            if not self.settings.vlm_url:
                return
            client = VlmClient(self.settings.vlm_url, self.settings.vlm_model, self.settings.vlm_api_key, timeout=10.0)
        self.pipeline.air_namer = AirNamer(client, self.pipeline.air_counter, on_named=self._on_air_named,
                                           min_conf=self.settings.air_name_conf).start()

    def current_provenance(self) -> str:
        return self.provenance_of({"source": self.pipeline.source.spec.raw})

    def _air_cot(self, d, hostile: bool = False) -> dict:
        from fpv_vision import air

        typ = air.cot_type(d.label, hostile)
        if typ is None:
            return {"queued": False, "error": "not reported"}
        lat_lon = None
        pos = self.default_position()
        if pos:
            try:
                lat_lon = geo.parse_position(pos)
            except (ValueError, geo.PositionError):
                lat_lon = None
        if not lat_lon:
            return {"queued": False, "error": "no observer position (set SENSOR_LAT/SENSOR_LON)"}
        tid = d.track_id % air.TRACK_ID_OFFSET
        prov = self.current_provenance()
        what = ("hostile " if hostile else "") + d.label + (f" ({d.note}, named by the vision model)" if d.note else "")
        remarks = (f"{spotrep.PROVENANCE_COT.get(prov, '')}Air watch: {what} seen by {self.tak.callsign} at "
                   f"{spotrep.dtg()}, track {tid}, confidence {d.conf:.2f}. The position is the observer's, "
                   "not the target's: no range or bearing.")
        marker = dict(lat=lat_lon[0], lon=lat_lon[1], callsign=f"{d.label.capitalize()} {tid}", remarks=remarks,
                      uid=f"{self.tak.uid}-air-{tid}", cot_type=typ, stale_s=300, ce=1000.0,
                      sender_uid=self.tak.uid, sender_callsign=self.tak.callsign, how="m-g")
        queued = self.tak.send(cot.spot_report_cot(**marker))
        if queued:
            self._markers[marker["uid"]] = marker
        return {"queued": queued, "type": typ, "error": None if queued else self.tak.last_error}

    def _on_air_track(self, d) -> None:
        """A new air track was counted: CoT for drones, helicopters and aeroplanes; mesh alert for drones."""
        from fpv_vision import air

        ev = {"ts": time.time(), "track": d.track_id % air.TRACK_ID_OFFSET, "label": d.label,
              "conf": round(d.conf, 2), "provenance": self.current_provenance()}
        if d.label != "bird":
            ev["cot"] = self._air_cot(d)
        if d.label == "drone":
            text = (f"{spotrep.PROVENANCE_MESH.get(ev['provenance'], '')}Air watch: drone seen by "
                    f"{self.tak.callsign} {spotrep.dtg()}")[:199]
            sent = self.mesh.send(text)
            ev["mesh"] = {"sent": sent, "text": text, "error": None if sent else self.mesh.last_error}
        self.air_events = [ev, *self.air_events][:20]
        log.info("air watch: new %s track %s %s", d.label, ev["track"], {k: ev[k] for k in ("cot", "mesh") if k in ev})

    def _on_air_named(self, tid: int, old: str | None, new: str, detail: str) -> None:
        """The vision model renamed a track: retype its CoT (same uid), and alert on mesh if it is now a drone."""
        from fpv_vision import air
        from fpv_vision.detector import Detection

        p = self.pipeline
        cur = next((x for x in p.air_detections() if x.track_id == tid), None)
        cls_id = p.air_counter.voted(tid)[0] if tid in p.air_counter._votes else air.AIR_CONTACT
        d = Detection(cls_id, new, cur.conf if cur else 0.0, cur.xyxy if cur else (0, 0, 0, 0), tid, note=detail,
                      named_by="ai")
        ev = {"ts": time.time(), "track": tid % air.TRACK_ID_OFFSET, "label": new, "detail": detail, "renamed_from": old,
              "provenance": self.current_provenance(), "named_by": "ai"}
        if tid in p.air_counter._counted and new != "bird":  # its first CoT went out already: update it
            ev["cot"] = self._air_cot(d, hostile=tid in p.air_hostile)
        if new == "drone" and old != "drone" and tid in p.air_counter._counted:
            text = (f"{spotrep.PROVENANCE_MESH.get(ev['provenance'], '')}Air watch: drone ({detail or 'named by AI'}) "
                    f"seen by {self.tak.callsign} {spotrep.dtg()}")[:199]
            sent = self.mesh.send(text)
            ev["mesh"] = {"sent": sent, "text": text, "error": None if sent else self.mesh.last_error}
        self.air_events = [ev, *self.air_events][:20]
        log.info("air watch: track %s named %s (%s), was %s", ev["track"], new, detail, old)

    def mark_air(self, track: int, hostile: bool) -> dict:
        """Operator marks an air track hostile (red box, hostile CoT for a drone) or clears it."""
        if not self.settings.air:
            return {"ok": False, "error": "Air watch is off in FPV vision; set FPV_AIR=1"}
        from fpv_vision import air

        tid = int(track) % air.TRACK_ID_OFFSET + air.TRACK_ID_OFFSET
        (self.pipeline.air_hostile.add if hostile else self.pipeline.air_hostile.discard)(tid)
        d = next((x for x in self.pipeline.air_detections() if x.track_id == tid), None)
        out = {"ok": True, "track": tid % air.TRACK_ID_OFFSET, "hostile": hostile}
        if d is not None and d.label != "bird":
            out["cot"] = self._air_cot(d, hostile=hostile)
        return out

    def air_status(self) -> dict:
        from fpv_vision import air

        p = self.pipeline
        dets = p.air_detections()
        return {
            "enabled": self.settings.air,  # FPV_AIR=1; never offered in the UI
            "on": p.air_on, "loaded": p.air is not None,
            "available": p.air is not None or air.available(self.settings.models_dir),
            "error": self.air_error, "fps": p.air_stats["fps"], "ms": p.air_stats["ms"],
            "model": p.air.info() if p.air is not None else None,
            "tracks": [{"track": d.track_id % air.TRACK_ID_OFFSET, "label": d.label, "conf": round(d.conf, 2),
                        "note": d.note, "named_by": d.named_by,
                        "hostile": d.track_id in p.air_hostile} for d in dets if d.track_id is not None],
            "naming": p.air_namer.status() if p.air_namer is not None else
            {"enabled": False, "reason": "no VLM_URL: YOLOE names air tracks"},
            "totals": p.air_counter.snapshot(),
            "events": self.air_events[:5],
        }

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
            src = self.pipeline.source
            if src.spec.is_live and not src.connected:
                return {"ok": False, "error": f"{src.error or 'camera not connected'}: no picture to report"}
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
            marker = dict(lat=rep["lat"], lon=rep["lon"], callsign=rep["callsign"], remarks=rep["remarks"],
                          uid=f"fpv-vision-spot-{uuid.uuid4().hex[:12]}", sender=self.settings.callsign,
                          sender_uid=getattr(self.tak, "uid", None),
                          sender_callsign=getattr(self.tak, "callsign", None))
            queued = self.tak.send(cot.spot_report_cot(**marker))
            if queued:
                self._markers[marker["uid"]] = marker
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

    # ------------------------------------------------------------ demo (ui-kit DEMO-TRIGGER-CONTRACT)
    def demo_scenarios(self) -> dict:
        """fpv-vision has no simulated scenarios: the source picker covers the clips and the live GoPro."""
        return {"mode": self.settings.demo_mode, "scenarios": [], "active": []}

    def demo_clear(self) -> dict:
        """Remove every spot report (and Air watch marker) this run put on the map: each marker once
        more with stale = now, then a t-x-d-d __forcedelete, then our SA again (OTS marks the sender
        disconnected on a t-x-d-d). The report history on this page is kept."""
        markers, self._markers = self._markers, {}
        for m in markers.values():
            self.tak.send(cot.spot_report_cot(**{**m, "stale_s": 0}), force=True)
            self.tak.send(cot.delete_cot(target_uid=m["uid"], target_type=m.get("cot_type", "a-u-G"),
                                         lat=m["lat"], lon=m["lon"], sender=self.settings.callsign), force=True)
        if markers and hasattr(self.tak, "resend_sa"):
            self.tak.resend_sa()
        return {"ok": True, "cleared": len(markers)}

    def default_position(self) -> str:
        """Operator default (FPV_DEFAULT_POSITION), else shared SENSOR_LAT/SENSOR_LON. Never 0,0."""
        if self.settings.default_position:
            return self.settings.default_position
        lat, lon = self.settings.sensor_lat, self.settings.sensor_lon
        if lat is not None and lon is not None and not (lat == 0 and lon == 0):
            return f"{lat:.6f}, {lon:.6f}"
        return ""

    def default_grid(self) -> str:
        """The default position as a spaced MGRS grid (British usage), or "" if none or unparseable."""
        pos = self.default_position()
        if not pos:
            return ""
        try:
            lat, lon = geo.parse_position(pos)
            return geo.format_mgrs(geo.to_mgrs(lat, lon))
        except (ValueError, geo.PositionError):
            return ""

    def set_running(self, on: bool) -> dict:
        """Stop/start detection and the stream (the source stays open; a clip is paused too)."""
        self.pipeline.running = bool(on)
        src = self.pipeline.source
        if src.is_replay:
            src.pause(not on)
        return {"ok": True, "running": self.pipeline.running}

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
        from fpv_vision import sample

        def sim(spec, name, label):
            cached = sample.cached_sample(self.settings.data_dir, name) is not None
            return {"spec": spec, "label": label + ("" if cached else " (not downloaded: test pattern)"),
                    "group": "simulate", "cached": cached}

        items = [sim("simulate", "street", "Simulate: street crossing, handheld (CC0)"),
                 sim("simulate:aerial", "aerial", "Simulate: NASA drone pass (public domain)")]
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
        self.pipeline.running = True
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
                "detail": self.detail(), "demo_mode": self.settings.demo_mode, "demo_active": []}

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
            "totals": {**p.counter.snapshot(), **p.air_counter.snapshot()},
            "air": self.air_status(),
            "search": self.search_status(),
            "demo": self.demo_scenarios(),
            "live_summary": summary.describe(dets, shape[1] or 1) if shape[1] else "",
            "last_summary": self.last_summary,
            "hud": self.settings.show_hud,
            "recording": {"active": rec.active, "file": rec.path.name if rec.path else None,
                          "seconds": round(time.time() - rec.started) if rec.active else 0,
                          "max_seconds": rec.max_seconds},
            "vlm": {**vlm, "periodic_seconds": self.periodic_seconds},
            "tak": self.tak.status(),
            "mesh": self.mesh.status(),
            "heartbeat": self.heartbeat(),
            "camera_permission": sources.camera_permission_status(),
            "default_position": self.default_position(),
            "default_grid": self.default_grid(),
            "running": self.pipeline.running,
            "callsign": self.settings.callsign,
            "reports": self.reports[:5],
            "jobs": [j.to_dict(full=False) for j in list(self.jobs.values())[-3:]],
        }
