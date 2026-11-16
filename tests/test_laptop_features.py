"""Replay-as-live, crop, post-flight analysis, VLM fallback chain, heartbeat, device selection."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from fpv_vision import detector as det_mod
from fpv_vision import sources
from fpv_vision.config import Settings
from fpv_vision.gemini import GeminiDescriber
from fpv_vision.mesh import HEARTBEAT_TOPIC, MeshBus
from fpv_vision.vlm import OpenAICompatDescriber, SceneDescriber

from conftest import FakeDetector, wait_for


def make_clip(path, n=60, fps=30, size=(160, 90)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        f = np.full((size[1], size[0], 3), (i * 4) % 255, np.uint8)
        cv2.putText(f, str(i), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(f)
    w.release()
    return path


# ------------------------------------------------------------------ replay


def test_expand_playlist(tmp_path):
    a = make_clip(tmp_path / "a.avi", n=5)
    sub = tmp_path / "card" / "DCIM"
    sub.mkdir(parents=True)
    b = make_clip(sub / "b.avi", n=5)
    (sub / "notes.txt").write_text("x")
    assert sources.expand_playlist(f"{a}|{tmp_path / 'card'}") == [str(a), str(b)]
    assert sources.expand_playlist("/nope|") == []


def test_replay_is_wall_clock_paced_and_loops(tmp_path):
    clip = make_clip(tmp_path / "c.avi", n=30, fps=30)  # 1 s long
    src = sources.VideoSource(f"replay:{clip}").start()
    try:
        assert wait_for(lambda: src.connected, timeout=3)
        t0 = time.time()
        f0 = src.latest()[0]
        time.sleep(1.5)
        rate = (src.latest()[0] - f0) / (time.time() - t0)
        assert 20 < rate < 40  # ~30 fps: real time, not as fast as possible
        st = src.status()
        assert st["replay"] and st["replay_state"]["fps"] == 30 and st["replay_state"]["files"] == ["c.avi"]
        assert src.latest()[0] > 35  # looped past the end of a 30-frame file
    finally:
        src.stop()


def test_replay_seek_pause_and_next(tmp_path):
    a = make_clip(tmp_path / "a.avi", n=300, fps=30)  # 10 s
    b = make_clip(tmp_path / "b.avi", n=300, fps=30)
    src = sources.VideoSource(f"replay:{a}|{b}").start()
    try:
        assert wait_for(lambda: src.connected)
        src.seek(7.0)
        assert wait_for(lambda: src.replay["pos"] >= 7.0, timeout=3)
        src.pause(True)
        assert wait_for(lambda: src.replay["paused"], timeout=2)
        pos = src.replay["pos"]
        time.sleep(0.5)
        assert src.replay["pos"] == pos
        src.pause(False)
        src.next_file()
        assert wait_for(lambda: src.replay["file"] == "b.avi", timeout=3)
    finally:
        src.stop()


def test_crop_is_applied_to_frames():
    src = sources.VideoSource("synthetic").start()
    try:
        src.set_crop((0.25, 0.5, 0.5, 0.5))
        assert wait_for(lambda: src.latest()[1] is not None and src.latest()[1].shape[:2] == (270, 480))
        src.set_crop(None)
        assert wait_for(lambda: src.latest()[1].shape[:2] == (540, 960))
        src.set_crop((0.1, 0.1, 0.001, 0.5))  # degenerate -> cleared
        assert src.crop is None
    finally:
        src.stop()


def test_replay_missing_files_reports_error(tmp_path):
    src = sources.VideoSource(f"replay:{tmp_path}/none.mp4").start()
    try:
        assert wait_for(lambda: "no video files" in src.status()["error"], timeout=3)
    finally:
        src.stop()


# ------------------------------------------------------------------ analysis


def test_post_flight_analysis_job(settings, tmp_path):
    from fpv_vision.analysis import AnalysisJob, pick_best
    from fpv_vision.pipeline import Pipeline

    clip = make_clip(tmp_path / "flight.avi", n=150, fps=30)  # 5 s
    p = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic"))
    job = AnalysisJob(str(clip), p, FakeDetector, sample_fps=5).start()
    assert wait_for(lambda: job.state in ("done", "error"), timeout=20)
    d = job.to_dict()
    assert d["state"] == "done", d["error"]
    assert d["progress"] == 1.0 and d["total_s"] == pytest.approx(5.0, abs=0.1)
    assert len(d["timeline"]) == 5 and d["timeline"][0]["person"] == 2 and d["timeline"][0]["vehicle"] == 2
    assert d["totals"] == {"person": 2, "car": 1, "truck": 1}  # tracked once each, not per frame
    assert 1 <= len(d["best"]) <= 2  # 3 s minimum spacing in a 5 s clip
    snap = p.load_snapshot(d["best"][0]["name"])
    assert snap["video_file"] == "flight.avi" and snap["note"] == "post-flight"
    assert (settings.snapshots_dir / snap["image"]).exists()
    assert list((settings.data_dir / "analysis").glob("analysis_*.json"))
    # pick_best spacing
    cands = [{"file": "f", "t": t, "score": s} for t, s in ((0, 5), (1, 6), (5, 4), (9, 1))]
    assert [c["t"] for c in pick_best(cands, k=3, min_gap=3)] == [1, 5, 9]


# ------------------------------------------------------------------ VLM chain


class _FakeOllama(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": []}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.requests.append((self.path, body))
        out = {"choices": [{"message": {"content": "<think>hmm</think> 3 people beside a pickup at the treeline."}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


@pytest.fixture
def fake_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_openai_compat_backend(fake_ollama):
    vlm = OpenAICompatDescriber(fake_ollama + "/v1", "qwen3-vl:8b", min_interval=0.5)
    assert vlm.base == fake_ollama  # path stripped: contract says bare base URL
    assert vlm.probe()
    text = vlm.describe(b"\xff\xd8jpeg", context="2 people")
    assert text == "3 people beside a pickup at the treeline."
    path, body = _FakeOllama.requests[-1]
    assert path == "/v1/chat/completions" and body["model"] == "qwen3-vl:8b"
    img = body["messages"][0]["content"][1]
    assert img["type"] == "image_url" and img["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_scene_chain_falls_back_to_gemini_then_off():
    dead = OpenAICompatDescriber("http://127.0.0.1:9", "qwen3-vl:8b")  # nothing listens on port 9
    gem = GeminiDescriber("key", "gemini-x", min_interval=1)

    class Models:
        def generate_content(self, model, contents):
            return type("R", (), {"text": "cloud answer"})()

    gem._client = type("C", (), {"models": Models()})()
    chain = SceneDescriber("openai_compat", dead, gem)
    assert chain.status()["chain"] == ["openai_compat", "gemini", "off"]
    text, backend = chain.describe(b"jpeg")
    assert (text, backend) == ("cloud answer", "gemini")
    # no key -> only the local VLM, and when it is down we raise so the caller shows local summary
    chain2 = SceneDescriber("openai_compat", dead, GeminiDescriber("", "g"))
    with pytest.raises(RuntimeError):
        chain2.describe(b"jpeg")
    assert SceneDescriber("auto", None, GeminiDescriber("", "g")).selected == "off"
    assert SceneDescriber("auto", dead, GeminiDescriber("", "g")).selected == "openai_compat"


# ------------------------------------------------------------------ heartbeat / device


class _Info:
    rc = 0

    def wait_for_publish(self, timeout=None):
        return True


class _Client:
    def __init__(self):
        self.pub = []

    def publish(self, topic, body, qos=0, retain=False):
        self.pub.append((topic, json.loads(body), retain))
        return _Info()

    def disconnect(self):
        pass

    def loop_stop(self):
        pass


def test_heartbeat_contract_and_stopped_on_exit():
    hb = {"tool": "fpv-vision", "host": "isr-droneops", "url": "http://isr-droneops.local:8110/",
          "mode": "mps", "status": "ok", "ts": 0}
    bus = MeshBus(enabled=True, heartbeat_fn=lambda: dict(hb))
    bus._client, bus.connected = _Client(), True
    assert bus._publish_heartbeat()
    topic, body, retain = bus._client.pub[-1]
    assert topic == HEARTBEAT_TOPIC == "isrkit/heartbeat/fpv-vision" and retain
    assert {"tool", "host", "url", "mode", "status", "ts"} <= set(body) and body["ts"] > 0
    bus.stop()
    assert bus._client.pub[-1][1]["status"] == "stopped" and bus._client.pub[-1][2]


def test_mesh_send_disabled_but_heartbeat_allowed():
    bus = MeshBus(enabled=True, mesh_enabled=False, heartbeat_fn=lambda: {"tool": "fpv-vision"})
    bus._client, bus.connected = _Client(), True
    assert bus.send("hi") is False
    assert bus._publish_heartbeat()


def test_device_and_model_selection(monkeypatch):
    s = Settings()
    assert det_mod.choose_model(s, "cpu") == ("yolo26n", 416)
    assert det_mod.choose_model(s, "mps") == ("yolo26s", 640)
    assert det_mod.choose_model(s, "cuda") == ("yolo26s", 640)
    s.model, s.imgsz = "yolo26m", 960
    assert det_mod.choose_model(s, "mps") == ("yolo26m", 960)
    assert det_mod.select_device("cpu") == "cpu"
    assert det_mod.select_device("auto") in ("cpu", "mps", "cuda")
    assert det_mod.accelerator_of("0") == "cuda" and det_mod.accelerator_of("mps") == "mps"


def test_resolve_model_gpu_prefers_pytorch(tmp_path):
    (tmp_path / "yolo26s.pt").write_bytes(b"x")
    (tmp_path / "yolo26s_ncnn_model").mkdir()
    assert det_mod.resolve_model("yolo26s", tmp_path)[1] == "ncnn"
    assert det_mod.resolve_model("yolo26s", tmp_path, prefer_pytorch=True)[1] == "pytorch"


def test_service_heartbeat_modes(settings, monkeypatch):
    from fpv_vision.cot import CotSender
    from fpv_vision.pipeline import Pipeline
    from fpv_vision.service import Service

    monkeypatch.setenv("ISR_HOST", "ISR-DroneOps")
    p = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic"))
    svc = Service(settings, p, SceneDescriber("off", None, GeminiDescriber("", "g")), CotSender(enabled=False),
                  MeshBus(enabled=False))
    hb = svc.heartbeat()
    assert hb["tool"] == "fpv-vision" and hb["host"] == "isr-droneops"
    assert hb["url"] == "http://isr-droneops.local:8110/" and hb["mode"] == "simulate"
    assert hb["detail"].endswith("local summary") and isinstance(hb["detail"], str)
    p.source = sources.VideoSource("replay:/x.mp4")
    assert svc.mode() == "replay"
    p.source = sources.VideoSource("camera:iPhone")
    assert svc.mode() == "cpu"  # FakeDetector has no accelerator attr -> cpu


def test_duplicate_filter_drops_padded_frames_only():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    f = sources.DuplicateFilter()
    seq = []
    for i in range(10):  # a 30 fps camera padded to 60: each real frame sent twice
        real = cv2.add(base, rng.integers(0, 4, base.shape, dtype=np.uint8))  # sensor noise
        seq += [real, real.copy()]
    kept = [not f.is_duplicate(x) for x in seq]
    assert kept == [True, False] * 10
    assert f.dropped == 10


def test_duplicate_filter_keeps_noisy_static_scene_and_caps_runs():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    f = sources.DuplicateFilter()
    # real frames of a static scene still differ by sensor noise -> all kept
    noisy = [cv2.add(base, rng.integers(0, 6, base.shape, dtype=np.uint8)) for _ in range(6)]
    assert all(not f.is_duplicate(x) for x in noisy)
    # perfectly frozen picture: at most 3 in a row dropped, then one passes
    f2 = sources.DuplicateFilter(max_run=3)
    kept = [not f2.is_duplicate(base) for _ in range(9)]
    assert kept == [True, False, False, False, True, False, False, False, True]
    assert sources.DuplicateFilter(enabled=False).is_duplicate(base) is False


def test_heartbeat_host_defaults_to_gethostname(monkeypatch):
    import socket

    from fpv_vision.service import short_hostname

    monkeypatch.delenv("ISR_HOST", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "Field-Laptop.lan")
    assert short_hostname() == "field-laptop"
    monkeypatch.setenv("ISR_HOST", "isr-droneops")
    assert short_hostname() == "isr-droneops"


def test_provenance_prefixes_on_cot_and_mesh():
    from fpv_vision import spotrep

    dets = [__import__("fpv_vision.detector", fromlist=["Detection"]).Detection(0, "person", 0.9, (0, 0, 5, 9), 1)]
    for prov, cot_p, mesh_p in (("simulated", "SIMULATED: ", "[SIM] "), ("recorded", "RECORDED: ", "[REC] "),
                                ("live", "", "")):
        r = spotrep.build(dets=dets, position="51.15,-1.75", activity="x" * 400, summary_text="s",
                          callsign="FPV-VISION", provenance=prov)
        assert r["remarks"].startswith(cot_p + "SALUTE")
        assert r["mesh_text"].startswith(mesh_p + "SPOTREP") and len(r["mesh_text"]) < 200
    with pytest.raises(ValueError):
        spotrep.build(dets=dets, position="0, 0", activity="", summary_text="", callsign="X")


def test_provenance_of_snapshots():
    from fpv_vision.service import Service

    assert Service.provenance_of({"provenance": "recorded"}) == "recorded"
    assert Service.provenance_of({"source": "simulate"}) == "simulated"
    assert Service.provenance_of({"source": "/d/samples/sample_street_cc0_480p.webm"}) == "simulated"
    assert Service.provenance_of({"source": "replay:/x.mp4"}) == "recorded"
    assert Service.provenance_of({"source": "/Volumes/SD/DJI_0001.MP4", "video_file": "DJI_0001.MP4"}) == "recorded"
    assert Service.provenance_of({"source": "camera:USB3 Video"}) == "live"


def test_default_position_fallback(settings):
    from fpv_vision.cot import CotSender
    from fpv_vision.pipeline import Pipeline
    from fpv_vision.service import Service

    p = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic"))
    svc = Service(settings, p, SceneDescriber("off", None, GeminiDescriber("", "g")), CotSender(enabled=False),
                  MeshBus(enabled=False))
    assert svc.default_position() == ""
    settings.sensor_lat, settings.sensor_lon = 0.0, 0.0
    assert svc.default_position() == ""  # never 0,0
    settings.sensor_lat, settings.sensor_lon = 51.15, -1.75
    assert svc.default_position() == "51.150000, -1.750000"
    settings.default_position = "30U WB 87427 67248"
    assert svc.default_position() == "30U WB 87427 67248"
    settings.default_position, settings.sensor_lat, settings.sensor_lon = "", None, None
    p.source.start()
    try:
        assert wait_for(lambda: p.source.latest()[1] is not None)
        p._raw_frame = p._annotated = p.source.latest()[1]
        r = svc.spot_report(position="")
        assert r["ok"] is False and "no position" in r["error"]
    finally:
        p.source.stop()


def test_mesh_rate_limit_one_per_30s():
    bus = MeshBus(enabled=True)
    bus._client, bus.connected = _Client(), True
    assert bus.send("first")
    assert bus.send("second") is False and "rate limit" in bus.last_error
    bus._last_send -= 31
    assert bus.send("third")


def test_cot_rate_limit_per_uid():
    from fpv_vision import cot

    s = cot.CotSender(enabled=True)  # not started: just queues
    a = cot.spot_report_cot(lat=51.0, lon=-1.0, callsign="A", remarks="r", uid="fpv-a")
    b = cot.spot_report_cot(lat=51.0, lon=-1.0, callsign="B", remarks="r", uid="fpv-b")
    assert s.send(a) and s.send(b)
    assert s.send(a) is False and "rate limit" in s.last_error
    s._last_uid["fpv-a"] -= 6
    assert s.send(a)


def test_replay_snapshot_is_marked_recorded(settings, tmp_path):
    from fpv_vision.pipeline import Pipeline

    clip = make_clip(tmp_path / "DJI_0001.MP4.avi", n=90, fps=30)
    p = Pipeline(settings, FakeDetector(), sources.VideoSource(f"replay:{clip}")).start()
    try:
        assert wait_for(lambda: p.latest_jpeg()[1])
        snap = p.take_snapshot()
        assert snap["provenance"] == "recorded" and snap["video_file"] == "DJI_0001.MP4.avi"
    finally:
        p.stop()
