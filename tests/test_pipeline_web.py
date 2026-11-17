import time

import pytest
from fastapi.testclient import TestClient

from fpv_vision import sources, summary
from fpv_vision.cot import CotSender
from fpv_vision.detector import Detection, NullDetector, resolve_model
from fpv_vision.gemini import GeminiDescriber, RateLimited
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline, TrackCounter
from fpv_vision.service import Service
from fpv_vision.vlm import SceneDescriber
from fpv_vision.web.app import create_app

from conftest import FakeDetector, wait_for


def test_parse_spec():
    assert sources.parse_spec("uvc:0").kind == "uvc"
    assert sources.parse_spec("/dev/video2").target == "/dev/video2"
    assert sources.parse_spec("2").kind == "uvc"
    assert sources.parse_spec("rtsp://10.0.0.5:8554/live").kind == "network"
    assert sources.parse_spec("rtmp://x/live").kind == "network"
    assert sources.parse_spec("file:/tmp/a.mp4").target == "/tmp/a.mp4"
    assert sources.parse_spec("file:/tmp/a.mp4").kind == "replay"
    assert sources.parse_spec("clip.mp4").kind == "replay"
    assert sources.parse_spec("replay:/a.mp4|/b").target == "/a.mp4|/b"
    assert sources.parse_spec("camera:iPhone").kind == "camera"
    assert sources.parse_spec("screen").target == "1" and sources.parse_spec("screen:2").target == "2"
    assert sources.parse_spec("gopro-usb").kind == "gopro-usb"
    assert sources.parse_spec("gopro-usb:172.21.153.51").target == "172.21.153.51"
    s = sources.parse_spec("simulate:aerial")
    assert s.kind == "simulate" and s.target == "aerial"


def test_resolve_model_prefers_ncnn(tmp_path):
    (tmp_path / "yolo26n.pt").write_bytes(b"x")
    assert resolve_model("yolo26n", tmp_path) == (str(tmp_path / "yolo26n.pt"), "pytorch")
    (tmp_path / "yolo26n.onnx").write_bytes(b"x")
    assert resolve_model("yolo26n", tmp_path)[1] == "onnx"
    (tmp_path / "yolo26n_ncnn_model").mkdir()
    assert resolve_model("yolo26n", tmp_path)[1] == "ncnn"


def test_track_counter_counts_unique_tracks_once():
    c = TrackCounter(min_hits=3)
    p = Detection(0, "person", 0.9, (0, 0, 1, 1), 7)
    for _ in range(10):
        c.update([p])
    c.update([Detection(0, "person", 0.9, (0, 0, 1, 1), 8)])  # only 1 hit: not counted (flicker)
    assert c.snapshot() == {"person": 1}
    for _ in range(3):
        c.update([Detection(2, "car", 0.9, (0, 0, 1, 1), 9)])
    assert c.snapshot() == {"person": 1, "car": 1}
    c.reset()
    assert c.snapshot() == {}


def test_track_class_is_voted_once_per_track():
    """A white van flickering truck/bus/train is one truck: one class per track, seen once."""
    c = TrackCounter(min_hits=3)
    van = [("truck", 7, 0.55), ("bus", 5, 0.5), ("train", 6, 0.45), ("truck", 7, 0.6), ("bus", 5, 0.4),
           ("truck", 7, 0.5), ("train", 6, 0.3), ("bus", 5, 0.45)]
    for label, cls_id, conf in van:
        out = c.update([Detection(cls_id, label, conf, (0, 0, 9, 9), 11)])
    assert c.snapshot() == {"truck": 1}
    assert out[0].label == "truck" and out[0].cls_id == 7  # boxes and "in view" use the voted class too

    # a low-confidence bus wins the vote but is folded into truck; a confident bus stays a bus
    for _ in range(5):
        c.update([Detection(5, "bus", 0.5, (0, 0, 9, 9), 12), Detection(5, "bus", 0.8, (0, 0, 9, 9), 13)])
    assert c.snapshot() == {"truck": 2, "bus": 1}
    assert sum(c.snapshot().values()) == 3  # "seen" = unique track IDs

    # untracked low-confidence train is shown as a truck
    assert c.update([Detection(6, "train", 0.4, (0, 0, 9, 9), None)])[0].label == "truck"


def test_local_summary():
    dets = FakeDetector().detect(__import__("numpy").zeros((540, 960, 3), "uint8"))
    text = summary.describe(dets, 960)
    assert text.startswith("2 people near a car, left of frame")
    assert "1 truck right" in text
    assert summary.describe([], 960).startswith("No people")
    assert summary.size_text(dets) == "2 pax, 2 veh"


def test_file_source_loops(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "tiny.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    src = sources.VideoSource(f"file:{path}").start()
    try:
        assert wait_for(lambda: src.latest()[0] > 12, timeout=5)  # more frames than the file has
        assert src.status()["connected"]
    finally:
        src.stop()


def test_missing_file_reports_error_and_does_not_crash(tmp_path):
    src = sources.VideoSource(f"file:{tmp_path}/nope.mp4").start()
    try:
        assert wait_for(lambda: "nope.mp4" in src.status()["error"], timeout=3)
        assert not src.status()["connected"]
    finally:
        src.stop()


def test_gemini_rate_limit_and_disabled():
    g = GeminiDescriber("", "gemini-x")
    assert not g.enabled
    with pytest.raises(RuntimeError):
        g.describe(b"jpeg")

    g = GeminiDescriber("key", "gemini-x", min_interval=30)

    class FakeModels:
        def generate_content(self, model, contents):
            assert model == "gemini-x"
            assert contents[0].inline_data.mime_type == "image/jpeg"
            return type("R", (), {"text": " 2 people near a vehicle by the treeline. "})()

    g._client = type("C", (), {"models": FakeModels()})()
    assert g.describe(b"jpeg") == "2 people near a vehicle by the treeline."
    with pytest.raises(RateLimited):
        g.describe(b"jpeg")


@pytest.fixture
def app_client(settings):
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic")).start()
    gemini = GeminiDescriber("", settings.gemini_model)
    tak = CotSender(enabled=False).start()
    mesh = MeshBus(enabled=False).start()
    service = Service(settings, pipeline, SceneDescriber("auto", None, gemini), tak, mesh,
                      detector_factory=FakeDetector).start()
    client = TestClient(create_app(service))
    assert wait_for(lambda: pipeline.latest_jpeg()[1] and pipeline.current_detections()[0])
    yield client, service
    service.stop()
    pipeline.stop()


def test_status_and_counts(app_client):
    client, service = app_client
    assert wait_for(lambda: client.get("/api/status").json()["totals"].get("person") == 2)
    s = client.get("/api/status").json()
    assert s["mode"] == "simulate" and s["scene_backend"] == "off"
    assert s["heartbeat"]["tool"] == "fpv-vision" and s["heartbeat"]["status"] in ("ok", "degraded")
    assert s["counts"] == {"person": 2, "car": 1, "truck": 1}
    assert s["source"]["connected"]
    assert s["live_summary"].startswith("2 people near a car")
    assert client.post("/api/counts/reset").json()["ok"]


def test_index_and_frame(app_client):
    client, _ = app_client
    html = client.get("/").text
    assert "<h1>FPV vision</h1>" in html and 'id="isr-status"' in html
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_describe_local_and_gallery(app_client):
    client, _ = app_client
    r = client.post("/api/describe", json={"mode": "auto"}).json()
    assert r["ok"] and r["snapshot"]["summary_mode"] == "local"
    assert "people near a car" in r["snapshot"]["summary"]
    r2 = client.post("/api/describe", json={"mode": "gemini"}).json()
    assert "No scene model configured" in r2["warning"]
    snaps = client.get("/api/snapshots").json()
    assert len(snaps) == 2
    img = client.get(f"/media/snapshots/{snaps[0]['image']}")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert client.get("/media/snapshots/..%2Fsecret").status_code == 404
    assert client.post("/api/describe", json={"mode": "bogus"}).status_code == 400


def test_spot_report_api(app_client):
    client, service = app_client
    r = client.post("/api/spot-report", json={"position": "15T WG 00000 49776", "activity": "static"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["report"]["grid"] == "15T WG 00000 49776"
    assert rep["report"]["size"] == "2 pax, 2 veh"
    assert rep["cot"]["queued"] is False and rep["cot"]["connected"] is False  # TAK disabled in tests
    assert rep["report"]["provenance"] == "simulated"  # synthetic source
    assert rep["report"]["remarks"].startswith("SIMULATED: ")
    assert rep["mesh"]["text"].startswith("[SIM] SPOTREP")
    assert rep["mesh"]["sent"] is False
    assert len(rep["mesh"]["text"]) < 200
    # using an existing snapshot
    name = client.post("/api/snapshot").json()["name"]
    r = client.post("/api/spot-report", json={"position": "42.0,-93.0", "snapshot": name})
    assert r.status_code == 200 and r.json()["snapshot"] == name
    assert client.post("/api/spot-report", json={"position": "nowhere"}).status_code == 400


def test_record_clip(app_client):
    client, _ = app_client
    name = client.post("/api/record/start").json()["file"]
    time.sleep(1.2)
    assert client.post("/api/record/stop").json()["file"] == name
    clips = client.get("/api/clips").json()
    assert clips and clips[0]["name"] == name and clips[0]["bytes"] > 1000
    r = client.get(f"/media/clips/{name}")
    assert r.status_code == 200


def test_switch_source(app_client):
    client, service = app_client
    assert client.post("/api/source", json={"spec": "file:/does/not/exist.mp4"}).json()["ok"] is False
    assert client.post("/api/source", json={"spec": "synthetic"}).json()["ok"]
    assert wait_for(lambda: service.pipeline.source.status()["connected"])
    assert any(s["spec"] == "simulate" for s in client.get("/api/sources").json()["sources"])


def test_periodic_requires_gemini(app_client):
    client, _ = app_client
    assert client.post("/api/periodic", json={"seconds": 30}).json()["periodic_seconds"] == 0


def test_null_detector_pipeline_streams(settings):
    p = Pipeline(settings, NullDetector("test"), sources.VideoSource("synthetic")).start()
    try:
        assert wait_for(lambda: p.latest_jpeg()[1])
        assert p.current_detections()[0] == []
    finally:
        p.stop()


# sha256 of ui-kit v2 (isr-ui.css, isr-ui.js, fonts/): the served copies must be the kit files, unchanged
ISR_UI_SHA256 = {
    "isr-ui.css": "ed601e0494d7b8c93f246205d53a03916b15ceb7eb3a54526a6cd4a447879ec7",
    "isr-ui.js": "191f2b6dce0c52a760a8fdeddf4d5f127399d7d3e1684dcc26a5dbe78874d274",
    "fonts/barlow-condensed-500.woff2": "31ff5e6e96a096fde4d78bc838719e69b093db2d0b062b96e423c065a6d1b2b1",
    "fonts/barlow-condensed-600.woff2": "5bc49d41f32810a114765314392332ce2acae2276635eee265a50319a12c1b49",
    "fonts/ibm-plex-mono-400.woff2": "52de70091213c9eab340e2a7ac59e7b6b2b94e089bf69ef38479b22cfde4274c",
    "fonts/ibm-plex-mono-500.woff2": "f17f51948d10b67d907393ccbac551912f2c2bd2c5af19be3aabe2f456f8fe54",
    "fonts/OFL-BarlowCondensed.txt": "186d750eb496a4c17a76385f82be6aea2ac1cf2de074a811d63786cf374ea73f",
    "fonts/OFL-IBMPlexMono.txt": "7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da",
}


def test_page_uses_shared_isr_ui_kit(app_client):
    import hashlib

    client, _ = app_client
    html = client.get("/").text
    assert '<link rel="stylesheet" href="/isr-ui.css?v=2">' in html and '<script src="/isr-ui.js?v=2"></script>' in html
    for name, sha in ISR_UI_SHA256.items():
        r = client.get("/" + name)
        assert r.status_code == 200 and hashlib.sha256(r.content).hexdigest() == sha, name
    assert "text/css" in client.get("/isr-ui.css?v=2").headers["content-type"]
    assert client.get("/fonts/ibm-plex-mono-400.woff2").headers["content-type"] == "font/woff2"
    assert client.get("/fonts/..%2F..%2Fapp.py").status_code == 404
