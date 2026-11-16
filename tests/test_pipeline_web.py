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
from fpv_vision.web.app import create_app

from conftest import FakeDetector, wait_for


def test_parse_spec():
    assert sources.parse_spec("uvc:0").kind == "uvc"
    assert sources.parse_spec("/dev/video2").target == "/dev/video2"
    assert sources.parse_spec("2").kind == "uvc"
    assert sources.parse_spec("rtsp://10.0.0.5:8554/live").kind == "network"
    assert sources.parse_spec("rtmp://x/live").kind == "network"
    assert sources.parse_spec("file:/tmp/a.mp4").target == "/tmp/a.mp4"
    assert sources.parse_spec("clip.mp4").kind == "file"
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
    service = Service(settings, pipeline, gemini, tak, mesh).start()
    client = TestClient(create_app(service))
    assert wait_for(lambda: pipeline.latest_jpeg()[1] and pipeline.current_detections()[0])
    yield client, service
    service.stop()
    pipeline.stop()


def test_status_and_counts(app_client):
    client, service = app_client
    assert wait_for(lambda: client.get("/api/status").json()["totals"].get("person") == 2)
    s = client.get("/api/status").json()
    assert s["mode"] == "local"
    assert s["counts"] == {"person": 2, "car": 1, "truck": 1}
    assert s["source"]["connected"]
    assert s["live_summary"].startswith("2 people near a car")
    assert client.post("/api/counts/reset").json()["ok"]


def test_index_and_frame(app_client):
    client, _ = app_client
    assert "FPV VISION" in client.get("/").text
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_describe_local_and_gallery(app_client):
    client, _ = app_client
    r = client.post("/api/describe", json={"mode": "auto"}).json()
    assert r["ok"] and r["snapshot"]["summary_mode"] == "local"
    assert "people near a car" in r["snapshot"]["summary"]
    r2 = client.post("/api/describe", json={"mode": "gemini"}).json()
    assert "not configured" in r2["warning"]
    snaps = client.get("/api/snapshots").json()
    assert len(snaps) == 2
    img = client.get(f"/media/snapshots/{snaps[0]['image']}")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert client.get("/media/snapshots/..%2Fsecret").status_code == 404
    assert client.post("/api/describe", json={"mode": "bogus"}).status_code == 400


def test_spot_report_api(app_client):
    client, service = app_client
    r = client.post("/api/spot-report", json={"position": "30U WB 87427 67248", "activity": "static"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["report"]["grid"] == "30U WB 87427 67248"
    assert rep["report"]["size"] == "2 pax, 2 veh"
    assert rep["cot"] == {"queued": False, "connected": False}  # TAK disabled in tests
    assert rep["mesh"]["sent"] is False
    assert len(rep["mesh"]["text"]) < 200
    # using an existing snapshot
    name = client.post("/api/snapshot").json()["name"]
    r = client.post("/api/spot-report", json={"position": "51.15,-1.75", "snapshot": name})
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
