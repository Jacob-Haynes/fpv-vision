"""Air watch: class mapping, tiling merge, CoT types, palette and the toggle (no model needed)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeDetector, wait_for
from fake_ots import store_rejection
from fpv_vision import air, annotate, sources
from fpv_vision.cot import CotSender
from fpv_vision.detector import Detection
from fpv_vision.gemini import GeminiDescriber
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline
from fpv_vision.service import Service
from fpv_vision.vlm import SceneDescriber
from fpv_vision.web.app import create_app


# ------------------------------------------------------------------ class mapping
def test_prompt_mapping_and_cot_types():
    assert air.map_prompt("quadcopter") == air.map_prompt("drone") == (air.DRONE, "drone")
    assert air.map_prompt("helicopter") == (air.HELICOPTER, "helicopter")
    assert air.map_prompt("Aeroplane") == (air.AEROPLANE, "aeroplane")
    assert air.map_prompt("bird") == (air.BIRD, "bird")
    assert air.cot_type("drone") == "a-u-A-M-F-Q" and air.cot_type("drone", hostile=True) == "a-h-A-M-F-Q"
    assert air.cot_type("helicopter") == "a-u-A-M-H" and air.cot_type("aeroplane") == "a-n-A-C"
    assert air.cot_type("bird") is None and air.cot_type("bird", hostile=True) is None  # shown, never reported
    assert air.cot_type("helicopter", hostile=True) == "a-h-A-M-H" and air.cot_type("aeroplane", True) == "a-h-A-C"
    assert Detection(air.DRONE, "drone", 0.5, (0, 0, 1, 1)).category == "air"


def test_air_cot_types_are_storable_markers():
    from fpv_vision import cot

    eud = {"isrkit-fpv-vision-t"}
    for t in ("a-u-A-M-F-Q", "a-h-A-M-F-Q", "a-u-A-M-H", "a-n-A-C"):
        xml = cot.spot_report_cot(lat=42.0, lon=-93.0, callsign="x", remarks="r", cot_type=t, how="m-g",
                                  sender_uid="isrkit-fpv-vision-t")
        assert store_rejection(xml, eud) is None, t
        assert ET.fromstring(xml.encode()).get("how") == "m-g"


# ------------------------------------------------------------------ tiling + merge
def test_tiles_cover_the_frame_with_overlap():
    t = air.tiles(1920, 1080, grid=2, overlap=0.15)
    assert t[0] == (0, 0, 1920, 1080) and len(t) == 5
    inner = t[1:]
    assert min(r[0] for r in inner) == 0 and max(r[2] for r in inner) == 1920
    assert min(r[1] for r in inner) == 0 and max(r[3] for r in inner) == 1080
    assert inner[0][2] > inner[1][0]  # neighbouring tiles overlap
    assert air.tiles(640, 480, grid=1) == [(0, 0, 640, 480)]


def test_merge_is_class_agnostic_and_drops_tile_fragments():
    rows = np.array([
        [100, 100, 140, 130, 0.60, 4],  # bird, from the full frame
        [101, 99, 141, 131, 0.80, 0],   # same object, "drone" in a tile: higher confidence wins
        [500, 500, 520, 515, 0.40, 0],  # a different small object
        [102, 102, 120, 128, 0.50, 0],  # fragment inside the kept box
    ], float)
    out = air.merge(rows)
    assert len(out) == 2 and out[0][5] == 0 and out[0][4] == pytest.approx(0.8)
    # a tile box touching an inner tile edge is dropped (the full frame sees that object whole)
    cut = air.drop_cut(np.array([[950, 10, 1000, 60, 0.9, 0], [400, 10, 450, 60, 0.9, 0]], float),
                       (0, 0, 1000, 600), 1920, 1080)
    assert len(cut) == 1 and cut[0][0] == 400
    assert air.merge(np.zeros((0, 6))).shape == (0, 6)


def test_air_palette():
    d = lambda tid, cls, label: Detection(cls, label, 0.9, (0, 0, 5, 5), tid)  # noqa: E731
    assert annotate.box_colour(d(1, air.DRONE, "drone")) == annotate.UNKNOWN
    assert annotate.box_colour(d(1, air.DRONE, "drone"), {1}) == annotate.HOSTILE
    assert annotate.box_colour(d(2, air.HELICOPTER, "helicopter")) == annotate.NEUTRAL
    assert annotate.box_colour(d(3, air.AEROPLANE, "aeroplane")) == annotate.NEUTRAL
    assert annotate.box_colour(d(4, air.BIRD, "bird")) == annotate.MUTED
    assert annotate.box_colour(d(5, 0, "person")) == annotate.UNKNOWN


# ------------------------------------------------------------------ toggle + outputs
class FakeAir:
    """A drone (track 1), a bird (track 2) and a helicopter (track 3) in every frame."""

    name = "air"

    def detect(self, frame):
        o = air.TRACK_ID_OFFSET
        return [Detection(air.DRONE, "drone", 0.7, (10, 10, 30, 30), o + 1),
                Detection(air.BIRD, "bird", 0.6, (60, 10, 70, 20), o + 2),
                Detection(air.HELICOPTER, "helicopter", 0.8, (100, 10, 160, 40), o + 3)]

    def reset_tracks(self):
        pass

    def info(self):
        return {"model": "fake", "prompts": air.PROMPTS, "tiles": "2×2 + full frame"}


@pytest.fixture
def air_client(settings):
    settings.sensor_lat, settings.sensor_lon = 42.0, -93.0
    settings.air_fps = 30
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic")).start()
    tak, mesh = CotSender(enabled=False, station="t"), MeshBus(enabled=False)
    cot_sent, mesh_sent = [], []
    tak.send = lambda xml: cot_sent.append(xml) or True
    mesh.send = lambda text: mesh_sent.append(text) or True
    svc = Service(settings, pipeline, SceneDescriber("off", None, GeminiDescriber("", "g")), tak, mesh,
                  detector_factory=FakeDetector, air_factory=FakeAir).start()
    client = TestClient(create_app(svc))
    yield client, svc, cot_sent, mesh_sent
    svc.stop()
    pipeline.stop()


def test_air_toggle_counts_reports_and_hostile(air_client):
    client, svc, cot_sent, mesh_sent = air_client
    assert client.get("/api/status").json()["air"]["on"] is False
    assert client.post("/api/air", json={"on": True}).json() == {"ok": True, "air": True}
    assert wait_for(lambda: len(cot_sent) >= 2 and mesh_sent, timeout=5)
    s = client.get("/api/status").json()
    assert s["air"]["on"] and {"drone", "bird", "helicopter"} <= set(s["counts"])
    assert s["totals"]["drone"] == 1 and s["totals"]["person"] == 2
    types = {ET.fromstring(x.encode()).get("type") for x in cot_sent}
    assert types == {"a-u-A-M-F-Q", "a-u-A-M-H"}  # drone + helicopter; the bird is never reported
    drone = next(ET.fromstring(x.encode()) for x in cot_sent if "a-u-A-M-F-Q" in x)
    assert drone.find("detail/remarks").text.startswith("SIMULATED: Air watch: drone")
    assert drone.get("uid") == "isrkit-fpv-vision-t-air-1"
    assert mesh_sent == [mesh_sent[0]] and mesh_sent[0].startswith("[SIM] Air watch: drone")
    # operator marks the drone hostile: hostile CoT and the track flagged
    cot_sent.clear()
    r = client.post("/api/air/mark", json={"track": 1, "hostile": True}).json()
    assert r["hostile"] and r["cot"]["type"] == "a-h-A-M-F-Q"
    assert wait_for(lambda: any(t["hostile"] for t in client.get("/api/status").json()["air"]["tracks"]))
    # off again: no air boxes
    assert client.post("/api/air", json={"on": False}).json()["air"] is False
    assert wait_for(lambda: "drone" not in client.get("/api/status").json()["counts"])


def test_air_unavailable_reports_why(settings):
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic"))

    def missing():
        raise FileNotFoundError("air watch model not downloaded")

    svc = Service(settings, pipeline, SceneDescriber("off", None, GeminiDescriber("", "g")),
                  CotSender(enabled=False), MeshBus(enabled=False), air_factory=missing)
    client = TestClient(create_app(svc))
    r = client.post("/api/air", json={"on": True})
    assert r.status_code == 409 and "not downloaded" in r.json()["detail"]
    assert svc.air_status()["on"] is False and "not downloaded" in svc.air_status()["error"]


def test_fpv_air_env(monkeypatch):
    from fpv_vision.config import Settings

    monkeypatch.setenv("FPV_AIR", "1")
    monkeypatch.setenv("FPV_AIR_FPS", "6")
    s = Settings.from_env()
    assert s.air is True and s.air_fps == 6.0
    monkeypatch.delenv("FPV_AIR")
    assert Settings.from_env().air is False
