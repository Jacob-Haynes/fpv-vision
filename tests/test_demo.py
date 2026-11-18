"""Step-by-step demo mode (ui-kit DEMO-TRIGGER-CONTRACT): no scenarios, Clear removes our own
spot reports from the map, the heartbeat carries the mode, and a clip starts held on its first frame."""

import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeDetector, wait_for
from fpv_vision import sources
from fpv_vision.gemini import GeminiDescriber
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline
from fpv_vision.service import Service
from fpv_vision.tak_client import TakClient
from fpv_vision.vlm import SceneDescriber
from fpv_vision.web.app import create_app


def _queued(tak):
    out = []
    while not tak._q.empty():
        out.append(ET.fromstring(tak._q.get_nowait()[1]))
    return out


@pytest.fixture
def demo(settings):
    tak = TakClient(enabled=True, tool="fpv-vision", label="FPV", station="test")  # never started: sends stay queued
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic").start()).start()
    svc = Service(settings, pipeline, SceneDescriber("off", None, GeminiDescriber("", "g")), tak,
                  MeshBus(enabled=False)).start()
    yield TestClient(create_app(svc)), svc, tak
    svc.stop()
    pipeline.stop()


def test_scenarios_heartbeat_and_status_carry_the_mode(demo, settings):
    c, svc, _ = demo
    assert c.get("/api/demo/scenarios").json() == {"mode": "continuous", "scenarios": [], "active": []}
    settings.demo_mode = "on_demand"
    assert c.get("/api/demo/scenarios").json()["mode"] == "on_demand"
    hb = svc.heartbeat()
    assert hb["demo_mode"] == "on_demand" and hb["demo_active"] == [] and isinstance(hb["detail"], str)
    assert c.get("/api/status").json()["demo"]["mode"] == "on_demand"
    assert c.post("/api/demo/trigger", json={"id": "inbound"}).status_code == 404


def test_clear_removes_our_spot_reports_from_the_map(demo):
    c, svc, tak = demo
    assert wait_for(lambda: svc.pipeline.latest_frames()[1] is not None)  # an annotated frame to report
    r = c.post("/api/spot-report", json={"position": "42.0, -93.0", "activity": "2 pax", "send_mesh": False}).json()
    assert r["ok"] and r["cot"]["queued"]
    marker = _queued(tak)[0]
    uid = marker.get("uid")
    assert marker.get("type") == "a-u-G"

    t0 = time.time()
    assert c.post("/api/demo/clear").json() == {"ok": True, "cleared": 1}  # well inside the 5 s per-uid limit
    stale, delete, sa = _queued(tak)
    assert stale.get("uid") == uid and stale.get("stale") == stale.get("time")  # stale = now
    assert delete.get("type") == "t-x-d-d" and delete.find("detail/__forcedelete") is not None
    link = delete.find("detail/link")
    assert link.get("uid") == uid and link.get("type") == "a-u-G"
    assert float(delete.find("point").get("lat")) == 42.0  # the marker's point, never 0,0
    assert sa.get("uid") == tak.uid  # registration SA again after the delete
    assert time.time() - t0 < 5
    assert c.post("/api/demo/clear").json()["cleared"] == 0  # already gone
    assert c.get("/api/status").json()["reports"]  # this page's report history is kept


def test_demo_mode_env(monkeypatch):
    from fpv_vision.config import Settings

    assert Settings.from_env().demo_mode == "continuous"
    monkeypatch.setenv("DEMO_MODE", "on_demand")
    assert Settings.from_env().demo_mode == "on_demand"
    monkeypatch.setenv("DEMO_MODE", "rubbish")
    assert Settings.from_env().demo_mode == "continuous"


def test_a_clip_held_on_its_first_frame_waits_for_start(tmp_path):
    clip = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(90):
        w.write(np.full((120, 160, 3), i * 2, np.uint8))
    w.release()
    src = sources.VideoSource(f"replay:{clip}")
    src.hold_first_frame()
    src.start()
    try:
        assert wait_for(lambda: src.latest()[1] is not None and src.replay["paused"])
        first = src.latest()[0]
        time.sleep(0.4)
        assert src.latest()[0] == first == 1  # the first frame only, then nothing until Start
        src.pause(False)  # what Start (POST /api/run) does for a clip
        assert wait_for(lambda: src.latest()[0] > first + 3)
    finally:
        src.stop()
