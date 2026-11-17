"""The vision model names Air watch tracks: mapping, low confidence, re-ask limit, queue, fallback, CoT."""

import threading
import urllib.error
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeDetector, wait_for
from fpv_vision import air, sources
from fpv_vision.air_namer import AirNamer, crop, parse_answer, to_label
from fpv_vision.cot import CotSender
from fpv_vision.detector import Detection
from fpv_vision.gemini import GeminiDescriber
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline, TrackCounter
from fpv_vision.service import Service
from fpv_vision.vlm import SceneDescriber
from fpv_vision.web.app import create_app

O = air.TRACK_ID_OFFSET
FRAME = np.zeros((720, 1280, 3), np.uint8)


def heli(tid=1, conf=0.85):
    return Detection(air.HELICOPTER, "helicopter", conf, (600, 300, 660, 340), O + tid)


class Fake:
    """Answers from a list (last one repeats); counts calls."""

    def __init__(self, *answers):
        self.answers, self.calls, self.model = list(answers), 0, "fake-vl"

    def __call__(self, jpeg):
        assert jpeg[:2] == b"\xff\xd8"
        self.calls += 1
        a = self.answers[min(self.calls, len(self.answers)) - 1]
        if isinstance(a, Exception):
            raise a
        return a


def test_answer_parsing_and_mapping():
    assert parse_answer('```json\n{"class": "Drone", "confidence": 0.93, "detail": "hexacopter with six arms and props"}\n```') == \
        {"class": "drone", "confidence": 0.93, "detail": "hexacopter with six arms and props"}
    assert parse_answer("{'class': 'airplane', 'confidence': 85, 'detail': 'airliner',}")["class"] == "aeroplane"
    assert parse_answer('{"class": "balloon", "confidence": 0.9, "detail": "x"}')["class"] == "other"
    with pytest.raises(ValueError):
        parse_answer("a drone, I think")
    assert to_label({"class": "drone", "confidence": 0.9, "detail": "quadcopter"}) == (air.DRONE, "drone", "quadcopter")
    assert to_label({"class": "drone", "confidence": 0.4, "detail": "?"})[1] == "air contact"  # low confidence
    assert to_label({"class": "other", "confidence": 0.95, "detail": "kite"})[:2] == (air.AIR_CONTACT, "air contact")
    assert air.cot_type("air contact") == "a-u-A"


def test_crop_padding_and_minimum_size():
    import cv2

    small = cv2.imdecode(np.frombuffer(crop(FRAME, (600, 300, 620, 310)), np.uint8), cv2.IMREAD_COLOR)
    assert min(small.shape[:2]) >= 224
    big = cv2.imdecode(np.frombuffer(crop(FRAME, (600, 300, 620, 310), pad=0.6, min_side=448), np.uint8), 1)
    assert min(big.shape[:2]) >= 448


def _namer(client, **kw):
    c = TrackCounter(min_hits=1)
    c.update([heli()])
    return AirNamer(client, c, **kw).start(), c


def test_confident_name_replaces_the_vote_once():
    fake = Fake({"class": "drone", "confidence": 0.95, "detail": "hexacopter"})
    n, c = _namer(fake)
    try:
        n.observe([heli()], FRAME, now=0.0)
        n.observe([heli()], FRAME, now=0.3)
        assert not n.queue and fake.calls == 0  # not stable for 0.5 s yet
        n.observe([heli()], FRAME, now=0.6)
        assert wait_for(lambda: fake.calls == 1 and n.tracks[O + 1].name)
        d = n.apply([heli()])[0]
        assert (d.label, d.note, d.named_by, d.cls_id) == ("drone", "hexacopter", "ai", air.DRONE)
        assert c.voted(O + 1)[1] == "drone" and c.snapshot() == {"drone": 1}
        for t in (4, 8, 12):
            n.observe([heli()], FRAME, now=t)
        assert fake.calls == 1  # a confident name is never re-asked
        # birds (by YOLOE) and low-confidence boxes are never sent
        n.observe([Detection(air.BIRD, "bird", 0.9, (0, 0, 9, 9), O + 2)], FRAME, now=20)
        n.observe([heli(3, conf=0.5)], FRAME, now=20)
        n.observe([heli(3, conf=0.5)], FRAME, now=21)
        assert fake.calls == 1
    finally:
        n.stop()


def test_low_confidence_is_air_contact_and_reasked_at_most_twice():
    fake = Fake({"class": "drone", "confidence": 0.3, "detail": "blur"})
    n, c = _namer(fake)
    try:
        n.observe([heli()], FRAME, now=0.0)
        n.observe([heli()], FRAME, now=0.6)
        assert wait_for(lambda: fake.calls == 1 and not n.tracks[O + 1].pending)
        assert n.apply([heli()])[0].label == "air contact"
        n.observe([heli()], FRAME, now=2.0)  # under 3 s since the last ask
        assert fake.calls == 1 and not n.queue
        for t, want in ((3.7, 2), (6.8, 3), (10.0, 3), (20.0, 3)):
            n.observe([heli()], FRAME, now=t)
            assert wait_for(lambda: fake.calls == want and not n.tracks[O + 1].pending), t
        assert n.tracks[O + 1].asks == 3
    finally:
        n.stop()


def test_queue_is_capped_newest_first():
    gate = threading.Event()

    class Slow(Fake):
        def __call__(self, jpeg):
            gate.wait(5)
            return super().__call__(jpeg)

    fake = Slow({"class": "drone", "confidence": 0.9, "detail": "x"})
    n, _ = _namer(fake, queue_cap=3)
    try:
        for tid in range(1, 6):
            n.observe([heli(tid)], FRAME, now=0.0)
        for tid in range(1, 6):
            n.observe([heli(tid)], FRAME, now=0.6)
            if tid == 1:
                assert wait_for(lambda: n.in_flight == 1)  # track 1 is in flight
        assert [q[0] - O for q in n.queue] == [5, 4, 3] and n.dropped == 1  # track 2 dropped
        assert not n.tracks[O + 2].pending and n.tracks[O + 2].asks == 0  # it can be asked again later
        gate.set()
        assert wait_for(lambda: fake.calls == 4 and not n.queue)
    finally:
        gate.set()
        n.stop()


def test_unreachable_keeps_the_yoloe_vote():
    fake = Fake(urllib.error.URLError("connection refused"))
    n, c = _namer(fake)
    try:
        n.observe([heli()], FRAME, now=0.0)
        n.observe([heli()], FRAME, now=0.6)
        assert wait_for(lambda: fake.calls == 1 and not n.reachable)
        assert n.apply([heli()])[0].label == "helicopter" and c.voted(O + 1)[1] == "helicopter"
        n.observe([heli()], FRAME, now=4.0)  # in back-off: no new call
        assert fake.calls == 1 and n.status()["reachable"] is False
    finally:
        n.stop()


# ------------------------------------------------------------------ service: CoT retyping
class HeliThatIsADrone:
    def detect(self, frame):
        return [heli()]

    def reset_tracks(self):
        pass

    def info(self):
        return {"model": "fake"}


def test_named_drone_retypes_cot_and_alerts_mesh(settings):
    settings.sensor_lat, settings.sensor_lon = 42.0, -93.0
    settings.air_fps = 30
    settings.air = True  # FPV_AIR=1
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic")).start()
    tak, mesh = CotSender(enabled=False, station="t"), MeshBus(enabled=False)
    cot_sent, mesh_sent = [], []
    tak.send = lambda xml: cot_sent.append(xml) or True
    mesh.send = lambda text: mesh_sent.append(text) or True
    gate = threading.Event()

    class Namer(Fake):
        def __call__(self, jpeg):
            gate.wait(5)  # let the first (YOLOE) CoT go out first
            return super().__call__(jpeg)

    svc = Service(settings, pipeline, SceneDescriber("off", None, GeminiDescriber("", "g")), tak, mesh,
                  air_factory=HeliThatIsADrone,
                  namer_client=Namer({"class": "drone", "confidence": 0.95, "detail": "hexacopter"})).start()
    client = TestClient(create_app(svc))
    try:
        assert client.post("/api/air", json={"on": True}).json()["ok"]
        assert wait_for(lambda: cot_sent, timeout=5)
        assert ET.fromstring(cot_sent[0].encode()).get("type") == "a-u-A-M-H" and not mesh_sent
        gate.set()
        assert wait_for(lambda: len(cot_sent) >= 2 and mesh_sent, timeout=5)
        second = ET.fromstring(cot_sent[1].encode())
        assert second.get("type") == "a-u-A-M-F-Q" and second.get("uid") == ET.fromstring(cot_sent[0].encode()).get("uid")
        assert "hexacopter" in second.find("detail/remarks").text
        assert mesh_sent[0].startswith("[SIM] Air watch: drone (hexacopter)")
        assert wait_for(lambda: client.get("/api/status").json()["air"]["tracks"][0]["label"] == "drone")
        a = client.get("/api/status").json()["air"]
        assert a["tracks"][0]["label"] == "drone" and a["tracks"][0]["named_by"] == "ai"
        assert a["naming"]["named"] == 1 and a["totals"] == {"drone": 1}
    finally:
        gate.set()
        svc.stop()
        pipeline.stop()


def test_timeout_is_slow_not_unreachable():
    from fpv_vision.air_namer import _is_unreachable

    assert _is_unreachable(urllib.error.URLError(ConnectionRefusedError(61, "refused")))
    assert not _is_unreachable(TimeoutError("timed out"))
    assert not _is_unreachable(urllib.error.URLError("timed out"))
    assert not _is_unreachable(ValueError("no JSON in reply"))
