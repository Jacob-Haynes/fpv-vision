"""Search: tiling + merge with ground classes, the 2-hit gate and the toggle (no model weights needed)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeDetector, wait_for
from fpv_vision import search, sources, tiling
from fpv_vision.cot import CotSender
from fpv_vision.detector import Detection
from fpv_vision.gemini import GeminiDescriber
from fpv_vision.mesh import MeshBus
from fpv_vision.pipeline import Pipeline
from fpv_vision.service import Service
from fpv_vision.vlm import SceneDescriber
from fpv_vision.web.app import create_app

PERSON, CAR, TRUCK, MOTORCYCLE = 0, 2, 7, 3


# ------------------------------------------------------------------ merge with ground classes
def test_merge_groups_vehicles_but_keeps_people_apart():
    rows = np.array([
        [100, 100, 200, 160, 0.70, CAR],    # a van called car on the full frame
        [101, 99, 201, 161, 0.55, TRUCK],   # ... and truck in a tile: same group, one box
        [150, 120, 165, 158, 0.60, PERSON],  # a person standing against it: kept
        [400, 300, 410, 330, 0.30, PERSON],  # a distant figure
        [401, 301, 411, 331, 0.45, PERSON],  # the same figure from an overlapping tile
        [148, 118, 170, 160, 0.40, MOTORCYCLE],  # a motorcycle under the person: another group, kept
    ], float)
    out = tiling.merge(rows, groups=search.GROUPS)
    by_cls = sorted((int(r[5]), round(float(r[4]), 2)) for r in out)
    assert by_cls == [(PERSON, 0.45), (PERSON, 0.6), (CAR, 0.7), (MOTORCYCLE, 0.4)]
    # with no groups (Air watch) the person inside the van box would have gone
    assert len(tiling.merge(rows)) < len(out)


def test_tile_edge_fragments_are_dropped_for_ground_boxes():
    w, h = 3840, 2160
    rects = tiling.tiles(w, h, grid=3)
    assert len(rects) == 10 and rects[0] == (0, 0, w, h)
    inner = rects[5]  # the centre tile: every edge is an inner edge
    x0, y0, x1, y1 = inner
    rows = np.array([[x0 + 1, y0 + 50, x0 + 40, y0 + 120, 0.9, PERSON],  # cut by the left edge
                     [x0 + 200, y0 + 200, x0 + 230, y0 + 290, 0.9, PERSON]], float)  # whole
    kept = tiling.drop_cut(rows, inner, w, h)
    assert len(kept) == 1 and kept[0][0] == x0 + 200


def test_grid_choice():
    assert search.grid_for(2160, 3840) == 3 and search.grid_for(1440, 2560) == 3
    assert search.grid_for(1080, 1920) == 2 and search.grid_for(720, 1280) == 2
    assert search.grid_for(2160, 3840, "2") == 2 and search.grid_for(1080, 1920, 1) == 1


# ------------------------------------------------------------------ predict_tiles with a fake model
class _A:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a.copy()


class _Boxes:
    def __init__(self, rows):
        rows = np.asarray(rows, float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = _A(rows[:, :4]), _A(rows[:, 4]), _A(rows[:, 5])

    def __len__(self):
        return len(self.cls.a)


class _Res:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class FakeYolo:
    """Finds a 'person' at a fixed frame position, but only in crops where it is big enough
    (i.e. the tiles, not the full frame), like the real model with a distant figure."""

    def __init__(self, at=(1000, 600, 1012, 630), present=lambda n: True, conf=0.4):
        self.at, self.present, self.conf, self.calls, self.crop_sizes = at, present, conf, 0, []

    def predict(self, crops, **kw):
        self.calls += 1
        h, w = crops[0].shape[:2]  # the first crop is the full frame
        rects = tiling.tiles(w, h, self._grid)
        out = []
        for i, (c, (x0, y0, x1, y1)) in enumerate(zip(crops, rects)):
            self.crop_sizes.append(c.shape[:2])
            ax0, ay0, ax1, ay1 = self.at
            inside = ax0 >= x0 + 5 and ay0 >= y0 + 5 and ax1 <= x1 - 5 and ay1 <= y1 - 5
            hit = i > 0 and inside and self.present(self.calls)  # too small to see on the full frame
            out.append(_Res([[ax0 - x0, ay0 - y0, ax1 - x0, ay1 - y0, self.conf, PERSON]] if hit else []))
        return out


def _search(fake, hits=2, grid=2):
    fake._grid = grid
    return search.SearchDetector("yolo26s.pt", model=fake, grid=grid, hits=hits, conf=0.25, classes=[0, 2, 7])


def test_predict_tiles_maps_back_and_merges_overlaps():
    fake = FakeYolo(at=(950, 520, 970, 560))  # in the overlap of all four 2×2 tiles of 1920×1080
    fake._grid = 2
    frame = np.zeros((1080, 1920, 3), np.uint8)
    rows = tiling.predict_tiles(fake, frame, 2, imgsz=640, conf=0.25, device="cpu", groups=search.GROUPS)
    assert len(rows) == 1  # four tile hits, one box
    assert tuple(int(v) for v in rows[0][:4]) == (950, 520, 970, 560) and rows[0][5] == PERSON
    assert fake.crop_sizes[0] == (1080, 1920) and len(fake.crop_sizes) == 5


# ------------------------------------------------------------------ 2-hit gate
def test_hit_gate_needs_consecutive_passes():
    g = tiling.HitGate(2)
    assert g.update([1, 2]) == set()
    assert g.update([1]) == {1}          # 1 seen twice in a row; 2 missed a pass
    assert g.update([1, 2]) == {1}       # 2 starts again from one
    assert g.update([2]) == {2}
    assert tiling.HitGate(1).update([5]) == {5}


def test_search_shows_a_figure_only_after_two_passes_and_never_a_one_off():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    det = _search(FakeYolo(present=lambda n: True))
    first, second, third = det.detect(frame), det.detect(frame), det.detect(frame)
    assert first == []
    assert len(second) == 1 and second[0].label == "person" and second[0].track_id > search.TRACK_ID_OFFSET
    assert len(third) == 1 and third[0].track_id == second[0].track_id
    # a one-pass flash (a false positive) is never shown
    flash = _search(FakeYolo(present=lambda n: n == 2))
    assert [len(flash.detect(frame)) for _ in range(4)] == [0, 0, 0, 0]
    assert det.info()["tiles"] == "2×2 + full frame" and det.info()["hits"] == 2


# ------------------------------------------------------------------ toggle + status
class FakeSearch:
    name, backend, accelerator = "search", "fake", "cpu"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(PERSON, "person", 0.31, (5, 5, 9, 15), search.TRACK_ID_OFFSET + 1)]

    def reset_tracks(self):
        pass

    def info(self):
        return {"name": "search", "tiles": "2×2 + full frame", "hits": 2}


@pytest.fixture
def client(settings):
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic")).start()
    fake = FakeSearch()
    svc = Service(settings, pipeline, SceneDescriber("off", None, GeminiDescriber("", "g")),
                  CotSender(enabled=False), MeshBus(enabled=False), search_factory=lambda: fake).start()
    yield TestClient(create_app(svc)), svc, fake
    svc.stop()
    pipeline.stop()


def test_search_toggle_replaces_the_normal_pass(client):
    c, svc, fake = client
    assert c.get("/api/status").json()["search"]["on"] is False
    assert c.post("/api/search", json={"on": True}).json() == {"ok": True, "search": True}
    assert wait_for(lambda: c.get("/api/status").json()["counts"] == {"person": 1})
    s = c.get("/api/status").json()
    assert s["search"]["on"] and s["search"]["model"]["tiles"] == "2×2 + full frame" and s["search"]["hits"] == 2
    assert fake.calls > 0
    assert c.post("/api/search", json={"on": False}).json()["search"] is False
    assert wait_for(lambda: c.get("/api/status").json()["counts"].get("person") == 2)  # normal detector again


def test_search_and_air_watch_together_cap_the_air_rate(client, settings):
    c, svc, _ = client
    settings.air_fps, settings.search_air_fps = 30, 2

    class Air:
        name = "air"

        def detect(self, frame):
            return []

        def reset_tracks(self):
            pass

        def info(self):
            return {}

    svc.pipeline.air = Air()
    assert c.post("/api/air", json={"on": True}).json()["air"] is True
    assert c.post("/api/search", json={"on": True}).json()["search"] is True
    assert wait_for(lambda: c.get("/api/status").json()["air"]["fps"] > 0, timeout=6)
    assert wait_for(lambda: 0 < c.get("/api/status").json()["air"]["fps"] <= 2.6, timeout=6)
    assert c.get("/api/status").json()["counts"] == {"person": 1}


def test_search_unavailable_reports_why(settings):
    pipeline = Pipeline(settings, FakeDetector(), sources.VideoSource("synthetic"))
    svc = Service(settings, pipeline, SceneDescriber("off", None, GeminiDescriber("", "g")),
                  CotSender(enabled=False), MeshBus(enabled=False))  # FakeDetector has no weights
    r = TestClient(create_app(svc)).post("/api/search", json={"on": True})
    assert r.status_code == 409 and "no detector model" in r.json()["detail"]


def test_fpv_search_env(monkeypatch):
    from fpv_vision.config import Settings

    monkeypatch.setenv("FPV_SEARCH", "1")
    monkeypatch.setenv("FPV_SEARCH_CONF", "0.3")
    monkeypatch.setenv("FPV_SEARCH_GRID", "3")
    s = Settings.from_env()
    assert s.search is True and s.search_conf == 0.3 and s.search_grid == "3" and s.search_hits == 2
    monkeypatch.delenv("FPV_SEARCH")
    assert Settings.from_env().search is False
