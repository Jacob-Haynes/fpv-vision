"""Stable detections (fpv_vision.stable + TrackCounter): implausible vehicle boxes are dropped,
tracks are shown and counted only once confirmed, a missed track coasts on its last box, faded,
a re-identified object keeps its track, and a track keeps one label."""

import numpy as np
import pytest

from conftest import wait_for
from fpv_vision import annotate, sources, summary
from fpv_vision.detector import Detection, Detector
from fpv_vision.pipeline import Pipeline, TrackCounter
from fpv_vision.stable import StableConfig, iou, parse_class_conf

SHAPE = (1080, 1920)
CAR, TRUCK, BUS, PERSON = 2, 7, 5, 0


def det(cls_id, label, conf, xyxy, tid):
    return Detection(cls_id, label, conf, tuple(xyxy), tid)


def car(tid, x=100, conf=0.5):
    return det(CAR, "car", conf, (x, 500, x + 60, 530), tid)


def counter(**kw):
    return TrackCounter(stable=StableConfig(**kw))


def ids(out, coast=None):
    return sorted(d.track_id for d in out if coast is None or d.coast == coast)


# ------------------------------------------------------------------ geometry
def test_vehicle_boxes_no_vehicle_could_fill_are_dropped():
    cfg = StableConfig()
    assert cfg.plausible(car(1), SHAPE)
    building = det(TRUCK, "truck", 0.8, (0, 0, 1200, 600), 2)  # 35% of the frame
    fence = det(6, "train", 0.8, (100, 500, 900, 560), 3)  # 800 x 60: a strip
    assert not cfg.plausible(building, SHAPE) and not cfg.plausible(fence, SHAPE)
    tall_person = det(PERSON, "person", 0.8, (100, 100, 120, 400), 4)  # people are never size-checked
    assert cfg.plausible(tall_person, SHAPE)


def test_a_truck_many_times_the_size_of_the_cars_around_it_is_a_building():
    cfg = StableConfig()
    roof = det(TRUCK, "truck", 0.7, (0, 0, 400, 150), 9)  # 60000 px; the cars are 1800
    van = det(TRUCK, "truck", 0.7, (0, 0, 90, 45), 10)  # 4050 px: 2.25 cars
    assert cfg.plausible(roof, SHAPE, car_area=0)  # no cars seen yet: no reference, no check
    assert not cfg.plausible(roof, SHAPE, car_area=1800) and cfg.plausible(van, SHAPE, car_area=1800)
    assert StableConfig(heavy_max_car_ratio=0).plausible(roof, SHAPE, car_area=1800)

    c = counter(confirm_vehicle=1)
    for k in range(25):  # cars build the reference
        c.update([car(100 + k, x=20 * k)], SHAPE, k * 0.04)
    out = c.update([roof, van], SHAPE, 1.0)
    assert ids(out, coast=False) == [10]  # (the cars coast: one track, handed over along the row)


# ------------------------------------------------------------------ confirmation and floors
def test_vehicles_need_three_consecutive_detections_and_a_miss_starts_again():
    c = counter()
    assert c.update([car(1)], SHAPE, 0.0) == []
    assert c.update([car(1)], SHAPE, 0.1) == []
    assert c.update([], SHAPE, 0.2) == []  # missed before confirming: back to zero
    assert c.update([car(1)], SHAPE, 0.3) == [] and c.update([car(1)], SHAPE, 0.4) == []
    out = c.update([car(1)], SHAPE, 0.5)
    assert ids(out) == [1] and c.just_counted == [1] and c.totals == {"car": 1}


def test_class_floors_hold_back_weak_trucks_but_not_cars():
    c = counter(class_conf=parse_class_conf("truck=0.55"))
    weak_truck = det(TRUCK, "truck", 0.45, (500, 500, 590, 545), 2)
    for k in range(6):
        out = c.update([car(1, conf=0.35), weak_truck], SHAPE, k * 0.1)
    assert ids(out) == [1] and c.totals == {"car": 1}


def test_search_has_its_own_higher_floor():
    truck = det(TRUCK, "truck", 0.5, (500, 500, 590, 545), 2)
    normal, search = counter(), counter()
    for k in range(3):
        a = normal.update([truck], SHAPE, k * 0.1)
        b = search.update([truck], SHAPE, k * 0.1, search=True)
    assert ids(a) == [2] and b == []  # 0.5 clears the normal 0.40, not Search's 0.55


def test_a_low_confidence_bus_is_shown_as_a_truck():
    c = counter()
    bus = det(BUS, "bus", 0.5, (500, 500, 590, 545), 3)  # under RARE_MIN_CONF
    for k in range(3):
        out = c.update([bus], SHAPE, k * 0.1)
    assert [d.label for d in out] == ["truck"]


def test_search_gate_counts_towards_confirmation():
    c = counter()
    assert c.update([car(1)], SHAPE, 0.0, prior_hits=1) == []
    assert ids(c.update([car(1)], SHAPE, 0.1, prior_hits=1)) == [1]  # Search already saw it once


def test_untracked_boxes_are_never_shown():
    c = counter(confirm_vehicle=1)
    assert c.update([det(CAR, "car", 0.9, (1, 1, 50, 30), None)], SHAPE, 0.0) == []


# ------------------------------------------------------------------ flicker
def test_a_missed_track_coasts_faded_then_goes():
    c = counter(coast_s=0.5)
    for k in range(3):
        c.update([car(1)], SHAPE, k * 0.1)
    out = c.update([], SHAPE, 0.4)
    assert ids(out, coast=True) == [1] and out[0].xyxy == car(1).xyxy
    assert summary.count_by_label(out) == {"car": 1}  # In view doesn't blink either
    assert ids(c.update([car(1)], SHAPE, 0.6), coast=False) == [1]  # back: drawn normally again
    assert ids(c.update([], SHAPE, 1.0)) == [1]
    assert c.update([], SHAPE, 1.2) == []  # 0.6 s since last seen: gone
    assert c.totals == {"car": 1}


def test_a_new_tracker_id_on_a_held_box_takes_the_track_over():
    c = counter()
    for k in range(3):
        c.update([car(1)], SHAPE, k * 0.1)
    c.update([], SHAPE, 0.3)  # a blur: the tracker loses it ...
    out = c.update([car(7, x=105)], SHAPE, 0.4)  # ... and finds it again under a new id
    assert ids(out) == [1] and not out[0].coast  # shown at once, as track 1, not as a new car
    assert ids(c.update([car(7, x=110)], SHAPE, 0.5)) == [1]
    assert c.totals == {"car": 1}
    far = c.update([car(1, x=110), car(8, x=900)], SHAPE, 0.6)  # a different car elsewhere confirms on its own
    assert ids(far) == [1]


def test_a_takeover_that_turns_out_to_be_a_neighbour_keeps_both_boxes():
    c = counter()
    for k in range(3):
        c.update([car(1)], SHAPE, k * 0.1)
    c.update([], SHAPE, 0.3)
    assert ids(c.update([car(7, x=130)], SHAPE, 0.4)) == [1]  # took over track 1 ...
    out = c.update([car(1), car(7, x=130)], SHAPE, 0.5)  # ... but car 1 is back: 7 is another car
    assert ids(out) == [1, 7] and not any(d.coast for d in out)
    assert c.totals == {"car": 2}


def test_a_confirmed_track_keeps_its_label():
    c = counter()
    for k in range(3):
        c.update([car(1, conf=0.6)], SHAPE, k * 0.1)
    van = lambda conf: det(TRUCK, "truck", conf, car(1).xyxy, 1)
    labels = [d.label for k in range(3) for d in c.update([van(0.65)], SHAPE, 0.3 + k * 0.1)]
    assert labels == ["car"] * 3  # truck now leads the vote, but not by 1.5x
    labels = [d.label for k in range(4) for d in c.update([van(0.65)], SHAPE, 0.6 + k * 0.1)]
    assert labels[-1] == "truck" and sum(a != b for a, b in zip(labels, labels[1:])) == 1  # clearly a truck: one switch
    assert c.totals == {"truck": 1}


def test_switching_needs_the_new_class_floor_too():
    c = counter(class_conf={"truck": 0.55})
    for k in range(3):
        c.update([car(1, conf=0.4)], SHAPE, k * 0.1)
    for k in range(10):  # many weak truck votes outnumber the car ones, but stay under the floor
        out = c.update([det(TRUCK, "truck", 0.45, car(1).xyxy, 1)], SHAPE, 0.3 + k * 0.1)
    assert [d.label for d in out] == ["car"]


def test_forget_live_drops_held_boxes():
    c = counter()
    for k in range(3):
        c.update([car(1)], SHAPE, k * 0.1)
    c.forget_live()
    assert c.update([], SHAPE, 0.35) == []


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


# ------------------------------------------------------------------ settings and drawing
def test_env(monkeypatch):
    from fpv_vision.config import Settings

    s = Settings.from_env()
    cfg = StableConfig.from_settings(s)
    assert cfg.floor("truck") == 0.4 and cfg.floor("car") == 0.0 and cfg.confirm_vehicle == 3 and cfg.coast_s == 0.5
    monkeypatch.setenv("FPV_CLASS_CONF", "truck=0.55, train=0.7,junk")
    monkeypatch.setenv("FPV_CONFIRM_HITS", "4")
    monkeypatch.setenv("FPV_COAST_S", "0.8")
    monkeypatch.setenv("FPV_VEHICLE_MAX_AREA", "0.15")
    monkeypatch.setenv("FPV_HEAVY_MAX_CAR_RATIO", "0")
    cfg = StableConfig.from_settings(Settings.from_env())
    assert cfg.floor("truck") == 0.55 and cfg.floor("train") == 0.7 and cfg.floor("bus") == 0.4
    assert cfg.confirm_vehicle == 4 and cfg.coast_s == 0.8 and cfg.vehicle_max_area == 0.15
    assert cfg.heavy_max_car_ratio == 0
    monkeypatch.setenv("FPV_SEARCH_CLASS_CONF", "truck=0.7")
    cfg = StableConfig.from_settings(Settings.from_env())
    assert cfg.floor("truck", search=True) == 0.7 and cfg.floor("bus", search=True) == 0.55
    monkeypatch.setenv("FPV_STABLE", "0")
    assert StableConfig.from_settings(Settings.from_env()) is None


def test_coasting_boxes_are_drawn_faded():
    img = np.zeros((200, 300, 3), np.uint8)
    d = det(CAR, "car", 0.5, (50, 50, 150, 150), 1)
    live = annotate.draw(img.copy(), [d])
    held = annotate.draw(img.copy(), [Detection(**{**d.__dict__, "coast": True})])
    assert tuple(live[150, 100]) == annotate.UNKNOWN  # bottom edge, below the label
    assert tuple(held[150, 100]) == annotate.faded(annotate.UNKNOWN)
    assert sum(annotate.faded(annotate.UNKNOWN)) < sum(annotate.UNKNOWN)


def test_scaling_keeps_every_field():
    d = Detection(100, "drone", 0.8, (10, 10, 20, 20), 5, note="hexacopter", named_by="ai", coast=True)
    (s,) = annotate.scale_detections([d], 2, 2)
    assert s.xyxy == (20, 20, 40, 40) and (s.note, s.named_by, s.coast) == ("hexacopter", "ai", True)


# ------------------------------------------------------------------ in the pipeline
class Blinking(Detector):
    """One car the model loses on every fourth frame, like the real thing on a blurry pan."""

    name = backend = "fake"

    def __init__(self):
        self.n = 0

    def detect(self, frame):
        self.n += 1
        h, w = frame.shape[:2]
        return [] if self.n % 4 == 0 else [det(CAR, "car", 0.6, (w // 4, h // 2, w // 4 + 40, h // 2 + 20), 1)]


def test_the_pipeline_shows_a_blinking_car_steadily(settings):
    p = Pipeline(settings, Blinking(), sources.VideoSource("synthetic")).start()
    try:
        assert wait_for(lambda: p.counter.totals == {"car": 1})
        seen = []
        for _ in range(40):
            seen.append(p.current_counts().get("car", 0))
            wait_for(lambda: False, timeout=0.02)
        assert seen.count(1) == len(seen)  # never blinks off
    finally:
        p.stop()


def test_fpv_stable_off_shows_the_raw_tracks(settings):
    settings.stable = False
    p = Pipeline(settings, Blinking(), sources.VideoSource("synthetic"))
    assert p.counter.stable is None
