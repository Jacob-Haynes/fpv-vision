"""Stable detections: fewer false vehicles and no flicker on the ground picture.

From a drone, YOLO calls roofs, walls and whole buildings "truck" and fences "train", usually
for a frame or two at a time and at middling confidence, and real objects drop out for a frame
now and then. ``TrackCounter`` (``fpv_vision.pipeline``) uses this config to:

* drop vehicle boxes that cannot be a vehicle from the air: more than ``vehicle_max_area`` of
  the frame, or longer than ``vehicle_max_aspect`` times their width (fences, kerbs, walls);
* drop a truck, bus or train box more than ``heavy_max_car_ratio`` times the size of the cars
  seen lately (the median of the last ``CAR_REF_SAMPLES`` car boxes): that is a roof or a whole
  building, not a lorry. On the user's clips buildings called truck were 4-700x a car (most over
  10x), real vans 0.4-4x, and only drone close-ups more;
* confirm a track only after ``confirm_vehicle`` consecutive detections (``confirm_other`` for
  people and boats), and only when its voted mean confidence reaches the class floor
  (``class_conf``: higher for truck, bus and train than for car or person; Search, which runs
  at a lower confidence over tiles, has its own, higher ``search_class_conf``);
* draw and count confirmed tracks only;
* keep (coast) a confirmed track's last box, faded, for ``coast_s`` after it was last seen;
* hand a held track over to a new tracker id that appears on top of it (``handover_iou``), so
  a blur or a fast pan doesn't turn one car into three, or blink while the new id confirms;
* give each track one label, changed only when another class clearly wins the vote
  (``label_switch`` times the current one's confidence total).

Every value has an environment variable (see ``Settings.from_env``); ``FPV_STABLE=0`` turns it
all off and shows the tracker's raw output, as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fpv_vision.detector import Detection, category

DEFAULT_CLASS_CONF = {"truck": 0.4, "bus": 0.4, "train": 0.4}  # above the detector's 0.30; see the README
DEFAULT_SEARCH_CLASS_CONF = {"truck": 0.55, "bus": 0.55, "train": 0.55}  # Search runs at 0.25
HEAVY = {"truck", "bus", "train"}
CAR_REF_SAMPLES, CAR_REF_MIN = 300, 20  # car boxes kept for the size reference; fewer = no check


def parse_class_conf(raw: str, default: dict[str, float] | None = None) -> dict[str, float]:
    """'truck=0.55,bus=0.6' -> {'truck': 0.55, 'bus': 0.6}. Bad entries are skipped."""
    out = dict(DEFAULT_CLASS_CONF if default is None else default)
    for part in (raw or "").split(","):
        name, _, val = part.partition("=")
        try:
            out[name.strip().lower()] = float(val)
        except ValueError:
            continue
    return out


@dataclass
class StableConfig:
    class_conf: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_CONF))
    search_class_conf: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SEARCH_CLASS_CONF))
    vehicle_max_area: float = 0.25  # fraction of the frame
    vehicle_max_aspect: float = 4.0  # long side / short side
    confirm_vehicle: int = 3  # consecutive detections before a vehicle is shown or counted
    confirm_other: int = 3  # people, boats
    coast_s: float = 0.5
    label_switch: float = 1.5
    handover_iou: float = 0.3
    heavy_max_car_ratio: float = 10.0  # 0 = no check

    @classmethod
    def from_settings(cls, s) -> "StableConfig | None":
        if not getattr(s, "stable", True):
            return None
        return cls(class_conf=dict(s.class_conf), search_class_conf=dict(s.search_class_conf),
                   vehicle_max_area=s.vehicle_max_area,
                   vehicle_max_aspect=s.vehicle_max_aspect, confirm_vehicle=max(1, s.confirm_hits),
                   confirm_other=max(1, s.confirm_hits_other), coast_s=max(0.0, s.coast_s),
                   label_switch=max(1.0, s.label_switch), handover_iou=s.handover_iou,
                   heavy_max_car_ratio=max(0.0, s.heavy_max_car_ratio))

    def floor(self, label: str, search: bool = False) -> float:
        return (self.search_class_conf if search else self.class_conf).get(label, 0.0)

    def needed(self, cls_id: int) -> int:
        return self.confirm_vehicle if category(cls_id) == "vehicle" else self.confirm_other

    def plausible(self, d: Detection, frame_shape: tuple[int, int] | None, car_area: float = 0.0) -> bool:
        """False for a vehicle box no vehicle could fill: most of the frame, a long thin strip, or
        (truck, bus, train) many times the size of the cars around it. ``car_area``: the median
        recent car box area, 0 when there isn't one yet."""
        if category(d.cls_id) != "vehicle":
            return True
        x1, y1, x2, y2 = d.xyxy
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if max(w, h) / min(w, h) > self.vehicle_max_aspect:
            return False
        if car_area and self.heavy_max_car_ratio and d.label in HEAVY and w * h > self.heavy_max_car_ratio * car_area:
            return False
        if frame_shape and frame_shape[0] and frame_shape[1]:
            return w * h <= self.vehicle_max_area * frame_shape[0] * frame_shape[1]
        return True


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
