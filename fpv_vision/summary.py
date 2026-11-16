"""Offline scene summary built only from detector output (no cloud, no ML beyond YOLO).

Produces short ISR-style text like "2 people near a car, left of frame; 1 truck centre."
It is deliberately conservative: it reports what the detector boxed and where in the
*image*, never a ground position.
"""

from __future__ import annotations

from collections import Counter

from fpv_vision.detector import Detection

PLURALS = {"person": "people", "bus": "buses"}


def plural(label: str, n: int) -> str:
    if n == 1:
        return f"1 {label}"
    return f"{n} {PLURALS.get(label, label + 's')}"


def _region(d: Detection, w: int) -> str:
    cx = (d.xyxy[0] + d.xyxy[2]) / 2
    if cx < w / 3:
        return "left"
    if cx > 2 * w / 3:
        return "right"
    return "centre"


def _near(a: Detection, b: Detection) -> bool:
    ax = (a.xyxy[0] + a.xyxy[2]) / 2
    ay = (a.xyxy[1] + a.xyxy[3]) / 2
    bx = (b.xyxy[0] + b.xyxy[2]) / 2
    by = (b.xyxy[1] + b.xyxy[3]) / 2
    scale = max(b.xyxy[2] - b.xyxy[0], b.xyxy[3] - b.xyxy[1], a.xyxy[3] - a.xyxy[1])
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < 1.5 * scale


def count_by_label(dets: list[Detection]) -> dict[str, int]:
    return dict(Counter(d.label for d in dets))


def size_text(dets: list[Detection]) -> str:
    """SALUTE 'Size' field: e.g. '3 pax, 2 veh'."""
    pax = sum(1 for d in dets if d.category == "person")
    veh = sum(1 for d in dets if d.category == "vehicle")
    other = len(dets) - pax - veh
    parts = []
    if pax:
        parts.append(f"{pax} pax")
    if veh:
        parts.append(f"{veh} veh")
    if other:
        parts.append(f"{other} other")
    return ", ".join(parts) or "nil seen"


def equipment_text(dets: list[Detection]) -> str:
    c = Counter(d.label for d in dets if d.category != "person")
    return ", ".join(f"{n}x {k}" for k, n in c.most_common()) or "none seen"


def describe(dets: list[Detection], frame_width: int) -> str:
    if not dets:
        return "No people or vehicles detected in view."
    people = [d for d in dets if d.category == "person"]
    vehicles = [d for d in dets if d.category == "vehicle"]
    others = [d for d in dets if d.category == "other"]
    phrases: list[str] = []

    if people and vehicles:
        near = [p for p in people if any(_near(p, v) for v in vehicles)]
        if near:
            v0 = min(vehicles, key=lambda v: min(abs(v.xyxy[0] - p.xyxy[0]) for p in near))
            regions = Counter(_region(p, frame_width) for p in near)
            where = regions.most_common(1)[0][0]
            phrases.append(f"{plural('person', len(near))} near a {v0.label}, {where} of frame")
            people = [p for p in people if p not in near]
            vehicles = [v for v in vehicles if v is not v0]

    for group in (people, vehicles, others):
        by_label: dict[str, list[Detection]] = {}
        for d in group:
            by_label.setdefault(d.label, []).append(d)
        for label, items in by_label.items():
            regions = Counter(_region(d, frame_width) for d in items)
            where = " and ".join(r for r, _ in regions.most_common(2))
            phrases.append(f"{plural(label, len(items))} {where}")

    text = "; ".join(phrases)
    return text[0].upper() + text[1:] + "."
