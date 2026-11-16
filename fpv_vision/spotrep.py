"""SALUTE spot reports from a snapshot: CoT marker for ATAK + short mesh text.

Location is ALWAYS operator-entered (or the configured default). We do not estimate the
ground position of detections from video: that needs drone GPS/altitude telemetry plus
camera pose (gimbal angles, lens intrinsics) and a terrain model.
"""

from __future__ import annotations

import time

from fpv_vision import geo
from fpv_vision.detector import Detection

MAX_MESH = 199


def dtg(ts: float | None = None) -> str:
    """NATO date-time group, e.g. 161230ZNOV26."""
    t = time.gmtime(ts or time.time())
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    return time.strftime("%d%H%MZ", t) + months[t.tm_mon - 1] + time.strftime("%y", t)


def _short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: max(0, n - 3)].rstrip() + "..."


PROVENANCE_COT = {"simulated": "SIMULATED: ", "recorded": "RECORDED: ", "live": ""}
PROVENANCE_MESH = {"simulated": "[SIM] ", "recorded": "[REC] ", "live": ""}


def build(*, dets: list[Detection], position: str, activity: str, summary_text: str, callsign: str,
          unit: str = "unk", ts: float | None = None, provenance: str = "live") -> dict:
    """Contract C4.3: CoT remarks start 'SIMULATED: '/'RECORDED: ', mesh '[SIM] '/'[REC] '; live has none."""
    from fpv_vision.summary import equipment_text, size_text

    lat, lon = geo.parse_position(position)
    if lat == 0 and lon == 0:
        raise geo.PositionError("0,0 is not a real position (set SENSOR_LAT/SENSOR_LON or enter a grid)")
    ts = ts or time.time()
    try:
        grid = geo.format_mgrs(geo.to_mgrs(lat, lon))
    except geo.PositionError:
        grid = f"{lat:.5f},{lon:.5f}"
    size = size_text(dets)
    equip = equipment_text(dets)
    act = activity.strip() or summary_text or "observed"
    when = dtg(ts)

    cot_prefix = PROVENANCE_COT.get(provenance, "")
    mesh_prefix = PROVENANCE_MESH.get(provenance, "")
    remarks = (
        f"{cot_prefix}SALUTE (operator-reported location, not sensor-derived)\n"
        f"S: {size}\nA: {act}\nL: {grid} ({lat:.5f}, {lon:.5f})\nU: {unit}\nT: {when}\nE: {equip}\n"
        f"Summary: {summary_text or '-'}\nSource: {callsign} video"
    )

    head = f"{mesh_prefix}SPOTREP {callsign} S:{size} L:{grid} T:{when} U:{unit} E:{equip} A:"
    room = MAX_MESH - len(head)
    mesh_text = head + _short(act, max(room, 10))
    if len(mesh_text) > MAX_MESH:
        mesh_text = _short(mesh_text, MAX_MESH)

    return {
        "lat": lat, "lon": lon, "grid": grid, "dtg": when, "size": size, "activity": act,
        "unit": unit, "equipment": equip, "remarks": remarks, "mesh_text": mesh_text,
        "callsign": f"SPOT {when[:6]}",
        "provenance": provenance,
    }
