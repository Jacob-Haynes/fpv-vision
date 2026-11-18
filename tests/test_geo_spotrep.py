import pytest

from fpv_vision import geo, spotrep
from fpv_vision.detector import Detection
from fpv_vision.mesh import MeshBus


def test_mgrs_known_vector():
    # reference value from the widely used `mgrs` package README
    assert geo.to_mgrs(42.0, -93.0) == "15TWG0000049776"
    lat, lon = geo.from_mgrs("15TWG0000049776")
    assert lat == pytest.approx(42.0, abs=1e-4) and lon == pytest.approx(-93.0, abs=1e-4)


@pytest.mark.parametrize("lat,lon", [(42.0, -93.0), (-33.86, 151.21), (0.1, 0.1), (-0.1, -60.0),
                                     (60.0, 5.0), (78.0, 15.0), (-79.0, 170.0), (64.1, -21.9)])
def test_mgrs_round_trip(lat, lon):
    g = geo.to_mgrs(lat, lon)
    back = geo.from_mgrs(g)
    # 1 m MGRS truncation -> compare as ground distance, not degrees
    import math
    dn = (back[0] - lat) * 111_320
    de = (back[1] - lon) * 111_320 * math.cos(math.radians(lat))
    assert math.hypot(dn, de) < 2.0


def test_parse_position_formats():
    assert geo.parse_position("42.0, -93.0") == (42.0, -93.0)
    assert geo.parse_position("42.0 -93.0") == (42.0, -93.0)
    lat, lon = geo.parse_position("15T WG 00000 49776")
    assert lat == pytest.approx(42.0, abs=1e-4) and lon == pytest.approx(-93.0, abs=1e-4)
    # 6-figure grid -> centre of the 100 m square
    lat6, lon6 = geo.parse_position("15TWG000497")
    assert lat6 == pytest.approx(42.0, abs=2e-3) and lon6 == pytest.approx(-93.0, abs=2e-3)
    for bad in ("", "hello", "95, 10", "30UWB123"):
        with pytest.raises(ValueError):
            geo.parse_position(bad)


def test_dtg_format():
    # 2026-11-16 12:30:00Z
    assert spotrep.dtg(1794832200) == "161230ZNOV26"


def test_spot_report_text_limits():
    dets = [Detection(0, "person", 0.9, (0, 0, 10, 20), 1), Detection(2, "car", 0.9, (0, 0, 40, 20), 2)]
    rep = spotrep.build(dets=dets, position="42.0,-93.0", activity="x" * 500, summary_text="s",
                        callsign="FPV-VISION")
    assert rep["size"] == "1 pax, 1 veh"
    assert rep["equipment"] == "1x car"
    assert rep["grid"].startswith("15T WG")
    assert len(rep["mesh_text"]) < 200
    assert rep["mesh_text"].isascii()
    assert "operator-reported location" in rep["remarks"]


def test_mesh_payload_contract():
    p = MeshBus.payload("hello " * 60, source="fpv-vision")
    assert set(p) == {"text", "to", "channel", "source"}
    assert p["to"] is None and p["channel"] == 0 and p["source"] == "fpv-vision"
    assert len(p["text"]) < 200
