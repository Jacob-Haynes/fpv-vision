"""TakClient against a fake OpenTAKServer that enforces the C5 rules."""

import socket
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import pytest

from fpv_vision import cot
from fpv_vision.tak_client import TakClient, registration_sa

from conftest import wait_for
from fake_ots import FakeOTS

DELAY = 0.3  # client register_delay used in tests (real default 3 s)


@pytest.fixture
def ots():
    srv = FakeOTS(channel_delay=0.25)
    yield srv
    srv.close()


def client(ots, **kw):
    kw.setdefault("register_delay", DELAY)
    kw.setdefault("station", "teststation")
    return TakClient("127.0.0.1", ots.port, **kw).start()


def spot(uid="fpv-vision-spot-1", remarks="RECORDED: SALUTE ...", sender_uid="isrkit-fpv-vision-teststation"):
    return cot.spot_report_cot(lat=42.0, lon=-93.0, callsign="SPOT 161230", remarks=remarks, uid=uid,
                               sender_uid=sender_uid, sender_callsign="FPV teststation")


def _t(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()


# ------------------------------------------------------------------ registration SA


def test_registration_sa_without_position_is_never_drawn():
    xml = registration_sa(uid="isrkit-fpv-vision-isr-droneops", callsign="FPV isr-droneops", lat=None, lon=None,
                          device="fpv-vision", version="0.1.0", now=1794832200)
    ev = ET.fromstring(xml.encode())
    assert ev.get("uid") == "isrkit-fpv-vision-isr-droneops" and ev.get("type") == "a-f-G-E-S"
    for k in ("time", "start", "stale"):
        assert ev.get(k).endswith("Z")
    assert ev.get("stale") == ev.get("start")  # stale = start: never drawn
    pt = ev.find("point")
    assert (pt.get("lat"), pt.get("lon"), pt.get("hae"), pt.get("ce"), pt.get("le")) == \
        ("0", "0", "9999999", "9999999", "9999999")
    c = ev.find("detail/contact")
    assert c.get("callsign") == "FPV isr-droneops" and c.get("endpoint") == "*:-1:stcp"
    g = ev.find("detail/__group")
    assert g.get("name") == "Cyan" and g.get("role") == "Team Member"
    tv = ev.find("detail/takv")
    assert tv.get("platform") == "isr-kit" and tv.get("device") == "fpv-vision" and tv.get("version") == "0.1.0"
    assert tv.get("os")


def test_registration_sa_with_sensor_position():
    xml = registration_sa(uid="u", callsign="FPV x", lat=42.0, lon=-93.0, device="fpv-vision", version="1",
                          now=1794832200)
    ev = ET.fromstring(xml.encode())
    assert float(ev.find("point").get("lat")) == 42.0 and float(ev.find("point").get("lon")) == -93.0
    assert _t(ev.get("stale")) - _t(ev.get("start")) == pytest.approx(120, abs=0.01)
    # 0,0 configured is treated as "no position" (unknown, stale = start)
    ev0 = ET.fromstring(registration_sa(uid="u", callsign="c", lat=0.0, lon=0.0, device="d", version="1").encode())
    assert ev0.get("stale") == ev0.get("start") and ev0.find("point").get("ce") == "9999999"


def test_identity_is_stable_per_station():
    a = TakClient(station="ISR-DroneOps", enabled=False)
    b = TakClient(station="isr-droneops", enabled=False)
    assert a.uid == b.uid == "isrkit-fpv-vision-isr-droneops"
    assert a.callsign == b.callsign == "FPV isr-droneops"


# ------------------------------------------------------------------ the fake enforces the rules


def test_fake_ots_drops_naive_senders(ots):
    """Sanity check of the fake: a one-shot sender and an early registration both lose."""
    s = socket.create_connection(("127.0.0.1", ots.port))
    s.sendall(spot().encode())  # rule 1: no SA
    s.sendall(registration_sa(uid="naive", callsign="naive", lat=None, lon=None, device="d", version="1").encode())
    s.sendall(spot(uid="x2").encode())  # rule 2: SA was too early, so this is still dropped
    assert wait_for(lambda: len(ots.dropped) == 3, timeout=3)
    assert not ots.accepted and not ots.registrations
    s.close()


# ------------------------------------------------------------------ the client


def test_client_registers_then_delivers_queued_report(ots):
    c = client(ots)
    try:
        assert c.send(spot())  # queued before registration completes
        assert not c.registered
        assert wait_for(lambda: ots.accepted, timeout=5)
        assert ots.registrations[0] == ("isrkit-fpv-vision-teststation", "FPV teststation")
        sender, ev = ots.accepted[0]
        assert sender == c.uid and 'uid="fpv-vision-spot-1"' in ev and "RECORDED: SALUTE" in ev
        assert not ots.dropped and c.status()["registered"] and c.connects == 1
    finally:
        c.stop()


def test_client_waits_before_registering(ots):
    c = client(ots, register_delay=0.6)
    try:
        time.sleep(0.3)
        assert c.connected and not c.registered and not ots.registrations
        assert wait_for(lambda: ots.registrations, timeout=3)
        assert not ots.dropped
    finally:
        c.stop()


def test_client_reconnects_and_re_registers(ots):
    c = client(ots)
    try:
        assert wait_for(lambda: c.registered, timeout=3)
        ots.kick_all()
        assert wait_for(lambda: not c.connected, timeout=3)
        assert c.send(spot(uid="after-reconnect"))
        assert wait_for(lambda: any('uid="after-reconnect"' in e for _, e in ots.accepted), timeout=8)
        assert c.connects == 2 and ots.sa_count(c.uid) == 2
        assert not ots.refused  # same uid keeps its callsign across reconnects
    finally:
        c.stop()


def test_client_survives_tak_down_at_startup():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    c = TakClient("127.0.0.1", port, station="teststation", register_delay=DELAY).start()
    try:
        assert c.send(spot())
        assert wait_for(lambda: c.last_error != "", timeout=3) and not c.connected
        srv = FakeOTS(channel_delay=0.25, port=port)
        try:
            assert wait_for(lambda: srv.accepted, timeout=10)  # backoff retry, register, deliver
        finally:
            srv.close()
    finally:
        c.stop()


def test_sa_is_resent_periodically(ots):
    c = client(ots, sa_interval=0.4)
    try:
        assert wait_for(lambda: ots.sa_count(c.uid) >= 3, timeout=4)
        assert not ots.refused
    finally:
        c.stop()


def test_callsign_taken_by_another_uid_is_refused(ots):
    squatter = TakClient("127.0.0.1", ots.port, station="teststation", register_delay=DELAY).start()
    squatter.uid = "someone-else"  # same callsign, different uid
    try:
        assert wait_for(lambda: ots.registrations, timeout=3)
        c = client(ots)
        try:
            assert wait_for(lambda: ots.refused, timeout=3)
            assert ots.refused[0] == ("isrkit-fpv-vision-teststation", "FPV teststation")
        finally:
            c.stop()
    finally:
        squatter.stop()


def test_rate_limit_per_uid_and_disabled():
    c = TakClient(enabled=True)  # not started: just queues
    assert c.send(spot(uid="a")) and c.send(spot(uid="b"))
    assert c.send(spot(uid="a")) is False and "rate limit" in c.last_error
    c._last_uid["a"] -= 6
    assert c.send(spot(uid="a"))
    assert TakClient(enabled=False).send(spot()) is False


# ------------------------------------------------------------------ C5.5 storability


def test_store_rules_of_the_fake():
    from fake_ots import store_rejection

    eud = {"isrkit-fpv-vision-teststation"}
    assert store_rejection(spot(), eud) is None
    assert "not an EUD" in store_rejection(spot(sender_uid="fpv-vision-spot-0"), eud)
    sa = registration_sa(uid="u", callsign="c", lat=42.0, lon=-93.0, device="d", version="1")
    assert "takv" in store_rejection(sa, eud) or "not stored" in store_rejection(sa, eud)
    zero = spot().replace('lat="42.0000000" lon="-93.0000000"', 'lat="0" lon="0"')
    assert store_rejection(zero, eud) == "no real position"
    assert "not stored" in store_rejection(spot().replace('type="a-u-G"', 'type="b-m-p-s-p-loc"'), eud)
    assert store_rejection(spot().replace('type="a-u-G"', 'type="b-m-p-s-m"'), eud) is None


def test_spot_report_marker_is_storable(ots):
    c = client(ots)
    try:
        assert c.send(spot())
        assert wait_for(lambda: ots.stored or ots.not_stored, timeout=5)
        assert ots.not_stored == [] and len(ots.stored) == 1
        ev = ET.fromstring(ots.stored[0].encode())
        link = ev.find("detail/link")
        assert link.get("uid") == c.uid and link.get("parent_callsign") == c.callsign
        assert ev.find("detail/contact").get("endpoint") is None and ev.find("detail/takv") is None
        assert ev.find("detail/precisionlocation").get("geopointsrc") == "USER"
    finally:
        c.stop()


def test_marker_without_registered_uid_has_no_link_and_refuses_0_0():
    xml = cot.spot_report_cot(lat=42.0, lon=-93.0, callsign="SPOT", remarks="r")
    assert "<link" not in xml
    with pytest.raises(ValueError):
        cot.spot_report_cot(lat=0.0, lon=0.0, callsign="SPOT", remarks="r")
