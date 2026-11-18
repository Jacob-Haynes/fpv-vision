import json
import xml.etree.ElementTree as ET

from fpv_vision import cot
from fpv_vision.mesh import TX_TOPIC, MeshBus

from conftest import wait_for


def test_spot_report_cot_is_valid_xml():
    xml = cot.spot_report_cot(lat=42.0, lon=-93.0, callsign="SPOT <1>", remarks="2 pax & 1 veh", now=1794832200)
    root = ET.fromstring(xml.encode())
    assert root.tag == "event" and root.get("type") == "a-u-G" and root.get("how") == "h-e"
    pt = root.find("point")
    assert float(pt.get("lat")) == 42.0 and float(pt.get("lon")) == -93.0
    assert root.find("detail/remarks").text == "2 pax & 1 veh"
    assert root.find("detail/contact").get("callsign") == "SPOT <1>"
    assert root.get("time").startswith("2026-11-16T12:30:00")


def test_cot_disabled_never_queues():
    sender = cot.CotSender(enabled=False).start()
    assert sender.send("<event/>") is False


class _FakeInfo:
    rc = 0


class _FakeClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, body, qos=0):
        self.published.append((topic, json.loads(body)))
        return _FakeInfo()


def test_mesh_send_publishes_contract_json():
    bus = MeshBus(enabled=True)
    bus._client = _FakeClient()
    bus.connected = True
    assert bus.send("SPOTREP test")
    topic, body = bus._client.published[0]
    assert topic == TX_TOPIC
    assert body == {"text": "SPOTREP test", "to": None, "channel": 0, "source": "fpv-vision"}


def test_mesh_offline_does_not_raise():
    bus = MeshBus(host="127.0.0.1", port=1, enabled=True).start()  # nothing listens on port 1
    try:
        assert bus.send("hello") is False
    finally:
        bus.stop()


def test_mesh_rx_parsing():
    bus = MeshBus(enabled=False)

    class Msg:
        payload = json.dumps({"from": "!abcd1234", "text": "copy", "channel": 0, "ts": 1}).encode()

    bus._on_message(None, None, Msg())
    bus._on_message(None, None, type("M", (), {"payload": b"not json"})())
    assert list(bus.rx) == [{"from": "!abcd1234", "text": "copy", "channel": 0, "ts": 1}]
