"""Shared MQTT bus: Meshtastic text (never the radio's serial port) + isr-kit heartbeat.

The blufor-tracker bridge owns the radio. We publish ``{"text", "to", "channel", "source"}``
to ``meshbus/tx`` and listen on ``meshbus/rx`` so the UI can show replies.

Every 10 s we publish a retained heartbeat to ``isrkit/heartbeat/fpv-vision`` so the Pi
launcher can link to this tool wherever it runs (Pi or Mac). On a clean exit we publish a
retained ``stopped``; the MQTT last-will marks an unclean drop as ``error``.

Remote machines (the Macs) must use the ``isrkit`` MQTT user: RabbitMQ's ``guest`` account
only works over loopback on the Pi.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

log = logging.getLogger(__name__)

TX_TOPIC = "meshbus/tx"
RX_TOPIC = "meshbus/rx"
HEARTBEAT_TOPIC = "isrkit/heartbeat/fpv-vision"
HEARTBEAT_PERIOD = 10.0
MAX_TEXT = 199


class MeshBus:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883, user: str = "guest", password: str = "guest",
                 enabled: bool = True, source: str = "fpv-vision", mesh_enabled: bool = True,
                 heartbeat_fn=None):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.enabled = enabled  # connect to the broker at all
        self.mesh_enabled = mesh_enabled  # allow meshbus/tx sends
        self.source = source
        self.heartbeat_fn = heartbeat_fn  # () -> dict; None disables the heartbeat
        self.heartbeats = 0
        self._hb_stop = threading.Event()
        self.connected = False
        self.last_error = ""
        self.sent = 0
        self.rx: deque[dict] = deque(maxlen=20)
        self._client = None
        self._lock = threading.Lock()

    def start(self) -> "MeshBus":
        if not self.enabled:
            return self
        try:
            import paho.mqtt.client as mqtt

            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fpv-vision-{int(time.time())}")
            c.username_pw_set(self.user, self.password)
            if self.heartbeat_fn:
                try:
                    will = dict(self.heartbeat_fn(), status="error", ts=int(time.time()))
                    c.will_set(HEARTBEAT_TOPIC, json.dumps(will), qos=1, retain=True)
                except Exception:  # noqa: BLE001
                    pass
            c.on_connect = self._on_connect
            c.on_disconnect = self._on_disconnect
            c.on_message = self._on_message
            c.reconnect_delay_set(min_delay=1, max_delay=30)
            c.connect_async(self.host, self.port, keepalive=30)
            c.loop_start()
            self._client = c
            if self.heartbeat_fn:
                threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        except Exception as e:  # noqa: BLE001 - broker optional
            self.last_error = str(e)
            log.warning("mesh bus unavailable: %s", e)
        return self

    def _publish_heartbeat(self, status: str | None = None) -> bool:
        if self._client is None or not self.connected or not self.heartbeat_fn:
            return False
        try:
            hb = self.heartbeat_fn()
            if status:
                hb["status"] = status
            hb["ts"] = int(time.time())
            info = self._client.publish(HEARTBEAT_TOPIC, json.dumps(hb), qos=1, retain=True)
            if info.rc == 0:
                self.heartbeats += 1
                return True
        except Exception as e:  # noqa: BLE001
            log.debug("heartbeat failed: %s", e)
        return False

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.is_set():
            self._publish_heartbeat()
            self._hb_stop.wait(HEARTBEAT_PERIOD)

    def stop(self) -> None:
        self._hb_stop.set()
        if self._client is not None:
            try:
                if self.connected and self.heartbeat_fn:
                    hb = dict(self.heartbeat_fn(), status="stopped", ts=int(time.time()))
                    info = self._client.publish(HEARTBEAT_TOPIC, json.dumps(hb), qos=1, retain=True)
                    info.wait_for_publish(timeout=2)
                # disconnect while the network loop still runs, so the broker sees a clean
                # DISCONNECT and does not fire the 'error' last-will
                self._client.disconnect()
                time.sleep(0.2)
                self._client.loop_stop()
            except Exception:  # noqa: BLE001
                pass

    # paho callbacks
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            self.connected, self.last_error = False, str(reason_code)
            return
        self.connected, self.last_error = True, ""
        client.subscribe(RX_TOPIC)
        log.info("MQTT connected %s:%d as %s", self.host, self.port, self.user)
        threading.Thread(target=self._publish_heartbeat, daemon=True).start()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False
        if reason_code is not None and str(reason_code) not in ("Success", "Normal disconnection"):
            self.last_error = str(reason_code)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if isinstance(data, dict) and "text" in data:
                data.setdefault("ts", time.time())
                self.rx.appendleft(data)
        except (ValueError, UnicodeDecodeError):
            pass

    @staticmethod
    def payload(text: str, to: str | None = None, channel: int = 0, source: str = "fpv-vision") -> dict:
        return {"text": text[:MAX_TEXT], "to": to, "channel": channel, "source": source}

    def send(self, text: str, to: str | None = None, channel: int = 0) -> bool:
        if not self.enabled or not self.mesh_enabled or self._client is None:
            log.info("mesh disabled/unavailable; not sent: %s", text)
            return False
        if not self.connected:
            log.warning("mesh bus not connected; not sent: %s", text)
            self.last_error = "broker not connected"
            return False
        body = json.dumps(self.payload(text, to, channel, self.source))
        try:
            info = self._client.publish(TX_TOPIC, body, qos=1)
            ok = info.rc == 0
            if ok:
                self.sent += 1
            return ok
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            log.warning("mesh publish failed: %s", e)
            return False

    def status(self) -> dict:
        return {"enabled": self.enabled and self.mesh_enabled, "host": self.host, "port": self.port,
                "user": self.user, "connected": self.connected, "sent": self.sent, "error": self.last_error,
                "heartbeats": self.heartbeats, "rx": list(self.rx)[:10]}
