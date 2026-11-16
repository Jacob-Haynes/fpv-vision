"""Meshtastic text via the shared MQTT bus (never the radio's serial port).

The blufor-tracker bridge owns the radio. We publish ``{"text", "to", "channel", "source"}``
to ``meshbus/tx`` and listen on ``meshbus/rx`` so the UI can show replies.
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
MAX_TEXT = 199


class MeshBus:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883, user: str = "guest", password: str = "guest",
                 enabled: bool = True, source: str = "fpv-vision"):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.enabled = enabled
        self.source = source
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
            c.on_connect = self._on_connect
            c.on_disconnect = self._on_disconnect
            c.on_message = self._on_message
            c.reconnect_delay_set(min_delay=1, max_delay=30)
            c.connect_async(self.host, self.port, keepalive=30)
            c.loop_start()
            self._client = c
        except Exception as e:  # noqa: BLE001 - broker optional
            self.last_error = str(e)
            log.warning("mesh bus unavailable: %s", e)
        return self

    def stop(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # paho callbacks
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            self.connected, self.last_error = False, str(reason_code)
            return
        self.connected, self.last_error = True, ""
        client.subscribe(RX_TOPIC)
        log.info("mesh bus connected %s:%d", self.host, self.port)

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
        if not self.enabled or self._client is None:
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
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "connected": self.connected,
                "sent": self.sent, "error": self.last_error, "rx": list(self.rx)[:10]}
