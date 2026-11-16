"""Cursor-on-Target output to the TAK server (OpenTAKServer plain CoT TCP, default 127.0.0.1:8087).

A background thread owns the socket: messages are queued, the thread connects (and
reconnects with exponential backoff) and drains the queue. Nothing here raises into the
caller if the TAK server is down — status is exposed for the UI instead.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from xml.sax.saxutils import escape, quoteattr

log = logging.getLogger(__name__)

QUEUE_TTL = 300  # drop queued reports older than this if TAK never comes back
UID_MIN_INTERVAL = 5.0  # contract C4.4: at most 1 CoT per 5 s per uid

_UID_RE = __import__("re").compile(r'<event[^>]*\suid="([^"]+)"')


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def spot_report_cot(*, lat: float, lon: float, callsign: str, remarks: str, uid: str | None = None,
                    cot_type: str = "a-u-G", stale_s: int = 3600, ce: float = 50.0,
                    sender: str = "FPV-VISION", now: float | None = None) -> str:
    """A ground marker at an *operator-entered* position (how="h-e": human entered)."""
    now = now or time.time()
    uid = uid or f"fpv-vision-spot-{uuid.uuid4().hex[:12]}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type={quoteattr(cot_type)} how="h-e" '
        f'time="{iso(now)}" start="{iso(now)}" stale="{iso(now + stale_s)}">'
        f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="9999999.0" ce="{ce:.1f}" le="9999999.0"/>'
        "<detail>"
        f"<contact callsign={quoteattr(callsign)}/>"
        f"<remarks source={quoteattr(sender)}>{escape(remarks)}</remarks>"
        f'<link uid={quoteattr(sender)} relation="p-p" type="a-f-G-U-C"/>'
        '<archive/>'
        "</detail></event>"
    )


class CotSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 8087, enabled: bool = True):
        self.host, self.port, self.enabled = host, port, enabled
        self._q: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self.connected = False
        self.last_error = ""
        self.sent = 0
        self._thread: threading.Thread | None = None
        self._last_uid: dict[str, float] = {}

    def start(self) -> "CotSender":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="cot", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._close()

    def send(self, xml: str) -> bool:
        """Queue a CoT event; returns False only if disabled or the queue is full."""
        if not self.enabled:
            return False
        m = _UID_RE.search(xml)
        if m:
            uid, now = m.group(1), time.time()
            if now - self._last_uid.get(uid, 0.0) < UID_MIN_INTERVAL:
                self.last_error = f"CoT rate limit: uid {uid} sent < {UID_MIN_INTERVAL:.0f} s ago"
                return False
            self._last_uid[uid] = now
            if len(self._last_uid) > 1000:
                self._last_uid = {k: v for k, v in self._last_uid.items() if now - v < UID_MIN_INTERVAL}
        try:
            self._q.put_nowait((time.time(), xml.encode("utf-8")))
            return True
        except queue.Full:
            self.last_error = "queue full"
            return False

    def status(self) -> dict:
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "connected": self.connected,
                "queued": self._q.qsize(), "sent": self.sent, "error": self.last_error}

    def _connect(self) -> bool:
        try:
            s = socket.create_connection((self.host, self.port), timeout=5)
            s.settimeout(10)
            self._sock = s
            self.connected, self.last_error = True, ""
            log.info("TAK connected %s:%d", self.host, self.port)
            return True
        except OSError as e:
            self.connected, self.last_error = False, str(e)
            return False

    def _close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self.connected = False

    def _run(self) -> None:
        backoff = 1.0
        pending: tuple[float, bytes] | None = None
        while not self._stop.is_set():
            if pending is None:
                try:
                    pending = self._q.get(timeout=1.0)
                except queue.Empty:
                    continue
            if time.time() - pending[0] > QUEUE_TTL:
                log.warning("dropping stale CoT message (TAK unreachable for %ds)", QUEUE_TTL)
                pending = None
                continue
            if self._sock is None and not self._connect():
                log.info("TAK %s:%d unavailable (%s); retry in %.0fs", self.host, self.port, self.last_error, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30)
                continue
            backoff = 1.0
            try:
                assert self._sock is not None
                self._sock.sendall(pending[1])
                self.sent += 1
                pending = None
            except OSError as e:
                self.last_error = str(e)
                log.warning("TAK send failed: %s", e)
                self._close()
