"""Long-lived, registered CoT client for OpenTAKServer (contract C5).

Measured on OpenTAKServer 1.8.3, which drops CoT unless the sender follows three rules:

1. **Registered sender.** OTS stores every CoT with a foreign key to a known EUD
   (``cot.sender_uid``). Events from a connection that never announced itself with a
   ``<contact>`` SA are dropped. Fire-and-forget one-shot connections never reach ATAK.
2. **Register after the channel is ready.** OTS only binds a client's queue if the first
   ``<contact>`` event arrives after its async channel opens (~3 s after connect). So we
   wait ``register_delay`` seconds after every (re)connect before sending the SA.
3. **Stable identity.** OTS silently refuses a uid whose callsign is already taken by
   another uid. Our uid and callsign derive from ``ISR_HOST`` and never change.

So this client:
* keeps ONE TCP connection to TAK_HOST:TAK_PORT, reconnecting with exponential backoff;
* on every (re)connect waits ``register_delay``, sends the registration SA
  (uid ``isrkit-<tool>-<host>``, callsign ``<label> <host>``, type ``a-f-G-E-S``), and
  re-sends it every ``sa_interval`` seconds;
* queues other CoT (spot reports) and only sends it on that connection after
  registration;
* drains whatever OTS sends back, so the server never blocks on a full socket buffer;
* rate-limits to one event per ``uid_min_interval`` seconds per uid (contract C4.4).

Self-contained: standard library only.
"""

from __future__ import annotations

import logging
import platform
import queue
import re
import socket
import threading
import time
from datetime import datetime, timezone
from xml.sax.saxutils import quoteattr

log = logging.getLogger(__name__)

REGISTER_DELAY = 3.0  # OTS opens the client's async channel ~3 s after connect
SA_INTERVAL = 60.0  # re-send the registration SA this often
SA_STALE = 120.0  # SA stale time when we have a position
QUEUE_TTL = 300.0  # drop queued events older than this if TAK never comes back
UID_MIN_INTERVAL = 5.0  # contract C4.4: at most 1 CoT per 5 s per uid
UNKNOWN = 9999999  # CoT "unknown" value for hae/ce/le

_UID_RE = re.compile(r"""<event\b[^>]*\buid=(["'])(.*?)\1""")


def iso(ts: float) -> str:
    """ISO 8601 UTC with milliseconds, ending in Z (e.g. 2026-11-16T12:30:00.000Z)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def event_uid(xml: str) -> str | None:
    m = _UID_RE.search(xml)
    return m.group(2) if m else None


def registration_sa(*, uid: str, callsign: str, lat: float | None, lon: float | None, device: str,
                    version: str, now: float | None = None, stale_s: float = SA_STALE) -> str:
    """Registration SA (contract C5.2).

    With a position: a normal sensor point, stale now+``stale_s``. Without one: the only
    allowed 0,0 in the kit, with unknown hae/ce/le and stale == start so ATAK never draws it.
    """
    now = time.time() if now is None else now
    if lat is not None and lon is not None and not (lat == 0 and lon == 0):
        point = f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="{UNKNOWN}" ce="50" le="{UNKNOWN}"/>'
        stale = now + stale_s
    else:
        point = f'<point lat="0" lon="0" hae="{UNKNOWN}" ce="{UNKNOWN}" le="{UNKNOWN}"/>'
        stale = now
    os_name = f"{platform.system()} {platform.release()}".strip()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type="a-f-G-E-S" how="m-g" '
        f'time="{iso(now)}" start="{iso(now)}" stale="{iso(stale)}">'
        f"{point}"
        "<detail>"
        f'<contact callsign={quoteattr(callsign)} endpoint="*:-1:stcp"/>'
        '<__group name="Cyan" role="Team Member"/>'
        f'<takv platform="isr-kit" device={quoteattr(device)} os={quoteattr(os_name)} version={quoteattr(version)}/>'
        "</detail></event>"
    )


class TakClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8087, enabled: bool = True, *,
                 tool: str = "fpv-vision", label: str = "FPV", station: str = "localhost",
                 lat: float | None = None, lon: float | None = None, version: str = "0",
                 register_delay: float = REGISTER_DELAY, sa_interval: float = SA_INTERVAL,
                 uid_min_interval: float = UID_MIN_INTERVAL, connect_timeout: float = 5.0):
        self.host, self.port, self.enabled = host, port, enabled
        self.tool, self.station = tool, station.lower()
        self.uid = f"isrkit-{tool}-{self.station}"
        self.callsign = f"{label} {self.station}"
        self.lat, self.lon, self.version = lat, lon, version
        self.register_delay, self.sa_interval = register_delay, sa_interval
        self.uid_min_interval, self.connect_timeout = uid_min_interval, connect_timeout
        self._q: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._last_uid: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self.connected = False
        self.registered = False
        self.last_error = ""
        self.sent = 0
        self.sa_sent = 0
        self.connects = 0

    # ---------------------------------------------------------------- public API
    def start(self) -> "TakClient":
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="tak", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._close()
        if self._thread:
            self._thread.join(timeout=3)

    def send(self, xml: str, force: bool = False) -> bool:
        """Queue a CoT event for delivery after registration. False if disabled, rate-limited or full.

        ``force`` skips the per-uid limit, for the last word on a marker (stale-now before a delete)."""
        if not self.enabled:
            return False
        uid, now = event_uid(xml), time.time()
        if uid and not force:
            if now - self._last_uid.get(uid, 0.0) < self.uid_min_interval:
                self.last_error = f"CoT rate limit: uid {uid} sent < {self.uid_min_interval:.0f} s ago"
                return False
            self._last_uid[uid] = now
            if len(self._last_uid) > 1000:
                self._last_uid = {k: v for k, v in self._last_uid.items() if now - v < self.uid_min_interval}
        try:
            self._q.put_nowait((now, xml.encode("utf-8")))
            return True
        except queue.Full:
            self.last_error = "queue full"
            return False

    def resend_sa(self) -> bool:
        """Queue our registration SA behind what is already queued: OpenTAKServer marks the sender
        disconnected on a t-x-d-d, so it goes out again straight after the deletes."""
        if not self.enabled:
            return False
        try:
            self._q.put_nowait((time.time(), self._sa()))
            return True
        except queue.Full:
            return False

    def status(self) -> dict:
        return {"enabled": self.enabled, "host": self.host, "port": self.port, "connected": self.connected,
                "registered": self.registered, "uid": self.uid, "callsign": self.callsign,
                "queued": self._q.qsize(), "sent": self.sent, "sa_sent": self.sa_sent,
                "connects": self.connects, "error": self.last_error}

    # ---------------------------------------------------------------- internals
    def _sa(self) -> bytes:
        return registration_sa(uid=self.uid, callsign=self.callsign, lat=self.lat, lon=self.lon,
                               device=self.tool, version=self.version).encode("utf-8")

    def _connect(self) -> bool:
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            s.settimeout(10)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError as e:
            self.connected, self.last_error = False, str(e)
            return False
        self._sock = s
        self.connected, self.registered = True, False
        self.connects += 1
        self.last_error = ""
        threading.Thread(target=self._drain, args=(s,), name="tak-rx", daemon=True).start()
        log.info("TAK connected %s:%d (registering as %s / %s in %.0fs)", self.host, self.port, self.uid,
                 self.callsign, self.register_delay)
        return True

    def _drain(self, s: socket.socket) -> None:
        """Read and discard what OTS sends us; notice when it closes the connection."""
        while not self._stop.is_set():
            try:
                s.settimeout(1.0)
                data = s.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
        if self._sock is s:
            self.last_error = self.last_error or "TAK closed the connection"
            self._close()

    def _close(self) -> None:
        s, self._sock = self._sock, None
        if s:
            try:
                s.close()
            except OSError:
                pass
        self.connected = self.registered = False

    def _write(self, data: bytes) -> bool:
        s = self._sock
        if s is None:
            return False
        try:
            with self._send_lock:
                s.settimeout(10)
                s.sendall(data)
            return True
        except OSError as e:
            self.last_error = f"send failed: {e}"
            log.warning("TAK send failed: %s", e)
            self._close()
            return False

    def _register(self) -> bool:
        if self._stop.wait(self.register_delay) or self._sock is None:
            return False
        if self._write(self._sa()):
            self.registered = True
            self.sa_sent += 1
            log.info("TAK registered as %s (%s)", self.uid, self.callsign)
            return True
        return False

    def _run(self) -> None:
        backoff = 1.0
        pending: tuple[float, bytes] | None = None
        while not self._stop.is_set():
            if self._sock is None:
                if not self._connect():
                    log.info("TAK %s:%d unavailable (%s); retry in %.0fs", self.host, self.port, self.last_error,
                             backoff)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                backoff = 1.0
                if not self._register():
                    continue
                last_sa = time.time()
            if time.time() - last_sa >= self.sa_interval:
                if not self._write(self._sa()):
                    continue
                self.sa_sent += 1
                last_sa = time.time()
            if pending is None:
                try:
                    pending = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
            if time.time() - pending[0] > QUEUE_TTL:
                log.warning("dropping stale CoT (TAK unreachable for %.0fs)", QUEUE_TTL)
                pending = None
                continue
            if self.registered and self._write(pending[1]):
                self.sent += 1
                pending = None
