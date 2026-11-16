"""A tiny fake OpenTAKServer CoT/TCP endpoint that enforces the three behaviours measured on
OpenTAKServer 1.8.3 (contract C5):

1. Events from a connection that has not registered an EUD are dropped
   (Postgres FK on cot.sender_uid).
2. A connection's queue is only bound if its FIRST <contact ... endpoint=...> SA arrives at
   least ``channel_delay`` seconds after connect; an earlier first SA leaves the connection
   permanently unregistered.
3. An SA whose callsign is already held by a different uid is silently refused.
"""

from __future__ import annotations

import re
import socket
import threading
import time

_EVENT_RE = re.compile(r"<event\b.*?</event>", re.S)
_UID_RE = re.compile(r"""<event\b[^>]*\buid=(["'])(.*?)\1""")
_SA_RE = re.compile(r"""<contact\b[^>]*\bcallsign=(["'])(.*?)\1[^>]*\bendpoint=""")


class FakeOTS:
    def __init__(self, channel_delay: float = 0.25, port: int = 0):
        self.channel_delay = channel_delay
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.callsigns: dict[str, str] = {}  # callsign -> uid
        self.accepted: list[tuple[str, str]] = []  # (sender uid, event xml)
        self.dropped: list[tuple[str, str]] = []  # (reason, event xml)
        self.registrations: list[tuple[str, str]] = []  # (uid, callsign)
        self.refused: list[tuple[str, str]] = []
        self.connections = 0
        self._conns: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self.kick_all()
        try:
            self.srv.close()
        except OSError:
            pass

    def kick_all(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
                c.close()
            except OSError:
                pass

    def sa_count(self, uid: str) -> int:
        return sum(1 for u, _ in self.registrations if u == uid)

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            with self._lock:
                self._conns.append(c)
                self.connections += 1
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c: socket.socket) -> None:
        t0 = time.time()
        state = {"uid": None, "broken": False, "first_sa": True}
        buf = ""
        try:
            # like OTS, push other traffic at the client: a client that never reads would stall
            c.sendall(b'<event version="2.0" uid="other-eud" type="a-f-G" how="m-g"/>' * 50)
            while True:
                data = c.recv(65536)
                if not data:
                    return
                buf += data.decode("utf-8", "replace")
                while True:
                    m = _EVENT_RE.search(buf)
                    if not m:
                        break
                    self._handle(m.group(0), state, t0)
                    buf = buf[m.end():]
        except OSError:
            return

    def _handle(self, ev: str, state: dict, t0: float) -> None:
        um = _UID_RE.search(ev)
        uid = um.group(2) if um else ""
        sa = _SA_RE.search(ev)
        with self._lock:
            if sa:
                callsign = sa.group(2)
                if state["first_sa"]:
                    state["first_sa"] = False
                    if time.time() - t0 < self.channel_delay:
                        state["broken"] = True  # rule 2: queue never bound on this connection
                if state["broken"]:
                    self.dropped.append(("sa-before-channel", ev))
                    return
                owner = self.callsigns.get(callsign)
                if owner is not None and owner != uid:
                    self.refused.append((uid, callsign))  # rule 3
                    return
                self.callsigns[callsign] = uid
                state["uid"] = uid
                self.registrations.append((uid, callsign))
                return
            if state["uid"] is None:
                self.dropped.append(("unregistered-sender", ev))  # rule 1
                return
            self.accepted.append((state["uid"], ev))
