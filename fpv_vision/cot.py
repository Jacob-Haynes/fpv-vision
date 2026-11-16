"""Cursor-on-Target spot reports. Delivery goes through the registered, long-lived
:class:`fpv_vision.tak_client.TakClient` (contract C5); ``CotSender`` is kept as an alias."""

from __future__ import annotations

import time
import uuid
from xml.sax.saxutils import escape, quoteattr

from fpv_vision.tak_client import QUEUE_TTL, UID_MIN_INTERVAL, TakClient, iso  # noqa: F401

CotSender = TakClient


def spot_report_cot(*, lat: float, lon: float, callsign: str, remarks: str, uid: str | None = None,
                    cot_type: str = "a-u-G", stale_s: int = 3600, ce: float = 50.0,
                    sender: str = "FPV-VISION", sender_uid: str | None = None, now: float | None = None) -> str:
    """A ground marker at an *operator-entered* position (how="h-e": human entered).

    ``sender_uid`` is the registered TAK client uid (the EUD OTS knows), used in ``<link>``.
    """
    now = now or time.time()
    uid = uid or f"fpv-vision-spot-{uuid.uuid4().hex[:12]}"
    link_uid = sender_uid or sender
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type={quoteattr(cot_type)} how="h-e" '
        f'time="{iso(now)}" start="{iso(now)}" stale="{iso(now + stale_s)}">'
        f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="9999999.0" ce="{ce:.1f}" le="9999999.0"/>'
        "<detail>"
        f"<contact callsign={quoteattr(callsign)}/>"
        f"<remarks source={quoteattr(sender)}>{escape(remarks)}</remarks>"
        f'<link uid={quoteattr(link_uid)} relation="p-p" type="a-f-G-E-S"/>'
        '<archive/>'
        "</detail></event>"
    )
