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
                    sender: str = "FPV-VISION", sender_uid: str | None = None, sender_callsign: str | None = None,
                    now: float | None = None, how: str = "h-e") -> str:
    """A ground marker at an *operator-entered* position (how="h-e": human entered).

    Built so OpenTAKServer 1.8.3 both relays AND stores it (contract C5.5):
      * type ``a-u-G`` matches ``^a-[fhupansjk]-[ZPAGSUF]``;
      * no ``<takv>`` and no ``<contact endpoint=...>`` (those make it an EUD, not a marker);
      * a real position: 0,0 raises ValueError (contract C4.2);
      * ``<link>`` only to our own registered EUD uid (``sender_uid``), never another marker.
    """
    if lat == 0 and lon == 0:
        raise ValueError("refusing to build a CoT marker at 0,0")
    now = now or time.time()
    uid = uid or f"fpv-vision-spot-{uuid.uuid4().hex[:12]}"
    t = iso(now)
    detail = [
        f"<contact callsign={quoteattr(callsign)}/>",
        f"<remarks source={quoteattr(sender)}>{escape(remarks)}</remarks>",
    ]
    if sender_uid:  # markers.parent_uid -> euds.uid: must be the registered EUD
        attrs = f'uid={quoteattr(sender_uid)} relation="p-p" type="a-f-G-E-S" production_time="{t}"'
        if sender_callsign:
            attrs += f" parent_callsign={quoteattr(sender_callsign)}"
        detail.append(f"<link {attrs}/>")
    detail.append('<precisionlocation altsrc="USER" geopointsrc="USER"/>')  # operator-entered
    detail.append("<archive/>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<event version="2.0" uid={quoteattr(uid)} type={quoteattr(cot_type)} how={quoteattr(how)} '
        f'time="{t}" start="{t}" stale="{iso(now + stale_s)}">'
        f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="9999999.0" ce="{ce:.1f}" le="9999999.0"/>'
        f"<detail>{''.join(detail)}</detail></event>"
    )
