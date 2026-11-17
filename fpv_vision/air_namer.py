"""Name Air watch tracks with the local vision model (VLM).

YOLOE finds and tracks air objects well but names multirotors "helicopter" or "aeroplane".
So once a non-bird track with confidence >= ``min_conf`` has been stable for ``stable_s``, its box
is cropped (with padding, at least ``min_side`` px on the short side) and sent to the configured
OpenAI-compatible VLM (qwen3-vl:8b-instruct on Ollama). The prompt asks for JSON:

    {"class": "drone|helicopter|aeroplane|bird|other", "confidence": 0-1, "detail": "<=6 words"}

The answer replaces the YOLOE vote for that track (``TrackCounter.set_override``):
* a confident answer (>= ``accept_conf``) names it, e.g. "drone" with detail "hexacopter";
* a low-confidence answer, or "other", makes it an "air contact" (APP-6 unknown);
* no answer yet, or the VLM unreachable: the YOLOE vote stays.

Re-asked at most ``max_asks - 1`` times per track (unnamed or low confidence only), at least
``reask_gap`` s apart, with a larger crop. One call in flight, a queue of at most ``queue_cap``
(newest tracks first; the oldest is dropped), ``timeout`` s per call. Everything runs on its own
thread: the video and detection loops only call the cheap :meth:`observe` / :meth:`apply`.
"""

from __future__ import annotations

import base64
import collections
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from fpv_vision import air
from fpv_vision.detector import Detection

log = logging.getLogger(__name__)

CLASSES = ("drone", "helicopter", "aeroplane", "bird", "other")
PROMPT = (
    "You are helping an ISTAR observation post identify an object in the sky. Look at the object in the "
    "middle of this image. Reply with JSON only, no other text: "
    '{"class": "drone|helicopter|aeroplane|bird|other", "confidence": <0 to 1>, '
    '"detail": "<at most 6 words, e.g. hexacopter, quadcopter, airliner, gull>"}. '
    "drone = any unmanned multirotor or small fixed-wing UAV; helicopter = a crewed rotorcraft; "
    "aeroplane = a crewed fixed-wing aircraft. If unsure, lower the confidence."
)
SCHEMA = {
    "type": "object",
    "properties": {"class": {"type": "string", "enum": list(CLASSES)}, "confidence": {"type": "number"},
                   "detail": {"type": "string"}},
    "required": ["class", "confidence", "detail"],
}
VLM_CLASS = {"drone": (air.DRONE, "drone"), "helicopter": (air.HELICOPTER, "helicopter"),
             "aeroplane": (air.AEROPLANE, "aeroplane"), "bird": (air.BIRD, "bird")}
CONTACT = (air.AIR_CONTACT, "air contact")


def parse_answer(text: str) -> dict:
    """Lenient: the first {...} in the reply; class normalised; confidence clamped to 0-1."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:80]!r}")
    raw = m.group(0)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = json.loads(re.sub(r",\s*}", "}", raw.replace("'", '"')))
    cls = str(d.get("class", "other")).strip().lower()
    cls = {"airplane": "aeroplane", "plane": "aeroplane", "uav": "drone", "quadcopter": "drone",
           "multirotor": "drone", "aircraft": "aeroplane"}.get(cls, cls)
    if cls not in CLASSES:
        cls = "other"
    try:
        conf = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = conf / 100 if conf > 1 else conf
    detail = " ".join(str(d.get("detail", "")).split()[:6])
    return {"class": cls, "confidence": max(0.0, min(1.0, conf)), "detail": detail}


def to_label(answer: dict, accept_conf: float = 0.5) -> tuple[int, str, str]:
    """VLM answer -> (cls_id, label, detail). Low confidence or 'other' -> air contact."""
    if answer["class"] == "other" or answer["confidence"] < accept_conf:
        return CONTACT[0], CONTACT[1], answer.get("detail", "")
    cls_id, label = VLM_CLASS[answer["class"]]
    return cls_id, label, answer.get("detail", "")


def crop(frame: np.ndarray, xyxy, pad: float = 0.3, min_side: int = 224) -> bytes:
    """JPEG of the box with ``pad`` padding each side, upscaled so the short side >= ``min_side``."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = max(2, x2 - x1), max(2, y2 - y1)
    side = max(bw, bh) * (1 + 2 * pad)  # square-ish context around small targets
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = max(side, bw * (1 + 2 * pad)) / 2, max(side, bh * (1 + 2 * pad)) / 2
    xa, ya = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    xb, yb = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    img = frame[ya:yb, xa:xb]
    short = min(img.shape[:2])
    if short and short < min_side:
        s = min_side / short
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else b""


def _is_unreachable(e: Exception) -> bool:
    """Connection refused / unknown host = unreachable (back off). A timeout is just slow."""
    if isinstance(e, (TimeoutError, json.JSONDecodeError, ValueError)) or "timed out" in str(e):
        return False
    if isinstance(e, urllib.error.HTTPError):
        return False
    return isinstance(e, (urllib.error.URLError, ConnectionError, OSError))


class VlmClient:
    """OpenAI-compatible chat call for one crop. Structured JSON output when the server accepts it."""

    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 10.0):
        self.base = (url or "").rstrip("/")
        for suffix in ("/v1/chat/completions", "/v1", "/api/chat", "/api"):
            if self.base.endswith(suffix):
                self.base = self.base[: -len(suffix)]
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.structured = True  # cleared if the server rejects response_format

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    def __call__(self, jpeg: bytes) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
            ]}],
            "max_tokens": 80, "temperature": 0, "reasoning_effort": "none",  # reasoning off
        }
        if self.structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "air_object", "schema": SCHEMA}}
        try:
            return parse_answer(self._post(body))
        except urllib.error.HTTPError as e:
            if self.structured and e.code in (400, 422, 500, 501):  # e.g. MLX runner: no format support
                log.info("VLM %s rejected structured output (HTTP %s); parsing plain JSON", self.model, e.code)
                self.structured = False
                body.pop("response_format", None)
                return parse_answer(self._post(body))
            raise

    def warm(self) -> None:
        """Load the model before the first real question (Ollama loads it on first use: seconds)."""
        tiny = cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))[1].tobytes()
        body = {"model": self.model, "max_tokens": 1, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ok"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(tiny).decode()}}]}]}
        old, self.timeout = self.timeout, 60.0
        try:
            self._post(body)
        finally:
            self.timeout = old

    def _post(self, body: dict) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


@dataclass
class _Track:
    first_box: float
    stable_since: float
    asks: int = 0
    last_ask: float = 0.0
    pending: bool = False
    name: tuple[int, str, str] | None = None  # (cls_id, label, detail) once the VLM answered
    answer: dict | None = None
    named_at: float = 0.0
    history: list = field(default_factory=list)


class AirNamer:
    def __init__(self, client, counter, *, min_conf: float = 0.7, stable_s: float = 0.5, accept_conf: float = 0.5,
                 max_asks: int = 3, reask_gap: float = 3.0, queue_cap: int = 3, on_named=None,
                 unreachable_backoff: float = 15.0):
        self.client, self.counter = client, counter
        self.min_conf, self.stable_s, self.accept_conf = min_conf, stable_s, accept_conf
        self.max_asks, self.reask_gap, self.queue_cap = max_asks, reask_gap, queue_cap
        self.on_named = on_named  # callback(track_id, old_label, new_label, detail)
        self.unreachable_backoff = unreachable_backoff
        self.tracks: dict[int, _Track] = {}
        self.queue: collections.deque = collections.deque()
        self.dropped = 0
        self.calls = self.failures = 0
        self.in_flight = 0
        self.unreachable_until = 0.0
        self.last_error = ""
        self.latencies: collections.deque = collections.deque(maxlen=20)
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "AirNamer":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="air-namer", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def reset(self) -> None:
        with self._cv:
            self.tracks.clear()
            self.queue.clear()

    # ------------------------------------------------------------ called from the air loop (cheap)
    def observe(self, dets: list[Detection], frame: np.ndarray, now: float | None = None) -> None:
        """Queue crops of tracks that are ready to be named (or re-asked)."""
        now = time.time() if now is None else now
        seen = set()
        for d in dets:
            if d.track_id is None or d.cls_id < air.AIR_BASE:
                continue
            seen.add(d.track_id)
            t = self.tracks.get(d.track_id)
            if t is None:
                t = self.tracks[d.track_id] = _Track(first_box=now, stable_since=now)
            eligible = d.label != "bird" or t.name is not None  # birds (by YOLOE) are never sent
            if not eligible or d.conf < self.min_conf:
                t.stable_since = now  # stability counts confident, non-bird frames only
                continue
            if t.pending or now - t.stable_since < self.stable_s or now < self.unreachable_until:
                continue
            confident = t.name is not None and t.name[1] != CONTACT[1]
            if t.asks >= self.max_asks or confident or (t.asks and now - t.last_ask < self.reask_gap):
                continue
            larger = t.asks > 0  # re-ask with a larger, clearer crop
            jpeg = crop(frame, d.xyxy, pad=0.6 if larger else 0.3, min_side=448 if larger else 224)
            self._enqueue(d.track_id, jpeg, now)
        for tid in [k for k, t in self.tracks.items() if k not in seen and not t.pending and now - t.first_box > 120]:
            self.tracks.pop(tid, None)  # forget long-gone tracks

    def _enqueue(self, tid: int, jpeg: bytes, now: float) -> None:
        with self._cv:
            t = self.tracks[tid]
            t.pending, t.asks, t.last_ask = True, t.asks + 1, now
            self.queue.appendleft((tid, jpeg, now))  # newest first
            while len(self.queue) > self.queue_cap:
                old_tid, _, _ = self.queue.pop()  # drop the oldest
                self.dropped += 1
                if old_tid in self.tracks:
                    ot = self.tracks[old_tid]
                    ot.pending, ot.asks = False, max(0, ot.asks - 1)  # it was never asked
            self._cv.notify()

    def apply(self, dets: list[Detection]) -> list[Detection]:
        """Boxes with the VLM's name and detail ("drone", "hexacopter") and named_by='ai'."""
        out = []
        for d in dets:
            t = self.tracks.get(d.track_id) if d.track_id is not None else None
            if t is not None and t.name is not None:
                cls_id, label, detail = t.name
                d = Detection(cls_id, label, d.conf, d.xyxy, d.track_id, note=detail, named_by="ai")
            out.append(d)
        return out

    # ------------------------------------------------------------ worker
    def _run(self) -> None:
        warm = getattr(self.client, "warm", None)
        if warm:
            try:
                warm()
            except Exception as e:  # noqa: BLE001
                log.info("air namer: warm-up failed (%s)", e)
        while not self._stop.is_set():
            with self._cv:
                while not self.queue and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._stop.is_set():
                    return
                tid, jpeg, queued_at = self.queue.popleft()
                self.in_flight = 1
            t0 = time.time()
            try:
                answer = self.client(jpeg)
                ok = True
            except Exception as e:  # noqa: BLE001 - unreachable / timeout / bad JSON: keep the YOLOE vote
                ok, answer = False, None
                self.failures += 1
                self.last_error = str(e)[:160]
                if _is_unreachable(e):
                    self.unreachable_until = time.time() + self.unreachable_backoff
                log.warning("air namer: track %s not named (%s: %s)", tid % air.TRACK_ID_OFFSET, type(e).__name__, e)
            self.calls += 1
            self.latencies.append(time.time() - t0)
            with self._cv:
                self.in_flight = 0
                t = self.tracks.get(tid)
                if t is None:
                    continue
                t.pending = False
                if not ok:
                    continue
                self.unreachable_until, self.last_error = 0.0, ""
                new = to_label(answer, self.accept_conf)
                old_label = self.counter.voted(tid)[1] if tid in getattr(self.counter, "_votes", {}) else None
                t.answer, t.history = answer, [*t.history, answer]
                if t.name is None or t.name[1] == CONTACT[1]:
                    t.name, t.named_at = new, time.time()
                    self.counter.set_override(tid, new[0], new[1])
            if self.on_named and new[1] != old_label:
                try:
                    self.on_named(tid, old_label, new[1], new[2])
                except Exception as e:  # noqa: BLE001
                    log.warning("air namer callback failed: %s", e)

    # ------------------------------------------------------------ status
    @property
    def reachable(self) -> bool:
        return time.time() >= self.unreachable_until

    def status(self) -> dict:
        lat = sorted(self.latencies)
        named = [t for t in self.tracks.values() if t.name is not None]
        return {
            "enabled": True, "reachable": self.reachable, "in_flight": self.in_flight, "queued": len(self.queue),
            "calls": self.calls, "failures": self.failures, "dropped": self.dropped, "named": len(named),
            "median_s": round(lat[len(lat) // 2], 2) if lat else None, "error": self.last_error,
            "model": getattr(self.client, "model", ""),
        }
