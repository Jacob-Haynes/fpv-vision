"""Sample-clip prefetch (offline demo day) and the FOURCC log fix."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from fpv_vision import sample
from fpv_vision.sources import VideoSource


@pytest.fixture
def clip_bytes(tmp_path):
    p = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    return p.read_bytes()


@pytest.fixture
def server(clip_bytes):
    hits, agents = [], []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            agents.append(self.headers.get("User-Agent", ""))
            if self.path == "/busy" and hits.count("/busy") == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            if self.path in ("/busy", "/mirror"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(clip_bytes)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", hits, agents
    srv.shutdown()


@pytest.fixture(autouse=True)
def no_repo_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "REPO_SAMPLES", tmp_path / "no-repo-samples")


def test_retries_after_429_with_a_descriptive_user_agent(tmp_path, server):
    base, hits, agents = server
    p = sample.ensure_sample(tmp_path / "d", "street", url=base + "/busy", attempts=3, backoff=0.01)
    assert p and p.exists() and hits == ["/busy", "/busy"]
    assert "github.com/Jacob-Haynes/fpv-vision" in agents[0]
    # cached copy is reused: no further requests
    assert sample.ensure_sample(tmp_path / "d", "street", url=base + "/busy") == p and len(hits) == 2


def test_falls_back_to_mirror_then_gives_up_cleanly(tmp_path, server, monkeypatch):
    base, hits, _ = server
    monkeypatch.setitem(sample.SAMPLES["street"], "url", base + "/gone")
    monkeypatch.setitem(sample.SAMPLES["street"], "mirrors", [base + "/mirror"])
    assert sample.ensure_sample(tmp_path / "d", "street", attempts=2, backoff=0.01)
    assert hits == ["/gone", "/mirror"]  # 404 is not retried
    monkeypatch.setitem(sample.SAMPLES["aerial"], "url", base + "/gone")
    assert sample.ensure_sample(tmp_path / "d", "aerial", attempts=2, backoff=0.01) is None
    assert not list((tmp_path / "d" / "samples").glob("*.part"))


class _Cap:
    def __init__(self, fourcc):
        self.v = {cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080, cv2.CAP_PROP_FPS: 60,
                  cv2.CAP_PROP_FOURCC: fourcc}

    def get(self, k):
        return self.v[k]


def test_fourcc_only_shown_when_valid():
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    assert VideoSource._describe_mode(_Cap(mjpg)) == "1920x1080@60 MJPG"
    assert VideoSource._describe_mode(_Cap(-1)) == "1920x1080@60"  # was "ÿÿÿÿ"
    assert VideoSource._describe_mode(_Cap(0)) == "1920x1080@60"
    assert VideoSource._describe_mode(_Cap(0x01020304)) == "1920x1080@60"
