"""An unplugged capture card must never silently become another camera (HQ Mac bug)."""

import time

import numpy as np
import pytest

from fpv_vision import sources
from conftest import wait_for

MAC = {"index": 0, "name": "MacBook Pro Camera", "id": "0x1000"}
CARD = {"index": 0, "name": "Guermok USB3 Video", "id": "0x0100"}


class FakeCap:
    opened: list = []

    def __init__(self, dev, backend=None):
        FakeCap.opened.append(dev)
        self.dev = dev

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        time.sleep(0.01)
        return True, np.zeros((48, 64, 3), np.uint8)

    def get(self, k):
        return 0

    def set(self, *a):
        return True

    def release(self):
        pass


@pytest.fixture
def cams(monkeypatch):
    state = {"fresh": [MAC], "mine": [MAC]}
    monkeypatch.setattr(sources.sys, "platform", "darwin")
    monkeypatch.setattr(sources, "_avf_fresh", lambda timeout=10.0: list(state["fresh"]))
    monkeypatch.setattr(sources, "_avf_in_process", lambda: list(state["mine"]))
    monkeypatch.setattr(sources.cv2, "VideoCapture", FakeCap)
    FakeCap.opened = []
    return state


def test_name_never_falls_back_to_an_index(cams):
    assert sources.resolve_camera("USB3 Video") is None  # card unplugged: no match, no index
    assert sources.resolve_camera("0") == 0  # a bare number is an explicit index
    cams["fresh"] = [CARD, {**MAC, "index": 1}]
    assert sources.resolve_camera("usb3 video") == 0


def test_unplugged_card_opens_nothing(cams):
    # the stale in-process list still shows the card at index 0, which OpenCV now maps to the webcam
    cams["mine"] = [CARD, {**MAC, "index": 1}]
    src = sources.VideoSource("camera:USB3 Video").start()
    try:
        assert wait_for(lambda: src.error, timeout=3)
        assert FakeCap.opened == []
        assert src.error == "Capture card not connected" and not src.connected
        assert src.latest()[1] is None
    finally:
        src.stop()


def test_stale_process_list_is_refused(cams):
    cams["fresh"] = [CARD, {**MAC, "index": 1}]  # card back, fresh list
    cams["mine"] = [MAC]  # but this process still thinks index 0 is the webcam
    src = sources.VideoSource("camera:USB3 Video").start()
    try:
        assert wait_for(lambda: src.error, timeout=3)
        assert "out of date" in src.error and not src.connected and src.latest()[1] is None
    finally:
        src.stop()


def test_matching_card_opens_and_unplug_clears_frames(cams):
    cams["fresh"] = cams["mine"] = [CARD, {**MAC, "index": 1}]
    src = sources.VideoSource("camera:USB3 Video").start()
    try:
        assert wait_for(lambda: src.connected and src.latest()[1] is not None, timeout=3)
        assert FakeCap.opened == [0]
        src._clear_frame()
        assert src.latest()[1] is None
    finally:
        src.stop()
