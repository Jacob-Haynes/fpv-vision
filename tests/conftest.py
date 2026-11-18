import time

import pytest

from fpv_vision.config import Settings
from fpv_vision.detector import Detection, Detector


class FakeDetector(Detector):
    """Deterministic detections: 2 people next to a car, plus a truck, with stable track ids."""

    name = "fake"
    backend = "fake"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        return [
            Detection(0, "person", 0.9, (int(w * 0.10), int(h * 0.40), int(w * 0.14), int(h * 0.60)), 1),
            Detection(0, "person", 0.8, (int(w * 0.15), int(h * 0.40), int(w * 0.19), int(h * 0.60)), 2),
            Detection(2, "car", 0.85, (int(w * 0.18), int(h * 0.45), int(w * 0.30), int(h * 0.60)), 3),
            Detection(7, "truck", 0.7, (int(w * 0.75), int(h * 0.40), int(w * 0.90), int(h * 0.60)), 4),
        ]


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.source = "synthetic"
    s.tak_enabled = False
    s.mesh_enabled = False
    s.target_fps = 25
    s.ensure_dirs()
    return s


def wait_for(pred, timeout=8.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return False
