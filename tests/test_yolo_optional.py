"""Real-model smoke test. Runs only when weights/sample are present (e.g. after
`python -m fpv_vision.sample data && python -m fpv_vision.export --data-dir data`)."""

from pathlib import Path

import cv2
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE = DATA / "samples" / "sample_street_cc0_480p.webm"
MODELS = DATA / "models"


@pytest.mark.skipif(not SAMPLE.exists() or not any(MODELS.glob("yolo26n*")), reason="no local weights/sample")
def test_yolo_detects_people_on_street_sample():
    pytest.importorskip("ultralytics")
    from fpv_vision.detector import ISR_CLASSES, YoloDetector

    det = YoloDetector("yolo26n", MODELS, classes=ISR_CLASSES)
    cap = cv2.VideoCapture(str(SAMPLE))
    labels = set()
    tracked = False
    for _ in range(10):
        ok, frame = cap.read()
        assert ok
        dets = det.detect(frame)
        labels |= {d.label for d in dets}
        tracked |= any(d.track_id is not None for d in dets)
    assert "person" in labels and "car" in labels
    assert tracked
