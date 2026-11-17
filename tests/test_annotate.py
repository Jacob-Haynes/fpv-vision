import numpy as np

from fpv_vision import annotate
from fpv_vision.detector import Detection


def test_overlay_uses_kit_palette_only():
    img = np.zeros((720, 1280, 3), np.uint8)
    dets = [Detection(0, "person", 0.9, (100, 100, 300, 400), 7), Detection(2, "car", 0.8, (500, 300, 900, 500), 8)]
    out = annotate.draw(img, dets, ["hud"], badge="SIMULATED")
    assert tuple(out[250, 100]) == annotate.UNKNOWN  # box edge: APP-6 unknown yellow
    assert annotate.UNKNOWN == (128, 255, 255) and annotate.HOSTILE == (128, 128, 255)
    b, g, r = (out[..., i].astype(int) for i in range(3))
    assert not ((r > 200) & (g < 90) & (b < 90)).any()  # no saturated red
    assert not ((g > 150) & (r < 90) & (b < 90)).any()  # no saturated green
    hot = annotate.draw(np.zeros_like(img), dets, alert_tracks={7})
    assert tuple(hot[250, 100]) == annotate.HOSTILE and tuple(hot[400, 500]) == annotate.UNKNOWN
