"""Public-domain / CC0 sample clips for ``--simulate``, downloaded once into the data dir.

Both are Wikimedia Commons 480p VP9 transcodes (~9 MB). Commons occasionally regenerates
transcodes, so we validate by decoding rather than by checksum.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLES = {
    # NASA drone pass over the Banana Creek stands at the Artemis II launch: buses, cars, crowds.
    # Public domain (PD-USGov-NASA). Credit (courtesy): NASA Kennedy Space Center.
    # The last third is heavily backlit, so we loop the first 35 s.
    "aerial": {
        "file": "sample_aerial_nasa_480p.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/4/4c/"
               "Artemis_II_Live_Launch_Coverage_-_Drone%2C_Crowd_Shots_of_Banana_Creek_%28KSC-20260401-MH-AJN01-0001-"
               "Artemis_II_Launch_ISO_Drone_Crowd_Shot_Banana_Creek_4K%29.webm/"
               "Artemis_II_Live_Launch_Coverage_-_Drone%2C_Crowd_Shots_of_Banana_Creek_%28KSC-20260401-MH-AJN01-0001-"
               "Artemis_II_Launch_ISO_Drone_Crowd_Shot_Banana_Creek_4K%29.webm.480p.vp9.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Artemis_II_Live_Launch_Coverage_-_Drone,_Crowd_Shots_of_"
                "Banana_Creek_(KSC-20260401-MH-AJN01-0001-Artemis_II_Launch_ISO_Drone_Crowd_Shot_Banana_Creek_4K).webm",
        "licence": "Public domain (NASA)",
        "loop_seconds": 35,
    },
    # Handheld, eye-level: Yonge & Dundas scramble crossing, Toronto. Lots of people, cars, bikes.
    # CC0 by Raysonho @ Open Grid Scheduler / Grid Engine.
    "street": {
        "file": "sample_street_cc0_480p.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d2/DiagonalCrosswalkYongeDundas.webm/"
               "DiagonalCrosswalkYongeDundas.webm.480p.vp9.webm",
        "page": "https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm",
        "licence": "CC0 1.0",
        "loop_seconds": 0,
    },
}
DEFAULT_SAMPLE = "street"
USER_AGENT = "fpv-vision-sample-fetch/1.0 (training demo)"


def sample_info(name: str | None = None) -> dict:
    return SAMPLES.get(name or DEFAULT_SAMPLE, SAMPLES[DEFAULT_SAMPLE])


def _decodes(path: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def ensure_sample(data_dir: Path, name: str | None = None, url: str = "", timeout: float = 60.0) -> Path | None:
    """Return a local path to the sample clip, downloading it if needed. None if unavailable (offline)."""
    info = sample_info(name)
    dest = data_dir / "samples" / info["file"]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    src = url or info["url"]
    try:
        log.info("downloading sample clip (%s) ...", info["licence"])
        req = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
        if not _decodes(tmp):
            raise RuntimeError("downloaded sample does not decode")
        os.replace(tmp, dest)
        log.info("sample clip saved to %s", dest)
        return dest
    except Exception as e:  # noqa: BLE001 - offline is a normal state for the demo
        log.warning("sample clip unavailable (%s); simulate will use a synthetic test pattern", e)
        tmp.unlink(missing_ok=True)
        return None


if __name__ == "__main__":  # python -m fpv_vision.sample [data_dir]
    import sys

    logging.basicConfig(level=logging.INFO)
    from fpv_vision.config import Settings

    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Settings.from_env().data_dir
    for key in SAMPLES:
        print(key, ensure_sample(d, key))
