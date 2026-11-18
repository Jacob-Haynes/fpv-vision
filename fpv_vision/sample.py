"""Public-domain / CC0 sample clips for ``--simulate``, downloaded once into the data dir.

Both are Wikimedia Commons 480p VP9 transcodes (~9 MB). Commons occasionally regenerates
transcodes, so we validate by decoding rather than by checksum.

Demo day is offline, so fetch them beforehand while online:

    scripts/run-laptop.sh --setup-only          # venv + sample clips + model weights
    python -m fpv_vision.sample [data_dir]       # sample clips only

Downloads retry with backoff (honouring Retry-After on HTTP 429) and send a descriptive
User-Agent as Wikimedia's policy asks; each clip also has fallback URLs. To copy a clip in by
hand, put the file under ``<data_dir>/samples/`` with the name in ``SAMPLES[...]["file"]``.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
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
        "mirrors": [],  # the original is a multi-GB 4K file: copy the clip in by hand instead
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
        "mirrors": [  # a 360p transcode, then the original upload (~73 MB, full size) via two entry points
            "https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d2/DiagonalCrosswalkYongeDundas.webm/"
            "DiagonalCrosswalkYongeDundas.webm.360p.vp9.webm",
            "https://upload.wikimedia.org/wikipedia/commons/d/d2/DiagonalCrosswalkYongeDundas.webm",
            "https://commons.wikimedia.org/wiki/Special:FilePath/DiagonalCrosswalkYongeDundas.webm",
        ],
        "licence": "CC0 1.0",
        "loop_seconds": 0,
    },
}
DEFAULT_SAMPLE = "street"
# Wikimedia's User-Agent policy: name/version plus a way to reach the operator (the public repo).
USER_AGENT = "fpv-vision/0.1 (https://github.com/Jacob-Haynes/fpv-vision; sample clip prefetch) python-urllib"
REPO_SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"  # a checkout's ./data/samples


def sample_info(name: str | None = None) -> dict:
    return SAMPLES.get(name or DEFAULT_SAMPLE, SAMPLES[DEFAULT_SAMPLE])


def _decodes(path: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def cached_sample(data_dir: Path, name: str | None = None) -> Path | None:
    """An existing copy of the clip: <data_dir>/samples first, then the checkout's ./data/samples."""
    info = sample_info(name)
    for d in (Path(data_dir) / "samples", REPO_SAMPLES):
        p = d / info["file"]
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _fetch(src: str, tmp: Path, timeout: float) -> None:
    req = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)


def ensure_sample(data_dir: Path, name: str | None = None, url: str = "", timeout: float = 60.0,
                  attempts: int = 1, backoff: float = 2.0) -> Path | None:
    """Return a local path to the sample clip, downloading it if needed. None if unavailable (offline).

    Reuses a cached copy. Otherwise tries ``url`` (FPV_SAMPLE_URL) or the clip's URL, then its
    mirrors, each up to ``attempts`` times with exponential backoff (Retry-After honoured, max 60 s).
    The service calls this with one attempt so startup never stalls; prefetch uses several.
    """
    info = sample_info(name)
    have = cached_sample(data_dir, name)
    if have:
        return have
    dest = Path(data_dir) / "samples" / info["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urls = [url] if url else [info["url"], *info.get("mirrors", [])]
    last = ""
    for src in urls:
        for attempt in range(attempts):
            try:
                log.info("downloading sample clip (%s) from %s ...", info["licence"], src.split("/")[2])
                _fetch(src, tmp, timeout)
                if not _decodes(tmp):
                    raise RuntimeError("downloaded file does not decode")
                os.replace(tmp, dest)
                log.info("sample clip saved to %s", dest)
                return dest
            except Exception as e:  # noqa: BLE001 - offline is a normal state for the demo
                tmp.unlink(missing_ok=True)
                last = str(e)
                wait = backoff * (2 ** attempt)
                if isinstance(e, urllib.error.HTTPError):
                    if e.code == 404:
                        break  # this URL is gone; try the next one
                    ra = e.headers.get("Retry-After") if e.headers else None
                    if ra and ra.isdigit():
                        wait = float(ra)
                if attempt + 1 < attempts:
                    wait = min(wait, 60.0)
                    log.info("sample download failed (%s); retrying in %.0f s", e, wait)
                    time.sleep(wait)
    log.warning("sample clip %s unavailable (%s); simulate will use a synthetic test pattern. Fetch it while "
                "online with scripts/run-laptop.sh --setup-only, or copy %s into %s", info["file"], last,
                info["file"], dest.parent)
    return None


def prefetch(data_dir: Path, attempts: int = 4) -> dict[str, Path | None]:
    """Fetch every sample clip (for --setup-only). Returns name -> path (None = still missing)."""
    return {key: ensure_sample(data_dir, key, attempts=attempts, timeout=120) for key in SAMPLES}


if __name__ == "__main__":  # python -m fpv_vision.sample [data_dir]
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from fpv_vision.config import Settings

    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Settings.from_env().data_dir
    missing = [k for k, p in prefetch(d).items() if p is None]
    for k in SAMPLES:
        print(f"{k}: {cached_sample(d, k) or 'MISSING'}")
    sys.exit(1 if missing else 0)
