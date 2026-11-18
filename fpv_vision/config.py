"""Runtime settings, read from the environment (``/etc/fpv-vision.env`` on the Pi).

CLI flags in ``__main__`` override these.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from fpv_vision.stable import DEFAULT_CLASS_CONF, DEFAULT_SEARCH_CLASS_CONF, parse_class_conf

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "")
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _default_data_dir() -> Path:
    # /var/lib/fpv-vision on the Pi; a local folder when running from a checkout.
    sys_dir = Path("/var/lib/fpv-vision")
    if sys_dir.is_dir() and os.access(sys_dir, os.W_OK):
        return sys_dir
    return Path.cwd() / "data"


@dataclass
class Settings:
    # --- video ---
    source: str = "simulate"  # simulate | uvc:0 | uvc:/dev/video0 | rtsp://... | file:/path.mp4 | gopro-usb
    capture_width: int = 1920  # requested from cameras/capture cards; the device may negotiate lower
    capture_height: int = 1080
    capture_fps: int = 60
    sample_url: str = ""

    # --- detection ---
    model: str = "auto"  # auto = gpu_model on CUDA/MPS, cpu_model on CPU; or a name/path
    gpu_model: str = "yolo26s"
    cpu_model: str = "yolo26n"
    imgsz: int = 0  # 0 = auto: gpu_imgsz / cpu_imgsz
    gpu_imgsz: int = 640
    cpu_imgsz: int = 416
    infer_max_width: int = 1280  # downscale bigger frames (4K/2.7K) before inference
    conf: float = 0.30
    device: str = "auto"  # auto = CUDA > Apple MPS > CPU
    all_classes: bool = False  # False = ISR-relevant subset (people, vehicles, boats)
    target_fps: float = 20.0  # adaptive frame-skip aims to keep the overlay at least this smooth
    detector: str = "yolo"  # yolo | none (stream only)

    # --- web ---
    host: str = "0.0.0.0"
    port: int = 8110
    stream_width: int = 960
    jpeg_quality: int = 75
    air: bool = False  # Air watch (FPV_AIR=1 only, never in the UI): air detection belongs to OP sentry
    air_fps: float = 10.0  # max Air watch passes per second (each pass = full frame + 2x2 tiles)
    air_name_conf: float = 0.7  # Air watch confidence a track needs before the vision model is asked to name it
    search: bool = False  # Search: ground detector on tiles for small, distant people/vehicles (FPV_SEARCH)
    search_conf: float = 0.25
    search_grid: str = "auto"  # auto = 3x3 for 1440p and up, else 2x2; or 1/2/3
    search_hits: int = 2  # consecutive passes before a Search box is shown
    search_air_fps: float = 5.0  # Air watch rate cap while Search is on (they share the GPU)
    # stable detections (fpv_vision.stable): false-vehicle filters and flicker-free boxes
    stable: bool = True  # FPV_STABLE=0: the tracker's raw output, as before
    fast_track: bool = True  # FPV_FAST_TRACK=0: Ultralytics' own ByteTrack start-up (misses fast small objects)
    class_conf: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_CONF))  # FPV_CLASS_CONF=truck=0.4,bus=0.4
    search_class_conf: dict = field(default_factory=lambda: dict(DEFAULT_SEARCH_CLASS_CONF))  # FPV_SEARCH_CLASS_CONF
    vehicle_max_area: float = 0.25  # a vehicle box bigger than this share of the frame is a building
    vehicle_max_aspect: float = 4.0  # ... or longer than this many times its width, a fence or wall
    confirm_hits: int = 3  # consecutive detections before a truck, bus or train is shown or counted
    confirm_hits_other: int = 2  # every other class (people, bikes, cars, boats)
    confirm_fast_conf: float = 0.6  # a person, bike, motorcycle or car this confident is shown at once (0 = off)
    coast_s: float = 0.5  # a confirmed box stays, faded, this long after its last detection
    label_switch: float = 1.5  # a track's label changes only when another class's vote is this many times bigger
    handover_iou: float = 0.3  # a new tracker id this far over a held box takes that track over (0 = never)
    heavy_max_car_ratio: float = 10.0  # a truck/bus/train box this many times the recent cars' size is a building (0 = off)
    show_hud: bool = False  # debug text (model, fps, time) burned into the video; the provenance badge is always drawn

    # --- replay / footage library ---
    footage_dirs: str = "~/Movies/FPV"
    # --- demo style (ui-kit DEMO-TRIGGER-CONTRACT): continuous | on_demand (step by step)
    demo_mode: str = "continuous"  # os.pathsep-separated dirs listed in the UI's footage picker
    allow_screen: bool = sys.platform == "darwin"  # offer screen capture (phone mirroring) in the UI
    request_camera: bool = True  # macOS: ask for camera permission at startup (main thread)

    # --- storage ---
    data_dir: Path = field(default_factory=_default_data_dir)
    max_clip_seconds: int = 300

    # --- Gemini (optional cloud mode) ---
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_min_interval: float = 10.0  # seconds between any two calls
    gemini_periodic_seconds: float = 0.0  # 0 = off; else auto-summary period (>= min interval)

    # --- scene description backend (VLM) ---
    vlm_backend: str = "auto"  # auto | openai_compat | gemini | off
    vlm_url: str = ""  # bare base URL, e.g. http://isr-hq.local:11434 (Ollama)
    vlm_model: str = "qwen3-vl:8b-instruct"
    vlm_api_key: str = ""
    vlm_min_interval: float = 3.0

    # --- ATAK / mesh ---
    tak_host: str = "127.0.0.1"
    tak_port: int = 8087
    tak_enabled: bool = True
    callsign: str = "FPV-VISION"
    default_position: str = ""  # "lat,lon" or MGRS grid for spot reports
    sensor_lat: float | None = None  # shared isr-kit SENSOR_LAT/SENSOR_LON (fallback position)
    sensor_lon: float | None = None
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_user: str = "guest"
    mqtt_pass: str = "guest"
    mesh_enabled: bool = True
    mqtt_enabled: bool = True
    public_url: str = ""  # full UI URL for the heartbeat (default http://<host>.local:<port>/)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for d in (self.snapshots_dir, self.clips_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.source = _env("FPV_SOURCE", s.source)
        s.capture_width = _env_int("FPV_CAPTURE_WIDTH", s.capture_width)
        s.capture_height = _env_int("FPV_CAPTURE_HEIGHT", s.capture_height)
        s.capture_fps = _env_int("FPV_CAPTURE_FPS", s.capture_fps)
        s.sample_url = _env("FPV_SAMPLE_URL", s.sample_url)
        s.model = _env("FPV_MODEL", s.model)
        s.gpu_model = _env("FPV_GPU_MODEL", s.gpu_model)
        s.cpu_model = _env("FPV_CPU_MODEL", s.cpu_model)
        s.imgsz = _env_int("FPV_IMGSZ", s.imgsz)
        s.gpu_imgsz = _env_int("FPV_GPU_IMGSZ", s.gpu_imgsz)
        s.cpu_imgsz = _env_int("FPV_CPU_IMGSZ", s.cpu_imgsz)
        s.infer_max_width = _env_int("FPV_INFER_MAX_WIDTH", s.infer_max_width)
        s.device = _env("FPV_DEVICE", s.device)
        s.conf = _env_float("FPV_CONF", s.conf)
        s.all_classes = _env_bool("FPV_ALL_CLASSES", s.all_classes)
        s.target_fps = _env_float("FPV_TARGET_FPS", s.target_fps)
        s.detector = _env("FPV_DETECTOR", s.detector)
        s.host = _env("FPV_HOST", s.host)
        s.port = _env_int("FPV_PORT", s.port)
        s.stream_width = _env_int("FPV_STREAM_WIDTH", s.stream_width)
        s.jpeg_quality = _env_int("FPV_JPEG_QUALITY", s.jpeg_quality)
        s.show_hud = _env_bool("FPV_HUD", s.show_hud)
        s.air = _env_bool("FPV_AIR", s.air)
        s.air_fps = _env_float("FPV_AIR_FPS", s.air_fps)
        s.air_name_conf = _env_float("FPV_AIR_NAME_CONF", s.air_name_conf)
        s.search = _env_bool("FPV_SEARCH", s.search)
        s.search_conf = _env_float("FPV_SEARCH_CONF", s.search_conf)
        s.search_grid = os.environ.get("FPV_SEARCH_GRID", s.search_grid).strip() or "auto"
        s.search_hits = _env_int("FPV_SEARCH_HITS", s.search_hits)
        s.search_air_fps = _env_float("FPV_SEARCH_AIR_FPS", s.search_air_fps)
        s.stable = _env_bool("FPV_STABLE", s.stable)
        s.fast_track = _env_bool("FPV_FAST_TRACK", s.fast_track)
        s.class_conf = parse_class_conf(_env("FPV_CLASS_CONF"), s.class_conf)
        s.search_class_conf = parse_class_conf(_env("FPV_SEARCH_CLASS_CONF"), s.search_class_conf)
        s.vehicle_max_area = _env_float("FPV_VEHICLE_MAX_AREA", s.vehicle_max_area)
        s.vehicle_max_aspect = _env_float("FPV_VEHICLE_MAX_ASPECT", s.vehicle_max_aspect)
        s.confirm_hits = _env_int("FPV_CONFIRM_HITS", s.confirm_hits)
        s.confirm_hits_other = _env_int("FPV_CONFIRM_HITS_OTHER", s.confirm_hits_other)
        s.confirm_fast_conf = _env_float("FPV_CONFIRM_FAST_CONF", s.confirm_fast_conf)
        s.coast_s = _env_float("FPV_COAST_S", s.coast_s)
        s.label_switch = _env_float("FPV_LABEL_SWITCH", s.label_switch)
        s.handover_iou = _env_float("FPV_HANDOVER_IOU", s.handover_iou)
        s.heavy_max_car_ratio = _env_float("FPV_HEAVY_MAX_CAR_RATIO", s.heavy_max_car_ratio)
        if _env("FPV_DATA_DIR"):
            s.data_dir = Path(_env("FPV_DATA_DIR"))
        s.footage_dirs = _env("FPV_FOOTAGE_DIRS", s.footage_dirs)
        s.demo_mode = "on_demand" if _env("DEMO_MODE", s.demo_mode).strip().lower() in ("on_demand", "on-demand", "step") \
            else "continuous"
        s.allow_screen = _env_bool("FPV_ALLOW_SCREEN", s.allow_screen)
        s.request_camera = _env_bool("FPV_REQUEST_CAMERA", s.request_camera)
        s.max_clip_seconds = _env_int("FPV_MAX_CLIP_SECONDS", s.max_clip_seconds)
        s.gemini_api_key = _env("GEMINI_API_KEY")
        s.gemini_model = _env("GEMINI_MODEL", s.gemini_model) or DEFAULT_GEMINI_MODEL
        s.gemini_min_interval = _env_float("GEMINI_MIN_INTERVAL", s.gemini_min_interval)
        s.gemini_periodic_seconds = _env_float("GEMINI_PERIODIC_SECONDS", s.gemini_periodic_seconds)
        s.tak_host = _env("TAK_HOST", s.tak_host)
        s.tak_port = _env_int("TAK_PORT", s.tak_port)
        s.tak_enabled = _env_bool("TAK_ENABLED", s.tak_enabled)
        s.callsign = _env("FPV_CALLSIGN", s.callsign)
        s.default_position = _env("FPV_DEFAULT_POSITION", s.default_position)
        try:
            s.sensor_lat = float(_env("SENSOR_LAT")) if _env("SENSOR_LAT") else None
            s.sensor_lon = float(_env("SENSOR_LON")) if _env("SENSOR_LON") else None
        except ValueError:
            s.sensor_lat = s.sensor_lon = None
        s.mqtt_host = _env("MQTT_HOST", s.mqtt_host)
        s.mqtt_port = _env_int("MQTT_PORT", s.mqtt_port)
        s.mqtt_user = _env("MQTT_USER", s.mqtt_user)
        s.mqtt_pass = _env("MQTT_PASS", s.mqtt_pass)
        s.mesh_enabled = _env_bool("MESH_ENABLED", s.mesh_enabled)
        s.mqtt_enabled = _env_bool("MQTT_ENABLED", s.mqtt_enabled)
        s.public_url = _env("FPV_PUBLIC_URL", s.public_url)
        s.vlm_backend = _env("VLM_BACKEND", s.vlm_backend)
        s.vlm_url = _env("VLM_URL", s.vlm_url)
        s.vlm_model = _env("VLM_MODEL", s.vlm_model) or "qwen3-vl:8b-instruct"
        s.vlm_api_key = _env("VLM_API_KEY", s.vlm_api_key)
        s.vlm_min_interval = _env_float("VLM_MIN_INTERVAL", s.vlm_min_interval)
        return s
