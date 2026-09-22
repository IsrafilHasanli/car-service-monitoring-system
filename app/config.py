from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    rtsp_url: str = field(default_factory=lambda: os.getenv("RTSP_URL", "").strip())
    json_file: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_JSON_FILE", "data/queue_garage.json"))
    roi_file: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_ROI_FILE", "config/roi.json"))
    entered_plates_file: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_TEXT_FILE", "data/entered_plates.txt"))
    image_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_IMAGE_DIR", "data/images"))
    live_frame_file: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_LIVE_FRAME_FILE", "data/live_frame.jpg"))
    inference_log: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_INFERENCE_LOG", "data/anpr_inference.log"))

    vehicle_model_path: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_VEHICLE_MODEL", "models/yolov8n.pt"))
    plate_model_path: Path = field(default_factory=lambda: BASE_DIR / os.getenv("ANPR_PLATE_MODEL", "models/license_plate_detector.pt"))
    ocr_model_name: str = field(default_factory=lambda: os.getenv("ANPR_OCR_MODEL", "global-plates-mobile-vit-v2-model"))

    target_width: int = field(default_factory=lambda: _int("ANPR_TARGET_WIDTH", 1280))
    frame_skip: int = field(default_factory=lambda: max(1, _int("ANPR_FRAME_SKIP", 2)))
    reconnect_delay_sec: float = field(default_factory=lambda: _float("ANPR_RECONNECT_DELAY_SEC", 2.0))
    live_frame_interval_sec: float = field(default_factory=lambda: _float("ANPR_LIVE_FRAME_INTERVAL", 0.08))

    vehicle_confidence: float = field(default_factory=lambda: _float("ANPR_VEHICLE_CONFIDENCE", 0.5))
    plate_confidence: float = field(default_factory=lambda: _float("ANPR_PLATE_CONFIDENCE", 0.2))
    vehicle_img_size: int = field(default_factory=lambda: _int("ANPR_VEHICLE_IMGSZ", 640))
    plate_img_size: int = field(default_factory=lambda: _int("ANPR_PLATE_IMGSZ", 320))
    ocr_every_n_frames: int = field(default_factory=lambda: max(1, _int("ANPR_OCR_EVERY_N_FRAMES", 2)))
    min_vehicle_width: int = field(default_factory=lambda: _int("ANPR_MIN_VEHICLE_WIDTH", 180))
    min_vehicle_height: int = field(default_factory=lambda: _int("ANPR_MIN_VEHICLE_HEIGHT", 90))
    min_vehicle_area: int = field(default_factory=lambda: _int("ANPR_MIN_VEHICLE_AREA", 25000))
    vehicle_edge_margin: int = field(default_factory=lambda: _int("ANPR_VEHICLE_EDGE_MARGIN", 12))

    top_entry_ratio: float = field(default_factory=lambda: _float("ANPR_TOP_ENTRY_RATIO", 0.18))
    right_margin_px: int = field(default_factory=lambda: _int("ANPR_RIGHT_MARGIN_PX", 8))
    show_window: bool = field(default_factory=lambda: _bool("ANPR_SHOW_WINDOW", False))
    rtsp_transport: str = field(default_factory=lambda: os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"))


def load_settings() -> Settings:
    load_dotenv(ENV_FILE, override=True)
    loaded = Settings()
    loaded.image_dir.mkdir(parents=True, exist_ok=True)
    loaded.json_file.parent.mkdir(parents=True, exist_ok=True)
    loaded.entered_plates_file.parent.mkdir(parents=True, exist_ok=True)
    loaded.live_frame_file.parent.mkdir(parents=True, exist_ok=True)
    loaded.inference_log.parent.mkdir(parents=True, exist_ok=True)
    return loaded


settings = load_settings()
