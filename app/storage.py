from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException

from app.config import settings


def load_records() -> list[dict[str, Any]]:
    path = settings.json_file
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"{path.name} is invalid: {e}") from e

    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail=f"{path.name} must contain a JSON list")

    return [x for x in data if isinstance(x, dict)]


def save_records(records: list[dict[str, Any]]) -> None:
    settings.json_file.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_id_index(records: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, rec in enumerate(records):
        if "id" in rec:
            try:
                out[int(rec["id"])] = i
            except (TypeError, ValueError):
                continue
    return out


def normalize_image_path(image_path: str) -> Path:
    p = Path(image_path.replace("\\", "/"))
    if p.is_absolute():
        full = p.resolve()
    else:
        if p.parts and p.parts[0].lower() == settings.image_dir.name.lower():
            p = Path(*p.parts[1:])
        full = (settings.image_dir / p).resolve()

    img_root = settings.image_dir.resolve()
    if full != img_root and img_root not in full.parents:
        raise HTTPException(status_code=400, detail="Invalid image path")

    return full


def resolve_image_path(item: dict[str, Any], kind: str) -> Path | None:
    if kind == "vehicle":
        candidates = [item.get("vehicle_image_path")]
    elif kind == "plate":
        candidates = [item.get("plate_image_path")]
        if not item.get("vehicle_image_path"):
            candidates.append(item.get("image_path"))
    else:
        candidates = [item.get("vehicle_image_path"), item.get("plate_image_path"), item.get("image_path")]

    seen: set[Path] = set()
    for image_path in candidates:
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        full = normalize_image_path(image_path)
        if full in seen:
            continue
        seen.add(full)
        if full.exists():
            return full
    return None


def parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S_%f"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def format_duration(entry_time: Any, exit_time: Any) -> str:
    start = parse_dt(entry_time)
    end = parse_dt(exit_time)
    if not start or not end:
        return ""

    seconds = max(0, int((end - start).total_seconds()))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def decorate_record(item: dict[str, Any]) -> dict[str, Any]:
    entry_time = item.get("entry_time") or item.get("timestamp") or ""
    exit_time = item.get("exit_time") or ""
    item_id = item.get("id")
    plate = item.get("platenumber") or ""
    vehicle_path = resolve_image_path(item, "vehicle")
    plate_path = resolve_image_path(item, "plate")
    vehicle_name = item.get("vehicle_image_path") if vehicle_path else ""
    plate_name = item.get("plate_image_path") or item.get("image_path") if plate_path else ""
    auto_name = vehicle_name or plate_name

    return {
        **item,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "platenumber": plate or "Reading...",
        "plate_is_pending": not bool(plate),
        "status": item.get("status") or "Unknown",
        "duration": format_duration(entry_time, exit_time),
        "vehicle_image_url": f"/image-file?name={quote(vehicle_name)}" if vehicle_name else None,
        "plate_image_url": f"/image-file?name={quote(plate_name)}" if plate_name else None,
        "image_url": f"/image-file?name={quote(auto_name)}" if auto_name else None,
    }


def queue_payload() -> list[dict[str, Any]]:
    items = [decorate_record(item) for item in load_records() if item.get("platenumber")]
    return sorted(items, key=lambda x: str(x.get("entry_time", "")), reverse=True)
