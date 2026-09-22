from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.config import load_settings

_process: subprocess.Popen | None = None


def _worker_python(base_dir: Path) -> str:
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    venv_python = base_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def is_running() -> bool:
    return _process is not None and _process.poll() is None


def status() -> dict[str, Any]:
    current = load_settings()
    missing_rtsp = not bool(current.rtsp_url)
    return {
        "running": is_running(),
        "pid": _process.pid if is_running() else None,
        "live_frame": current.live_frame_file.exists(),
        "log_file": str(current.inference_log),
        "source_type": "rtsp",
        "configured": not missing_rtsp,
        "message": "RTSP_URL is empty in .env" if missing_rtsp else "Ready",
    }


def start() -> dict[str, Any]:
    global _process

    if is_running():
        return status()
    current = load_settings()
    if not current.rtsp_url:
        return status()

    env = os.environ.copy()
    env["ANPR_SHOW_WINDOW"] = "0"

    log = current.inference_log.open("a", encoding="utf-8")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _process = subprocess.Popen(
        [_worker_python(current.base_dir), "-m", "app.worker"],
        cwd=str(current.base_dir),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    return status()


def stop() -> dict[str, Any]:
    global _process

    if is_running():
        _process.terminate()
        try:
            _process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
    return status()
