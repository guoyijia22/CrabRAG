from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATH = PROJECT_ROOT / "VERSION"


def read_software_version(path: Path = VERSION_PATH) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"
    return value or "0.0.0"


SOFTWARE_VERSION = read_software_version()
_ONNX_PROBE_CACHE: tuple[float, dict[str, object]] | None = None
_ONNX_PROBE_LOCK = threading.Lock()
_ONNX_PROBE_TTL_SECONDS = 300.0


def build_info() -> dict[str, str]:
    return {
        "version": SOFTWARE_VERSION,
        "commit": os.getenv("CRABRAG_BUILD_COMMIT", "unknown"),
        "built_at": os.getenv("CRABRAG_BUILD_TIME", "unknown"),
    }


def onnxruntime_capability() -> dict[str, object]:
    """Probe ONNX in a child process because a broken DLL can terminate the importer."""
    global _ONNX_PROBE_CACHE
    now = time.monotonic()
    cached = _ONNX_PROBE_CACHE
    if cached is not None and now - cached[0] < _ONNX_PROBE_TTL_SECONDS:
        return dict(cached[1])
    with _ONNX_PROBE_LOCK:
        cached = _ONNX_PROBE_CACHE
        if cached is not None and now - cached[0] < _ONNX_PROBE_TTL_SECONDS:
            return dict(cached[1])
        try:
            process = subprocess.run(
                [sys.executable, "-c", "import onnxruntime"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            result = {
                "available": process.returncode == 0,
                "error_type": None if process.returncode == 0 else "OnnxRuntimeInitializationError",
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = {"available": False, "error_type": type(exc).__name__}
        _ONNX_PROBE_CACHE = (now, result)
        return dict(result)
