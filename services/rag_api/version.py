from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATH = PROJECT_ROOT / "VERSION"


def read_software_version(path: Path = VERSION_PATH) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"
    return value or "0.0.0"


SOFTWARE_VERSION = read_software_version()


def build_info() -> dict[str, str]:
    return {
        "version": SOFTWARE_VERSION,
        "commit": os.getenv("CRABRAG_BUILD_COMMIT", "unknown"),
        "built_at": os.getenv("CRABRAG_BUILD_TIME", "unknown"),
    }


def onnxruntime_capability() -> dict[str, object]:
    """Probe ONNX in a child process because a broken DLL can terminate the importer."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import onnxruntime"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {
        "available": result.returncode == 0,
        "error_type": None if result.returncode == 0 else "OnnxRuntimeInitializationError",
    }
