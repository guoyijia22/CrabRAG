from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_import(module_name: str, errors: list[str]) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - message is for humans.
        errors.append(f"Python package import failed: {module_name} ({exc})")


def check_file(relative_path: str, errors: list[str]) -> None:
    path = ROOT / relative_path
    if not path.is_file():
        errors.append(f"Missing required file: {relative_path}")


def check_dir(relative_path: str, errors: list[str]) -> None:
    path = ROOT / relative_path
    if not path.is_dir():
        errors.append(f"Missing required directory: {relative_path}")


def find_bun() -> str | None:
    portable_candidates = [
        ROOT / "runtime" / "bun" / "bun.exe",
        ROOT / "runtime" / "bun" / "bun",
    ]
    for candidate in portable_candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("bun")


def main() -> int:
    errors: list[str] = []
    optional_warnings: list[str] = []

    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ is required.")

    for module_name in [
        "fastapi",
        "uvicorn",
        "dotenv",
        "pydantic",
        "chromadb",
        "requests",
        "numpy",
    ]:
        check_import(module_name, errors)

    try:
        importlib.import_module("onnxruntime")
        optional_warnings.append("Local ONNX runtime available.")
    except Exception as exc:  # pragma: no cover - message is for humans.
        optional_warnings.append(f"Local ONNX runtime unavailable: {exc}")

    for relative_path in [
        "config/.env",
        "requirements.txt",
        "server/gateway.js",
        "apps/web/dist/index.html",
    ]:
        check_file(relative_path, errors)

    for relative_path in ["docs", "data", "logs"]:
        check_dir(relative_path, errors)

    if find_bun() is None:
        errors.append("Bun was not found on PATH or under runtime/bun.")

    if (ROOT / "package.json").is_file() and not (ROOT / "node_modules").is_dir():
        errors.append("node_modules is missing. Run bun install.")

    if errors:
        print("[CrabRAG] Environment check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    for warning in optional_warnings:
        print(f"[CrabRAG] Optional: {warning}")
    print("[CrabRAG] Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
