from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from services.rag_api.paths import resolve_env_file, resolve_project_root


ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = resolve_project_root(ROOT_DIR)
ENV_PATH = resolve_env_file(PROJECT_DIR, Path(r"D:\cd\.env"))


def load_runtime_environment() -> None:
    load_dotenv(ENV_PATH, override=False)
