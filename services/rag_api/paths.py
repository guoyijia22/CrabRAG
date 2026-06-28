from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root(default_root: Path) -> Path:
    value = os.getenv("CRABRAG_ROOT") or os.getenv("ELCQA_ROOT")
    return Path(value or default_root).resolve()


def resolve_env_file(project_root: Path, legacy_env_path: Path) -> Path:
    explicit = os.getenv("CRABRAG_ENV_FILE") or os.getenv("ELCQA_ENV_FILE")
    if explicit:
        return Path(explicit).resolve()
    if os.getenv("CRABRAG_ROOT") or os.getenv("ELCQA_ROOT"):
        return (project_root / "config" / ".env").resolve()
    return legacy_env_path.resolve()


def docs_dirs_env_value() -> str:
    return os.getenv("CRABRAG_DOCS_DIR") or os.getenv("ELCQA_DOCS_DIR") or ""
