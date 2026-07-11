from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR


INDEX_ROOT = PROJECT_DIR / "data" / "index"
ACTIVE_INDEX_PATH = INDEX_ROOT / "active.json"
GENERATIONS_DIR = INDEX_ROOT / "generations"


def new_generation_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gen-{timestamp}-{uuid.uuid4().hex[:8]}"


def load_index_state() -> dict[str, Any]:
    if not ACTIVE_INDEX_PATH.exists():
        return {
            "schema_version": 1,
            "active_generation": None,
            "previous_generation": None,
            "updated_at": "",
        }
    try:
        payload = json.loads(ACTIVE_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": 1,
            "active_generation": None,
            "previous_generation": None,
            "updated_at": "",
        }
    return {
        "schema_version": 1,
        "active_generation": payload.get("active_generation"),
        "previous_generation": payload.get("previous_generation"),
        "updated_at": str(payload.get("updated_at") or ""),
    }


def active_generation_id() -> str | None:
    value = load_index_state().get("active_generation")
    return str(value) if value else None


def publish_generation(generation_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if not generation_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in generation_id):
        raise ValueError("generation_id 无效")
    published_at = _now()
    generation_manifest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "published_at": published_at,
        **manifest,
    }
    _atomic_write_json(generation_artifact_path(generation_id, "manifest.json"), generation_manifest)
    current = load_index_state()
    state = {
        "schema_version": 1,
        "active_generation": generation_id,
        "previous_generation": current.get("active_generation"),
        "updated_at": published_at,
    }
    _atomic_write_json(ACTIVE_INDEX_PATH, state)
    return state


def rollback_generation() -> dict[str, Any]:
    state = load_index_state()
    current = state.get("active_generation")
    previous = state.get("previous_generation")
    if not current or not previous:
        raise ValueError("没有可回滚的上一索引代")
    previous_manifest = load_generation_manifest(str(previous))
    if previous_manifest.get("permission_schema_version") != 1:
        raise ValueError("上一索引代与当前权限模型不兼容")
    rolled_back = {
        "schema_version": 1,
        "active_generation": previous,
        "previous_generation": current,
        "updated_at": _now(),
    }
    _atomic_write_json(ACTIVE_INDEX_PATH, rolled_back)
    return rolled_back


def load_generation_manifest(generation_id: str) -> dict[str, Any]:
    path = generation_artifact_path(generation_id, "manifest.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"索引代清单不可用：{generation_id}") from exc
    return payload if isinstance(payload, dict) else {}


def generation_artifact_path(generation_id: str, name: str) -> Path:
    directory = GENERATIONS_DIR / generation_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def generation_collection_name(base_collection: str, generation_id: str, kind: str = "text") -> str:
    suffix = {"text": "text", "graph_entity": "graph_entity", "graph_relationship": "graph_relationship"}.get(kind)
    if suffix is None:
        raise ValueError(f"不支持的索引集合类型：{kind}")
    return f"{base_collection}__{suffix}__{generation_id}"


def active_artifact_path(name: str, fallback: Path) -> Path:
    generation_id = active_generation_id()
    if not generation_id:
        return fallback
    path = generation_artifact_path(generation_id, name)
    return path if path.exists() else fallback


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
