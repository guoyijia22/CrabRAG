from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR


INDEX_ROOT = PROJECT_DIR / "data" / "index"
ACTIVE_INDEX_PATH = INDEX_ROOT / "active.json"
GENERATIONS_DIR = INDEX_ROOT / "generations"
INDEX_LOCK_PATH = INDEX_ROOT / "build.lock"
_STATE_LOCK = threading.RLock()
_PINNED_GENERATIONS: dict[str, int] = {}


class IndexStateError(RuntimeError):
    pass


def new_generation_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gen-{timestamp}-{uuid.uuid4().hex[:8]}"


def load_index_state() -> dict[str, Any]:
    if not ACTIVE_INDEX_PATH.exists():
        if (INDEX_ROOT / ".governed").exists():
            raise IndexStateError("活动索引指针缺失，已拒绝降级到 legacy 索引")
        return {
            "schema_version": 1,
            "active_generation": None,
            "previous_generation": None,
            "updated_at": "",
        }
    try:
        payload = json.loads(ACTIVE_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexStateError("活动索引指针损坏，已拒绝降级到 legacy 索引") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise IndexStateError("活动索引指针格式不兼容")
    return {
        "schema_version": 1,
        "active_generation": payload.get("active_generation"),
        "previous_generation": payload.get("previous_generation"),
        "updated_at": str(payload.get("updated_at") or ""),
    }


def active_generation_id() -> str | None:
    value = load_index_state().get("active_generation")
    return str(value) if value else None


@contextmanager
def generation_build_lock():
    INDEX_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_LOCK_PATH.exists():
        age_seconds = datetime.now(timezone.utc).timestamp() - INDEX_LOCK_PATH.stat().st_mtime
        if age_seconds > 24 * 60 * 60:
            INDEX_LOCK_PATH.unlink()
    try:
        descriptor = os.open(INDEX_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("已有跨进程知识库构建正在运行") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started_at={_now()}".encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            INDEX_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def publish_generation(generation_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if not generation_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in generation_id):
        raise ValueError("generation_id 无效")
    with _STATE_LOCK:
        published_at = _now()
        generation_manifest = {
            "schema_version": 1,
            "generation_id": generation_id,
            "published_at": published_at,
            **manifest,
        }
        _atomic_write_json(generation_artifact_path(generation_id, "manifest.json"), generation_manifest)
        validate_generation_artifacts(generation_id, generation_manifest)
        current = load_index_state()
        state = {
            "schema_version": 1,
            "active_generation": generation_id,
            "previous_generation": current.get("active_generation"),
            "updated_at": published_at,
        }
        _atomic_write_json(ACTIVE_INDEX_PATH, state)
        marker = INDEX_ROOT / ".governed"
        marker.touch(exist_ok=True)
        return state


def rollback_generation() -> dict[str, Any]:
    with _STATE_LOCK:
        state = load_index_state()
        current = state.get("active_generation")
        previous = state.get("previous_generation")
        if not current or not previous:
            raise ValueError("没有可回滚的上一索引代")
        previous_manifest = load_generation_manifest(str(previous))
        if previous_manifest.get("permission_schema_version") != 1:
            raise ValueError("上一索引代与当前权限模型不兼容")
        validate_generation_artifacts(str(previous), previous_manifest)
        rolled_back = {
            "schema_version": 1,
            "active_generation": previous,
            "previous_generation": current,
            "updated_at": _now(),
        }
        _atomic_write_json(ACTIVE_INDEX_PATH, rolled_back)
        return rolled_back


@contextmanager
def pin_generation(generation_id: str):
    if not generation_id or generation_id == "legacy":
        yield
        return
    with _STATE_LOCK:
        _PINNED_GENERATIONS[generation_id] = _PINNED_GENERATIONS.get(generation_id, 0) + 1
    try:
        yield
    finally:
        with _STATE_LOCK:
            remaining = _PINNED_GENERATIONS.get(generation_id, 1) - 1
            if remaining > 0:
                _PINNED_GENERATIONS[generation_id] = remaining
            else:
                _PINNED_GENERATIONS.pop(generation_id, None)


def load_generation_manifest(generation_id: str) -> dict[str, Any]:
    path = generation_artifact_path(generation_id, "manifest.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"索引代清单不可用：{generation_id}") from exc
    return payload if isinstance(payload, dict) else {}


def record_generation_resources(generation_id: str, base_collection: str) -> dict[str, Any]:
    payload = {
        "generation_id": generation_id,
        "collections": [
            {"kind": kind, "name": generation_collection_name(base_collection, generation_id, kind)}
            for kind in ("text", "graph_entity", "graph_relationship")
        ],
    }
    _atomic_write_json(generation_artifact_path(generation_id, "resources.json"), payload)
    return payload


def register_generation_resource(generation_id: str, kind: str, name: str) -> dict[str, Any]:
    if not generation_id or not kind or not name:
        raise ValueError("索引代资源信息不完整")
    with _STATE_LOCK:
        path = generation_artifact_path(generation_id, "resources.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"索引代资源清单不可用：{generation_id}") from exc
        collections = payload.get("collections") if isinstance(payload, dict) else None
        if not isinstance(collections, list):
            raise ValueError(f"索引代资源清单格式无效：{generation_id}")
        resource = {"kind": kind, "name": name}
        if not any(isinstance(item, dict) and item.get("name") == name for item in collections):
            collections.append(resource)
            _atomic_write_json(path, payload)
        return payload


def validate_generation_artifacts(generation_id: str, manifest: dict[str, Any] | None = None) -> None:
    payload = manifest or load_generation_manifest(generation_id)
    for name in payload.get("required_artifacts", []) or []:
        if not generation_artifact_path(generation_id, str(name)).exists():
            raise ValueError(f"索引代产物缺失：{generation_id}/{name}")
    resources_path = generation_artifact_path(generation_id, "resources.json")
    if payload.get("required_artifacts") and not resources_path.exists():
        raise ValueError(f"索引代资源清单缺失：{generation_id}")


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


def cleanup_generations(base_collection: str, chroma_client, *, now: datetime | None = None) -> dict[str, Any]:
    with _STATE_LOCK:
        if INDEX_LOCK_PATH.exists():
            return {
                "deleted_generations": [],
                "errors": [],
                "skipped": "generation build in progress",
                "cleaned_at": _now(),
            }
        state = load_index_state()
        retained = {str(value) for value in (state.get("active_generation"), state.get("previous_generation")) if value}
        retained.update(_PINNED_GENERATIONS)
        current_time = (now or datetime.now(timezone.utc)).timestamp()
        deleted_generations: list[str] = []
        errors: list[dict[str, str]] = []
        if not GENERATIONS_DIR.exists():
            return {"deleted_generations": [], "errors": [], "cleaned_at": _now()}
        root = GENERATIONS_DIR.resolve()
        for directory in sorted((item for item in GENERATIONS_DIR.iterdir() if item.is_dir()), key=lambda item: item.name):
            generation_id = directory.name
            if generation_id in retained:
                continue
            resolved = directory.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append({"generation_id": generation_id, "error": "generation path escaped index root"})
                continue
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists() and current_time - directory.stat().st_mtime < 24 * 60 * 60:
                continue
            try:
                resources_path = directory / "resources.json"
                if not resources_path.exists():
                    errors.append({"generation_id": generation_id, "error": "missing explicit resources.json"})
                    continue
                resources = json.loads(resources_path.read_text(encoding="utf-8"))
                collections = resources.get("collections", []) if isinstance(resources, dict) else []
                if not isinstance(collections, list):
                    raise ValueError("invalid collection resource list")
                for resource in collections:
                    collection_name = str(resource.get("name") or "") if isinstance(resource, dict) else ""
                    if not collection_name:
                        continue
                    try:
                        chroma_client.delete_collection(collection_name)
                    except Exception:
                        pass
                shutil.rmtree(resolved)
                deleted_generations.append(generation_id)
            except Exception as exc:  # noqa: BLE001
                errors.append({"generation_id": generation_id, "error": str(exc)})
        return {"deleted_generations": deleted_generations, "errors": errors, "cleaned_at": _now()}


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
