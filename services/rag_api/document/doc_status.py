from __future__ import annotations

import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR, Settings
from services.rag_api.rag_settings import RagSettings

DOC_STATUS_PATH = PROJECT_DIR / "data" / "ingest" / "doc_status.json"
DOC_SNAPSHOT_DIR = PROJECT_DIR / "data" / "ingest" / "doc_snapshots"
CHUNK_IDENTITY_SCHEMA_VERSION = 2

PROCESSED = "PROCESSED"
DUPLICATE = "DUPLICATE"
FAILED = "FAILED"


def empty_manifest() -> dict[str, Any]:
    return {"version": 1, "documents": {}, "updated_at": ""}


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DOC_STATUS_PATH
    if not manifest_path.exists():
        return empty_manifest()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_manifest()
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), dict):
        return empty_manifest()
    payload.setdefault("version", 1)
    payload.setdefault("updated_at", "")
    return payload


def save_manifest(manifest: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DOC_STATUS_PATH
    manifest["updated_at"] = _now()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def document_id_for_path(path: Path) -> str:
    normalized = str(path.expanduser().resolve()).replace("\\", "/").lower()
    return f"doc-{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:16]}"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def pipeline_fingerprint(settings: Settings, rag_settings: RagSettings) -> str:
    payload = {
        "chunk_identity_schema_version": CHUNK_IDENTITY_SCHEMA_VERSION,
        "chunk_size": rag_settings.chunk_size,
        "chunk_overlap": rag_settings.chunk_overlap,
        "multi_vector_enabled": rag_settings.multi_vector_enabled,
        "embedding_fingerprint": embedding_fingerprint(settings),
        "collection_name": settings.collection_name,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def embedding_fingerprint(settings: Settings) -> str:
    payload = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_onnx_model_file": settings.embedding_onnx_model_file,
    }
    if settings.embedding_provider == "local_onnx":
        payload["local_artifacts"] = _local_embedding_artifacts(settings)
    else:
        payload["embedding_base_url"] = str(settings.embedding_base_url).rstrip("/")
        payload["embedding_openai_compatible"] = bool(settings.embedding_openai_compatible)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _local_embedding_artifacts(settings: Settings) -> dict[str, str]:
    model_dir = Path(settings.local_embedding_model_dir)
    model_candidates = [
        model_dir / settings.embedding_onnx_model_file,
        model_dir / "onnx" / settings.embedding_onnx_model_file,
    ]
    model_path = next((path for path in model_candidates if path.is_file()), model_candidates[0])
    paths = [
        model_path,
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"本地向量模型文件缺失：{', '.join(missing)}")
    artifacts: dict[str, str] = {}
    for path in paths:
        stat = path.stat()
        artifacts[path.relative_to(model_dir).as_posix()] = _cached_artifact_hash(
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
        )
    return artifacts


@lru_cache(maxsize=64)
def _cached_artifact_hash(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return hash_file(Path(path))


def load_snapshot(doc_id: str, directory: Path | None = None) -> dict[str, Any] | None:
    path = _snapshot_path(doc_id, directory)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_snapshot(doc_id: str, document: dict[str, Any], chunks: list[dict[str, Any]], directory: Path | None = None) -> None:
    snapshot_dir = directory or DOC_SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_path(doc_id, snapshot_dir).write_text(
        json.dumps({"document": document, "chunks": chunks, "updated_at": _now()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def delete_snapshot(doc_id: str, directory: Path | None = None) -> None:
    try:
        _snapshot_path(doc_id, directory).unlink()
    except FileNotFoundError:
        pass


def processed_content_index(manifest: dict[str, Any], current_doc_ids: set[str] | None = None) -> dict[str, str]:
    index: dict[str, str] = {}
    current_doc_ids = current_doc_ids or set()
    for doc_id, record in manifest.get("documents", {}).items():
        if current_doc_ids and doc_id not in current_doc_ids:
            continue
        if record.get("status") == PROCESSED and record.get("content_hash"):
            index[str(record["content_hash"])] = str(doc_id)
    return index


def processed_record(
    *,
    doc_id: str,
    path: Path,
    file_hash: str,
    content_hash: str,
    chunk_ids: list[str],
    fingerprint: str,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "doc_id": doc_id,
        "source_path": str(path.resolve()),
        "source_file": path.name,
        "file_hash": file_hash,
        "content_hash": content_hash,
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
        "status": PROCESSED,
        "chunk_ids": chunk_ids,
        "pipeline_fingerprint": fingerprint,
        "updated_at": _now(),
    }


def duplicate_record(
    *,
    doc_id: str,
    path: Path,
    file_hash: str,
    content_hash: str,
    duplicate_of: str,
    fingerprint: str,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "doc_id": doc_id,
        "source_path": str(path.resolve()),
        "source_file": path.name,
        "file_hash": file_hash,
        "content_hash": content_hash,
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
        "status": DUPLICATE,
        "duplicate_of": duplicate_of,
        "chunk_ids": [],
        "pipeline_fingerprint": fingerprint,
        "updated_at": _now(),
    }


def failed_record(
    *,
    doc_id: str,
    path: Path,
    file_hash: str,
    error: str,
    previous: dict[str, Any] | None,
    fingerprint: str,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "doc_id": doc_id,
        "source_path": str(path.resolve()),
        "source_file": path.name,
        "file_hash": file_hash,
        "content_hash": previous.get("content_hash", "") if previous else "",
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
        "status": FAILED,
        "error": error,
        "chunk_ids": list(previous.get("chunk_ids", [])) if previous else [],
        "pipeline_fingerprint": fingerprint,
        "updated_at": _now(),
    }


def _snapshot_path(doc_id: str, directory: Path | None = None) -> Path:
    safe = "".join(ch for ch in doc_id if ch.isalnum() or ch in {"-", "_"})
    return (directory or DOC_SNAPSHOT_DIR) / f"{safe}.json"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
