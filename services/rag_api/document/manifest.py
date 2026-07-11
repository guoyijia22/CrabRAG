from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = ".crabrag-manifest.json"
VALID_STATUSES = {"draft", "published", "retired"}


class ManifestError(ValueError):
    pass


def load_active_catalog(docs_dirs: list[Path], files: list[Path], *, cutoff: datetime) -> dict[str, Any]:
    roots = [root.expanduser().resolve() for root in docs_dirs]
    assigned: dict[Path, list[Path]] = {root: [] for root in roots}
    for file_path in (item.expanduser().resolve() for item in files):
        candidates = [root for root in roots if _is_relative_to(file_path, root)]
        if candidates:
            assigned[max(candidates, key=lambda item: len(item.parts))].append(file_path)

    documents: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    next_activation: datetime | None = None
    seen_document_ids: set[str] = set()
    for root in roots:
        manifest = load_or_create_manifest(root, assigned[root], now=cutoff)
        selection_manifest = _without_missing_automatic_documents(manifest, root, cutoff)
        root_documents = select_active_versions(selection_manifest, root, cutoff=cutoff)
        for document in root_documents:
            document_id = document["document_id"]
            if document_id in seen_document_ids:
                raise ManifestError(f"多个知识库目录使用了相同 document_id：{document_id}")
            seen_document_ids.add(document_id)
            documents.append(document)
        warnings.extend({**warning, "knowledge_base_id": manifest["knowledge_base_id"]} for warning in manifest.get("audit_warnings", []))
        for raw in manifest["documents"]:
            entry = _normalize_entry(raw)
            if entry["status"] != "published":
                continue
            effective = _parse_timestamp(entry["effective_at"], "effective_at")
            if effective > cutoff.astimezone(timezone.utc) and (next_activation is None or effective < next_activation):
                next_activation = effective
    return {
        "documents": sorted(documents, key=lambda item: item["document_id"]),
        "warnings": warnings,
        "next_activation_at": _format_timestamp(next_activation) if next_activation else None,
        "build_cutoff": _format_timestamp(cutoff),
    }


def load_or_create_manifest(root: Path, files: list[Path], *, now: datetime | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    path = root / MANIFEST_FILENAME
    timestamp = _format_timestamp(now or datetime.now(timezone.utc))
    changed = False
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"知识库清单无法读取：{path}") from exc
        if not isinstance(manifest, dict):
            raise ManifestError(f"知识库清单必须是 JSON 对象：{path}")
    else:
        manifest = {
            "schema_version": 1,
            "knowledge_base_id": f"kb-{uuid.uuid4().hex}",
            "documents": [],
            "audit_warnings": [],
        }
        changed = True

    _validate_manifest_shape(manifest, path)
    registered_paths = {str(item.get("path") or "").replace("\\", "/") for item in manifest["documents"]}
    warnings = manifest.setdefault("audit_warnings", [])
    for file_path in sorted((item.expanduser().resolve() for item in files), key=lambda item: str(item).lower()):
        try:
            relative_path = file_path.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative_path in registered_paths:
            continue
        document_id = _automatic_document_id(str(manifest["knowledge_base_id"]), relative_path)
        manifest["documents"].append(
            {
                "document_id": document_id,
                "path": relative_path,
                "version": "1",
                "status": "published",
                "effective_at": timestamp,
                "updated_at": timestamp,
                "acl": _public_acl(),
            }
        )
        warnings.append(
            {
                "code": "AUTO_PUBLIC_DOCUMENT",
                "severity": "high",
                "document_id": document_id,
                "path": relative_path,
                "detected_at": timestamp,
            }
        )
        registered_paths.add(relative_path)
        changed = True

    if changed:
        _atomic_write_json(path, manifest)
    return manifest


def select_active_versions(manifest: dict[str, Any], root: Path, *, cutoff: datetime) -> list[dict[str, Any]]:
    _validate_manifest_shape(manifest, root / MANIFEST_FILENAME)
    if cutoff.tzinfo is None:
        raise ManifestError("build_cutoff 必须包含时区")
    cutoff = cutoff.astimezone(timezone.utc)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in manifest["documents"]:
        entry = _normalize_entry(raw)
        grouped.setdefault(entry["document_id"], []).append(entry)

    active: list[dict[str, Any]] = []
    for document_id, entries in grouped.items():
        published_times: dict[datetime, str] = {}
        for entry in entries:
            if entry["status"] != "published":
                continue
            effective = _parse_timestamp(entry["effective_at"], "effective_at")
            if effective in published_times:
                raise ManifestError(f"文档 {document_id} 存在相同生效时间的多个已发布版本")
            published_times[effective] = entry["version"]
        candidates = [
            entry
            for entry in entries
            if entry["status"] == "published" and _parse_timestamp(entry["effective_at"], "effective_at") <= cutoff
        ]
        if not candidates:
            continue
        selected = max(candidates, key=lambda item: _parse_timestamp(item["effective_at"], "effective_at"))
        source_path = (root / selected["path"]).resolve()
        try:
            source_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ManifestError(f"文档路径超出知识库目录：{selected['path']}") from exc
        if not source_path.is_file():
            raise ManifestError(f"已发布文档不存在：{selected['path']}")
        active.append({**selected, "path": source_path, "knowledge_base_id": manifest["knowledge_base_id"]})
    return sorted(active, key=lambda item: item["document_id"])


def _validate_manifest_shape(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ManifestError(f"不支持的知识库清单版本：{path}")
    if not isinstance(manifest.get("knowledge_base_id"), str) or not manifest["knowledge_base_id"].strip():
        raise ManifestError(f"知识库清单缺少 knowledge_base_id：{path}")
    if not isinstance(manifest.get("documents"), list):
        raise ManifestError(f"知识库清单 documents 必须是数组：{path}")
    if not isinstance(manifest.setdefault("audit_warnings", []), list):
        raise ManifestError(f"知识库清单 audit_warnings 必须是数组：{path}")


def _normalize_entry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError("知识库清单文档项必须是对象")
    required = ("document_id", "path", "version", "status", "effective_at", "updated_at")
    for key in required:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ManifestError(f"知识库清单文档项缺少 {key}")
    if raw["status"] not in VALID_STATUSES:
        raise ManifestError(f"不支持的文档发布状态：{raw['status']}")
    _parse_timestamp(raw["effective_at"], "effective_at")
    _parse_timestamp(raw["updated_at"], "updated_at")
    acl = raw.get("acl") or _public_acl()
    if not isinstance(acl, dict) or acl.get("visibility") not in {"public", "restricted"}:
        raise ManifestError(f"文档 {raw['document_id']} 的 ACL 无效")
    normalized_acl = {
        "visibility": acl["visibility"],
        "users": _string_list(acl.get("users")),
        "roles": _string_list(acl.get("roles")),
        "groups": _string_list(acl.get("groups")),
        "policy_ref": str(acl.get("policy_ref") or ""),
        "revision": str(acl.get("revision") or "1"),
    }
    return {**raw, "acl": normalized_acl, "path": str(raw["path"]).replace("\\", "/")}


def _automatic_document_id(knowledge_base_id: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{knowledge_base_id}:{relative_path.lower()}".encode("utf-8")).hexdigest()
    return f"doc-{digest[:20]}"


def _public_acl() -> dict[str, Any]:
    return {"visibility": "public", "users": [], "roles": [], "groups": [], "policy_ref": "", "revision": "1"}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{field} 必须是 ISO-8601 时间：{value}") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"{field} 必须包含时区：{value}")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _without_missing_automatic_documents(manifest: dict[str, Any], root: Path, cutoff: datetime) -> dict[str, Any]:
    automatic_ids = {
        str(warning.get("document_id") or "")
        for warning in manifest.get("audit_warnings", [])
        if warning.get("code") == "AUTO_PUBLIC_DOCUMENT"
    }
    available: list[dict[str, Any]] = []
    changed = False
    warnings = manifest.setdefault("audit_warnings", [])
    missing_warning_ids = {
        str(warning.get("document_id") or "")
        for warning in warnings
        if warning.get("code") == "AUTO_PUBLIC_DOCUMENT_DELETED"
    }
    for raw in manifest["documents"]:
        document_id = str(raw.get("document_id") or "")
        source_path = (root / str(raw.get("path") or "")).resolve()
        if document_id not in automatic_ids or source_path.is_file():
            available.append(raw)
            continue
        if document_id not in missing_warning_ids:
            warnings.append(
                {
                    "code": "AUTO_PUBLIC_DOCUMENT_DELETED",
                    "severity": "high",
                    "document_id": document_id,
                    "path": str(raw.get("path") or ""),
                    "detected_at": _format_timestamp(cutoff),
                }
            )
            changed = True
    if changed:
        _atomic_write_json(root / MANIFEST_FILENAME, manifest)
    return {**manifest, "documents": available}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
