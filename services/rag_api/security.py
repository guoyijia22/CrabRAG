from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from services.rag_api import index_generation


@dataclass(frozen=True)
class PrincipalContext:
    subject: str
    roles: tuple[str, ...]
    groups: tuple[str, ...]
    permission_revision: str
    can_manage_index: bool = False

    @classmethod
    def anonymous(cls) -> "PrincipalContext":
        return cls(subject="anonymous", roles=(), groups=(), permission_revision="anonymous-v1")


@dataclass(frozen=True)
class RetrievalContext:
    generation_id: str
    principal: PrincipalContext
    allowed_document_ids: frozenset[str] | None
    permission_fingerprint: str


class PermissionProvider(Protocol):
    def allowed_document_ids(self, principal: PrincipalContext, generation_manifest: dict[str, Any]) -> frozenset[str]: ...


class PermissionServiceError(RuntimeError):
    pass


class LocalPermissionProvider:
    def allowed_document_ids(self, principal: PrincipalContext, generation_manifest: dict[str, Any]) -> frozenset[str]:
        allowed: set[str] = set()
        for document_id, record in (generation_manifest.get("documents") or {}).items():
            acl = record.get("acl") or {"visibility": "public"}
            if acl.get("visibility") == "public":
                allowed.add(str(document_id))
                continue
            if principal.subject != "anonymous" and principal.subject in _string_set(acl.get("users")):
                allowed.add(str(document_id))
                continue
            if set(principal.roles) & _string_set(acl.get("roles")):
                allowed.add(str(document_id))
                continue
            if set(principal.groups) & _string_set(acl.get("groups")):
                allowed.add(str(document_id))
        return frozenset(allowed)


def principal_from_headers(headers: Mapping[str, str], *, internal_token: str | None) -> PrincipalContext:
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    supplied_token = normalized.get("x-crabrag-internal-token", "")
    if not internal_token or supplied_token != internal_token:
        return PrincipalContext.anonymous()
    subject = normalized.get("x-crabrag-subject", "").strip() or "anonymous"
    return PrincipalContext(
        subject=subject,
        roles=_csv_tuple(normalized.get("x-crabrag-roles", "")),
        groups=_csv_tuple(normalized.get("x-crabrag-groups", "")),
        permission_revision=normalized.get("x-crabrag-permission-revision", "").strip() or "1",
        can_manage_index=normalized.get("x-crabrag-admin", "").strip().lower() in {"1", "true", "yes"},
    )


_RETRIEVAL_CONTEXT: ContextVar[RetrievalContext | None] = ContextVar("crabrag_retrieval_context", default=None)


def build_retrieval_context(
    principal: PrincipalContext,
    permission_provider: PermissionProvider | None = None,
) -> RetrievalContext:
    generation_id = index_generation.active_generation_id()
    if not generation_id:
        return RetrievalContext(
            generation_id="legacy",
            principal=principal,
            allowed_document_ids=None,
            permission_fingerprint=_permission_fingerprint(principal, "legacy"),
        )
    try:
        manifest = index_generation.load_generation_manifest(generation_id)
        allowed = (permission_provider or LocalPermissionProvider()).allowed_document_ids(principal, manifest)
    except Exception as exc:  # noqa: BLE001
        raise PermissionServiceError("权限上下文解析失败，已拒绝检索") from exc
    return RetrievalContext(
        generation_id=generation_id,
        principal=principal,
        allowed_document_ids=allowed,
        permission_fingerprint=_permission_fingerprint(principal, generation_id),
    )


def current_retrieval_context() -> RetrievalContext | None:
    return _RETRIEVAL_CONTEXT.get()


def pinned_artifact_path(name: str, fallback: Path) -> Path:
    context = current_retrieval_context()
    if context and context.generation_id != "legacy":
        path = index_generation.generation_artifact_path(context.generation_id, name)
        return path if path.exists() else fallback
    return index_generation.active_artifact_path(name, fallback)


def filter_graph_by_permission(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = current_retrieval_context()
    if context is None or context.allowed_document_ids is None:
        return nodes, edges
    allowed = context.allowed_document_ids
    filtered_nodes: list[dict[str, Any]] = []
    for node in nodes:
        document_ids = {str(item) for item in node.get("document_ids", []) or [] if item}
        authorized_ids = document_ids & allowed
        if not authorized_ids:
            continue
        document_sources = [
            item
            for item in node.get("document_sources", []) or []
            if str(item.get("document_id") or "") in authorized_ids
        ]
        filtered_nodes.append(
            {
                **node,
                "document_ids": sorted(authorized_ids),
                "document_sources": document_sources,
                "source_files": sorted({str(item.get("source_file")) for item in document_sources if item.get("source_file")}),
            }
        )
    filtered_edges = [
        edge
        for edge in edges
        if str(edge.get("document_id") or "") in allowed
    ]
    return filtered_nodes, filtered_edges


@contextmanager
def use_retrieval_context(context: RetrievalContext):
    token = _RETRIEVAL_CONTEXT.set(context)
    try:
        yield context
    finally:
        _RETRIEVAL_CONTEXT.reset(token)


def _permission_fingerprint(principal: PrincipalContext, generation_id: str) -> str:
    payload = {
        "generation_id": generation_id,
        "subject": principal.subject,
        "roles": sorted(principal.roles),
        "groups": sorted(principal.groups),
        "permission_revision": principal.permission_revision,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _string_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value if str(item)}
    return set()


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
