from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from services.rag_api.config import get_settings
from services.rag_api import index_generation
from services.rag_api.document.categories import save_kb_categories
from services.rag_api.document import doc_status
from services.rag_api.document.kb_naming import ensure_knowledge_base_name
from services.rag_api.document.loader import load_document, scan_supported_files
from services.rag_api.document.manifest import load_active_catalog
from services.rag_api.document.multi_vector import expand_multi_vector_chunks
from services.rag_api.document.splitter import split_documents
from services.rag_api.graph.kb_graph_builder import build_and_save_kb_graph
from services.rag_api.graph.schema_config import generate_graph_schema_suggestion
from services.rag_api.graph.graph_vector_store import index_graph_vectors_generation
from services.rag_api.rag_settings import load_rag_settings
from services.rag_api.vector.chroma_store import build_generation_chunks, embedding_batch_count


ProgressCallback = Callable[[dict[str, Any]], None]


def ingest_knowledge_base(progress_callback: ProgressCallback | None = None, full_rebuild: bool = False) -> dict:
    total_units = 7

    def progress(completed_units: int, current_step: str, message: str, **extra: Any) -> None:
        if progress_callback:
            percent = int(completed_units / total_units * 100) if total_units else 0
            progress_callback(
                {
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "percent": percent,
                    "current_step": current_step,
                    "message": message,
                    **extra,
                }
            )

    settings = get_settings()
    rag_settings = load_rag_settings()
    docs_dirs = [path.resolve() for path in getattr(settings, "docs_dirs", [settings.docs_dir])]
    progress(1, "扫描文档", "正在扫描并读取知识库目录")
    scanned_files = scan_supported_files(docs_dirs)
    build_cutoff = datetime.now(timezone.utc)
    generation_id = index_generation.new_generation_id(build_cutoff)
    index_generation.record_generation_resources(generation_id, settings.collection_name)
    catalog = load_active_catalog(docs_dirs, scanned_files, cutoff=build_cutoff)
    active_documents = {Path(item["path"]).resolve(): item for item in catalog["documents"]}
    files = sorted(active_documents, key=lambda path: str(path).lower())
    source_manifest_path = index_generation.active_artifact_path("doc_status.json", doc_status.DOC_STATUS_PATH)
    source_snapshot_dir = index_generation.active_artifact_path("snapshots", doc_status.DOC_SNAPSHOT_DIR)
    staging_snapshot_dir = index_generation.generation_artifact_path(generation_id, "snapshots")
    staging_snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = doc_status.load_manifest(source_manifest_path)
    fingerprint = doc_status.pipeline_fingerprint(settings, rag_settings)
    embedding_fingerprint = doc_status.embedding_fingerprint(settings)
    current_doc_ids = {str(item["document_id"]) for item in active_documents.values()}
    file_hashes = {path: doc_status.hash_file(path) for path in files}
    documents_by_id: dict[str, dict[str, Any]] = {}
    chunks_by_id: dict[str, list[dict[str, Any]]] = {}
    processed_count = 0
    skipped_count = 0
    removed_count = 0
    duplicate_count = 0
    failed_count = 0
    records = dict(manifest.get("documents", {}))

    if full_rebuild:
        records = {}

    tombstones = _retained_tombstones(manifest.get("tombstones", []), build_cutoff)
    for removed_doc_id in sorted(set(records) - current_doc_ids):
        previous = records.pop(removed_doc_id)
        tombstones.append(
            {
                "document_id": removed_doc_id,
                "document_version": previous.get("document_version", ""),
                "deleted_at": catalog["build_cutoff"],
                "reason": "source_removed_or_retired",
            }
        )
        removed_count += 1

    chunk_size = rag_settings.chunk_size
    chunk_overlap = rag_settings.chunk_overlap
    content_index: dict[str, str] = {}
    progress(2, "读取解析", f"已读取 {len(files)} 份文档，正在准备切片")

    for path in files:
        governance = active_documents[path]
        doc_id = str(governance["document_id"])
        manifest_revision = _manifest_revision(governance)
        previous = records.get(doc_id)
        if previous and str(previous.get("document_version") or "") != str(governance["version"]):
            tombstones.append(
                {
                    "document_id": doc_id,
                    "document_version": str(previous.get("document_version") or ""),
                    "deleted_at": catalog["build_cutoff"],
                    "reason": "version_replaced",
                }
            )
        file_hash = file_hashes[path]
        if _can_reuse_snapshot(previous, file_hash, fingerprint, manifest_revision):
            snapshot = doc_status.load_snapshot(doc_id, source_snapshot_dir)
            if snapshot:
                document, chunks = _snapshot_payload(snapshot)
                doc_status.save_snapshot(doc_id, document, chunks, staging_snapshot_dir)
                documents_by_id[doc_id] = document
                chunks_by_id[doc_id] = chunks
                if previous.get("content_hash"):
                    content_index.setdefault(str(previous["content_hash"]), doc_id)
                skipped_count += 1
                continue

        try:
            document = load_document(path)
            content_hash = doc_status.hash_content(document.get("content", ""))
            duplicate_of = content_index.get(content_hash)
            if duplicate_of and duplicate_of != doc_id:
                duplicate_count += 1

            document = {
                **document,
                "doc_id": doc_id,
                "document_id": doc_id,
                "content_hash": content_hash,
                "file_hash": file_hash,
                "version": str(governance["version"]),
                "publish_status": str(governance["status"]),
                "effective_at": str(governance["effective_at"]),
                "updated_at": str(governance["updated_at"]),
                "document_revision": doc_status.hash_content(f"{manifest_revision}:{content_hash}"),
                "acl_visibility": governance["acl"]["visibility"],
                "acl_users": json.dumps(governance["acl"]["users"], ensure_ascii=False),
                "acl_roles": json.dumps(governance["acl"]["roles"], ensure_ascii=False),
                "acl_groups": json.dumps(governance["acl"]["groups"], ensure_ascii=False),
                "policy_ref": governance["acl"]["policy_ref"],
                "acl_revision": governance["acl"]["revision"],
            }
            base_chunks = split_documents([document], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = expand_multi_vector_chunks(base_chunks, rag_settings)
            records[doc_id] = doc_status.processed_record(
                doc_id=doc_id,
                path=path,
                file_hash=file_hash,
                content_hash=content_hash,
                chunk_ids=[str(chunk["id"]) for chunk in chunks],
                fingerprint=fingerprint,
            )
            records[doc_id].update(_governance_record(governance, manifest_revision))
            doc_status.save_snapshot(doc_id, document, chunks, staging_snapshot_dir)
            documents_by_id[doc_id] = document
            chunks_by_id[doc_id] = chunks
            content_index.setdefault(content_hash, doc_id)
            processed_count += 1
        except Exception as exc:  # noqa: BLE001
            records[doc_id] = doc_status.failed_record(
                doc_id=doc_id,
                path=path,
                file_hash=file_hash,
                error=str(exc),
                previous=previous,
                fingerprint=fingerprint,
            )
            records[doc_id].update(_governance_record(governance, manifest_revision))
            failed_count += 1
            raise

    documents = _ordered_values(documents_by_id)
    chunks = [chunk for doc_id in sorted(chunks_by_id) for chunk in chunks_by_id[doc_id]]
    for chunk in chunks:
        chunk.setdefault("metadata", {})["embedding_fingerprint"] = embedding_fingerprint
        chunk["metadata"]["generation_id"] = generation_id
    progress(
        3,
        "清洗切片",
        f"已处理 {processed_count} 份文档，跳过 {skipped_count} 份，删除 {removed_count} 份，生成 {len(chunks)} 个基础片段",
    )
    vector_batch_units = embedding_batch_count(len(chunks))
    total_units = 7 + vector_batch_units
    progress(4, "多粒度处理", f"待入库片段数量 {len(chunks)}")
    vector_base_units = 4

    def vector_progress(update: dict[str, Any]) -> None:
        detail_current = int(update.get("detail_current") or 0)
        completed_units = min(vector_base_units + detail_current, vector_base_units + vector_batch_units)
        progress(
            completed_units,
            update.get("current_step", "向量化"),
            update.get("message", "正在向量化知识库片段"),
            detail_current=update.get("detail_current"),
            detail_total=update.get("detail_total"),
        )

    vector_stats = build_generation_chunks(
        chunks,
        generation_id,
        full_rebuild=full_rebuild,
        progress_callback=vector_progress,
    )
    reindexed_chunk_count = vector_stats["embedded_chunk_count"]
    chunk_count = len(chunks)
    progress(
        4 + vector_batch_units,
        "写入向量库",
        f"已写入 {chunk_count} 个 Chroma 向量片段，复用 {vector_stats['reused_embedding_count']} 个向量",
    )
    category_payload = save_kb_categories(
        documents,
        chunks,
        path=index_generation.generation_artifact_path(generation_id, "categories.json"),
    )
    progress(5 + vector_batch_units, "生成分类", f"已生成 {len(category_payload['items'])} 个知识库分类")
    kb_graph = build_and_save_kb_graph(
        category_payload,
        documents,
        chunks,
        path=index_generation.generation_artifact_path(generation_id, "graph.json"),
    )
    graph_index_stats = index_graph_vectors_generation(
        kb_graph.get("nodes", []),
        kb_graph.get("edges", []),
        generation_id,
        full_rebuild=full_rebuild,
    )
    progress(
        6 + vector_batch_units,
        "生成图谱索引",
        f"已写入 {graph_index_stats['graph_entity_index_count']} 个实体向量、{graph_index_stats['graph_relationship_index_count']} 个关系向量",
    )
    knowledge_base_name, knowledge_base_name_source = ensure_knowledge_base_name(category_payload, documents, chunk_count)
    graph_schema_suggestion = generate_graph_schema_suggestion(
        category_payload,
        documents,
        chunks,
        path=index_generation.generation_artifact_path(generation_id, "graph_schema_suggestion.json"),
    )
    progress(7 + vector_batch_units, "生成知识图谱", "知识库重建完成")
    doc_status.save_manifest(
        {"version": 2, "documents": records, "tombstones": tombstones},
        index_generation.generation_artifact_path(generation_id, "doc_status.json"),
    )
    index_generation.publish_generation(
        generation_id,
        {
            "permission_schema_version": 1,
            "required_artifacts": [
                "categories.json",
                "graph.json",
                "graph_schema_suggestion.json",
                "doc_status.json",
                "snapshots",
                "resources.json",
            ],
            "build_cutoff": catalog["build_cutoff"],
            "next_activation_at": catalog["next_activation_at"],
            "pipeline_fingerprint": fingerprint,
            "embedding_fingerprint": embedding_fingerprint,
            "documents": records,
            "tombstones": tombstones,
            "warnings": catalog["warnings"],
            "stats": {
                "document_count": len(documents),
                "chunk_count": chunk_count,
                **vector_stats,
                **graph_index_stats,
            },
        },
    )
    from services.rag_api.memory import conversation_memory
    from services.rag_api.retrieval.cache import RETRIEVAL_CACHE

    RETRIEVAL_CACHE.clear()
    conversation_memory.SESSION_MEMORY.clear()
    return {
        "status": "success",
        "generation_id": generation_id,
        "incremental": not full_rebuild,
        "build_cutoff": catalog["build_cutoff"],
        "next_activation_at": catalog["next_activation_at"],
        "governance_warnings": catalog["warnings"],
        "kb_dir": str(docs_dirs[0]),
        "kb_dirs": [str(path) for path in docs_dirs],
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "processed_document_count": processed_count,
        "skipped_document_count": skipped_count,
        "removed_document_count": removed_count,
        "duplicate_document_count": duplicate_count,
        "failed_document_count": failed_count,
        "reindexed_chunk_count": reindexed_chunk_count,
        "reused_embedding_count": vector_stats["reused_embedding_count"],
        "collection": settings.collection_name,
        "multi_vector_enabled": rag_settings.multi_vector_enabled,
        "categories": category_payload["items"],
        "category_count": len(category_payload["items"]),
        "knowledge_base_name": knowledge_base_name,
        "knowledge_base_name_source": knowledge_base_name_source,
        "graph_node_count": len(kb_graph.get("nodes", [])),
        "graph_edge_count": len(kb_graph.get("edges", [])),
        **graph_index_stats,
        "graph_schema_suggestion": graph_schema_suggestion,
    }


def _can_reuse_snapshot(previous: dict[str, Any] | None, file_hash: str, fingerprint: str, manifest_revision: str) -> bool:
    return bool(
        previous
        and previous.get("status") == doc_status.PROCESSED
        and previous.get("file_hash") == file_hash
        and previous.get("pipeline_fingerprint") == fingerprint
        and previous.get("manifest_revision") == manifest_revision
        and previous.get("chunk_ids")
    )


def _snapshot_payload(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = snapshot.get("document")
    chunks = snapshot.get("chunks")
    return (document if isinstance(document, dict) else {}, chunks if isinstance(chunks, list) else [])


def _ordered_values(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [items[key] for key in sorted(items)]


def _manifest_revision(governance: dict[str, Any]) -> str:
    payload = {
        "document_id": governance.get("document_id"),
        "version": governance.get("version"),
        "status": governance.get("status"),
        "effective_at": governance.get("effective_at"),
        "updated_at": governance.get("updated_at"),
        "path": str(governance.get("path") or ""),
        "knowledge_base_id": str(governance.get("knowledge_base_id") or ""),
        "acl": governance.get("acl"),
    }
    return doc_status.hash_content(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _governance_record(governance: dict[str, Any], manifest_revision: str) -> dict[str, Any]:
    return {
        "document_version": str(governance["version"]),
        "effective_at": str(governance["effective_at"]),
        "document_updated_at": str(governance["updated_at"]),
        "acl": governance["acl"],
        "manifest_revision": manifest_revision,
    }


def _retained_tombstones(tombstones: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    threshold = cutoff.astimezone(timezone.utc) - timedelta(days=30)
    retained: list[dict[str, Any]] = []
    for tombstone in tombstones:
        try:
            deleted_at = datetime.fromisoformat(str(tombstone.get("deleted_at") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if deleted_at.tzinfo is not None and deleted_at.astimezone(timezone.utc) >= threshold:
            retained.append(tombstone)
    return retained


if __name__ == "__main__":
    import json

    print(json.dumps(ingest_knowledge_base(), ensure_ascii=False, indent=2))
