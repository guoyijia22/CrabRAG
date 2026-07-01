from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.rag_api.config import get_settings
from services.rag_api.document.categories import save_kb_categories
from services.rag_api.document import doc_status
from services.rag_api.document.kb_naming import ensure_knowledge_base_name
from services.rag_api.document.loader import load_document, scan_supported_files
from services.rag_api.document.multi_vector import expand_multi_vector_chunks
from services.rag_api.document.splitter import split_documents
from services.rag_api.graph.kb_graph_builder import build_and_save_kb_graph
from services.rag_api.graph.schema_config import generate_graph_schema_suggestion
from services.rag_api.graph.graph_vector_store import index_graph_vectors_incremental
from services.rag_api.rag_settings import load_rag_settings
from services.rag_api.vector.chroma_store import embedding_batch_count, upsert_chunks_incremental


ProgressCallback = Callable[[dict[str, Any]], None]


def ingest_knowledge_base(progress_callback: ProgressCallback | None = None) -> dict:
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
    files = scan_supported_files(docs_dirs)
    manifest = doc_status.load_manifest()
    fingerprint = doc_status.pipeline_fingerprint(settings, rag_settings)
    current_doc_ids = {doc_status.document_id_for_path(path) for path in files}
    file_hashes = {path: doc_status.hash_file(path) for path in files}
    documents_by_id: dict[str, dict[str, Any]] = {}
    chunks_by_id: dict[str, list[dict[str, Any]]] = {}
    chunks_to_upsert: list[dict[str, Any]] = []
    delete_chunk_ids: list[str] = []
    processed_count = 0
    skipped_count = 0
    removed_count = 0
    duplicate_count = 0
    failed_count = 0
    records = dict(manifest.get("documents", {}))

    for removed_doc_id in sorted(set(records) - current_doc_ids):
        previous = records.pop(removed_doc_id)
        delete_chunk_ids.extend(str(chunk_id) for chunk_id in previous.get("chunk_ids", []) if chunk_id)
        doc_status.delete_snapshot(removed_doc_id)
        removed_count += 1

    chunk_size = rag_settings.chunk_size
    chunk_overlap = rag_settings.chunk_overlap
    content_index = _unchanged_content_index(files, file_hashes, records, fingerprint)
    progress(2, "读取解析", f"已读取 {len(files)} 份文档，正在准备切片")

    for path in files:
        doc_id = doc_status.document_id_for_path(path)
        previous = records.get(doc_id)
        file_hash = file_hashes[path]
        if _can_reuse_duplicate(previous, file_hash, fingerprint):
            duplicate_count += 1
            continue
        if _can_reuse_snapshot(previous, file_hash, fingerprint):
            snapshot = doc_status.load_snapshot(doc_id)
            if snapshot:
                document, chunks = _snapshot_payload(snapshot)
                documents_by_id[doc_id] = document
                chunks_by_id[doc_id] = chunks
                if previous.get("content_hash"):
                    content_index[str(previous["content_hash"])] = doc_id
                skipped_count += 1
                continue

        delete_chunk_ids.extend(str(chunk_id) for chunk_id in (previous or {}).get("chunk_ids", []) if chunk_id)
        try:
            document = load_document(path)
            content_hash = doc_status.hash_content(document.get("content", ""))
            duplicate_of = content_index.get(content_hash)
            if duplicate_of and duplicate_of != doc_id:
                records[doc_id] = doc_status.duplicate_record(
                    doc_id=doc_id,
                    path=path,
                    file_hash=file_hash,
                    content_hash=content_hash,
                    duplicate_of=duplicate_of,
                    fingerprint=fingerprint,
                )
                doc_status.delete_snapshot(doc_id)
                duplicate_count += 1
                continue

            document = {
                **document,
                "doc_id": doc_id,
                "content_hash": content_hash,
                "file_hash": file_hash,
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
            doc_status.save_snapshot(doc_id, document, chunks)
            documents_by_id[doc_id] = document
            chunks_by_id[doc_id] = chunks
            chunks_to_upsert.extend(chunks)
            content_index[content_hash] = doc_id
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
            doc_status.delete_snapshot(doc_id)
            failed_count += 1

    documents = _ordered_values(documents_by_id)
    chunks = [chunk for doc_id in sorted(chunks_by_id) for chunk in chunks_by_id[doc_id]]
    progress(
        3,
        "清洗切片",
        f"已处理 {processed_count} 份文档，跳过 {skipped_count} 份，删除 {removed_count} 份，生成 {len(chunks)} 个基础片段",
    )
    vector_batch_units = embedding_batch_count(len(chunks_to_upsert))
    total_units = 7 + vector_batch_units
    progress(4, "多粒度处理", f"待入库片段数量 {len(chunks_to_upsert)}")
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

    reindexed_chunk_count = 0
    if chunks_to_upsert or delete_chunk_ids:
        reindexed_chunk_count = upsert_chunks_incremental(
            chunks_to_upsert,
            delete_chunk_ids=delete_chunk_ids,
            progress_callback=vector_progress,
        )
    chunk_count = len(chunks)
    progress(4 + vector_batch_units, "写入向量库", f"已写入 {reindexed_chunk_count} 个 Chroma 向量片段")
    category_payload = save_kb_categories(documents, chunks)
    progress(5 + vector_batch_units, "生成分类", f"已生成 {len(category_payload['items'])} 个知识库分类")
    kb_graph = build_and_save_kb_graph(category_payload, documents, chunks)
    graph_index_stats = index_graph_vectors_incremental(kb_graph.get("nodes", []), kb_graph.get("edges", []))
    progress(
        6 + vector_batch_units,
        "生成图谱索引",
        f"已写入 {graph_index_stats['graph_entity_index_count']} 个实体向量、{graph_index_stats['graph_relationship_index_count']} 个关系向量",
    )
    knowledge_base_name, knowledge_base_name_source = ensure_knowledge_base_name(category_payload, documents, chunk_count)
    graph_schema_suggestion = generate_graph_schema_suggestion(category_payload, documents, chunks)
    progress(7 + vector_batch_units, "生成知识图谱", "知识库重建完成")
    doc_status.save_manifest({"version": 1, "documents": records})
    return {
        "status": "success",
        "incremental": True,
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


def _can_reuse_snapshot(previous: dict[str, Any] | None, file_hash: str, fingerprint: str) -> bool:
    return bool(
        previous
        and previous.get("status") == doc_status.PROCESSED
        and previous.get("file_hash") == file_hash
        and previous.get("pipeline_fingerprint") == fingerprint
        and previous.get("chunk_ids")
    )


def _can_reuse_duplicate(previous: dict[str, Any] | None, file_hash: str, fingerprint: str) -> bool:
    return bool(
        previous
        and previous.get("status") == doc_status.DUPLICATE
        and previous.get("file_hash") == file_hash
        and previous.get("pipeline_fingerprint") == fingerprint
    )


def _snapshot_payload(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = snapshot.get("document")
    chunks = snapshot.get("chunks")
    return (document if isinstance(document, dict) else {}, chunks if isinstance(chunks, list) else [])


def _ordered_values(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [items[key] for key in sorted(items)]


def _unchanged_content_index(
    files: list[Path],
    file_hashes: dict[Path, str],
    records: dict[str, Any],
    fingerprint: str,
) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in files:
        doc_id = doc_status.document_id_for_path(path)
        record = records.get(doc_id)
        if (
            record
            and record.get("status") == doc_status.PROCESSED
            and record.get("file_hash") == file_hashes[path]
            and record.get("pipeline_fingerprint") == fingerprint
            and record.get("content_hash")
        ):
            index[str(record["content_hash"])] = doc_id
    return index


if __name__ == "__main__":
    import json

    print(json.dumps(ingest_knowledge_base(), ensure_ascii=False, indent=2))
