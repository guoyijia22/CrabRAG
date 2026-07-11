from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable
import hashlib
from math import ceil
import re
from typing import Any

import chromadb
from chromadb.utils.batch_utils import create_batches

from services.rag_api.config import get_settings
from services.rag_api.document.categories import source_files_for_category
from services.rag_api.llm.siliconflow_client import embed_texts
from services.rag_api.rag_settings import load_rag_settings
from services.rag_api import index_generation
from services.rag_api.security import current_retrieval_context

EMBEDDING_BATCH_SIZE = 64
_COLLECTION_OVERRIDE: ContextVar[str | None] = ContextVar("collection_name_override", default=None)
ProgressCallback = Callable[[dict[str, Any]], None]


def get_collection():
    client = _get_chroma_client()
    return client.get_or_create_collection(name=_collection_name(), metadata={"hnsw:space": "cosine"})


def reset_collection():
    client = _get_chroma_client()
    try:
        client.delete_collection(_collection_name())
    except Exception:
        pass
    return client.get_or_create_collection(name=_collection_name(), metadata={"hnsw:space": "cosine"})


def add_chunks(chunks: list[dict], progress_callback: ProgressCallback | None = None) -> int:
    if not chunks:
        return 0
    documents = [chunk["content"] for chunk in chunks]
    embeddings = _embed_in_batches(documents, progress_callback=progress_callback)
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    client = _get_chroma_client()
    collection = reset_collection()
    for batch_ids, batch_embeddings, batch_metadatas, batch_documents in create_batches(
        client,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    ):
        collection.add(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)
    return len(chunks)


def upsert_chunks_incremental(
    chunks: list[dict],
    delete_chunk_ids: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> int:
    collection = get_collection()
    if delete_chunk_ids:
        collection.delete(ids=delete_chunk_ids)
    if not chunks:
        return 0
    documents = [chunk["content"] for chunk in chunks]
    embeddings = _embed_in_batches(documents, progress_callback=progress_callback)
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    client = _get_chroma_client()
    for batch_ids, batch_embeddings, batch_metadatas, batch_documents in create_batches(
        client,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    ):
        collection.upsert(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)
    return len(chunks)


def build_generation_chunks(
    chunks: list[dict],
    generation_id: str,
    *,
    full_rebuild: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    settings = get_settings()
    client = _get_chroma_client()
    target_name = index_generation.generation_collection_name(settings.collection_name, generation_id, "text")
    try:
        client.delete_collection(target_name)
    except Exception:
        pass
    target = client.get_or_create_collection(name=target_name, metadata={"hnsw:space": "cosine"})
    reusable = {} if full_rebuild else _load_reusable_embeddings(client, settings.collection_name)
    embeddings: list[list[float] | None] = [None] * len(chunks)
    missing_indexes: list[int] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.setdefault("metadata", {})
        metadata["generation_id"] = generation_id
        key = _embedding_reuse_key(chunk["content"], metadata)
        embedding = reusable.get(key)
        if embedding is None:
            embedding = reusable.get((key[0], key[1], None))
        if embedding is None:
            missing_indexes.append(index)
        else:
            embeddings[index] = list(embedding)

    if missing_indexes:
        missing_documents = [chunks[index]["content"] for index in missing_indexes]
        generated = _embed_in_batches(missing_documents, progress_callback=progress_callback)
        for index, embedding in zip(missing_indexes, generated):
            embeddings[index] = embedding

    if chunks:
        final_embeddings = [list(embedding or []) for embedding in embeddings]
        for batch_ids, batch_embeddings, batch_metadatas, batch_documents in create_batches(
            client,
            ids=[chunk["id"] for chunk in chunks],
            embeddings=final_embeddings,
            metadatas=[chunk["metadata"] for chunk in chunks],
            documents=[chunk["content"] for chunk in chunks],
        ):
            target.upsert(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)
    if target.count() != len(chunks):
        raise RuntimeError(f"索引代写入数量不一致：expected={len(chunks)}, actual={target.count()}")
    return {
        "chunk_count": len(chunks),
        "reused_embedding_count": len(chunks) - len(missing_indexes),
        "embedded_chunk_count": len(missing_indexes),
    }


def _get_chroma_client():
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def embedding_batch_count(document_count: int) -> int:
    if document_count <= 0:
        return 0
    return ceil(document_count / EMBEDDING_BATCH_SIZE)


def _embed_in_batches(documents: list[str], progress_callback: ProgressCallback | None = None) -> list[list[float]]:
    embeddings: list[list[float]] = []
    total = len(documents)
    total_batches = embedding_batch_count(total)
    provider = get_settings().embedding_provider
    step = "本地向量化" if provider == "local_onnx" else "向量化"
    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[start : start + EMBEDDING_BATCH_SIZE]
        embeddings.extend(embed_texts(batch))
        if progress_callback:
            current_batch = start // EMBEDDING_BATCH_SIZE + 1
            processed = min(start + len(batch), total)
            label = "本地向量化中" if provider == "local_onnx" else "向量化中"
            progress_callback(
                {
                    "current_step": step,
                    "message": f"{label}：第 {current_batch} / {total_batches} 批，已处理 {processed} / {total} 个片段",
                    "detail_current": current_batch,
                    "detail_total": total_batches,
                    "detail_processed": processed,
                    "detail_size": total,
                }
            )
    return embeddings


def collection_status() -> dict[str, Any]:
    collection = get_collection()
    return {"collection": _collection_name(), "count": collection.count(), "path": str(get_settings().chroma_dir)}


@contextmanager
def override_collection_name(collection_name: str | None):
    token = _COLLECTION_OVERRIDE.set(collection_name)
    try:
        yield
    finally:
        _COLLECTION_OVERRIDE.reset(token)


def _collection_name() -> str:
    override = _COLLECTION_OVERRIDE.get()
    if override:
        return override
    base = get_settings().collection_name
    context = current_retrieval_context()
    generation_id = context.generation_id if context and context.generation_id != "legacy" else index_generation.active_generation_id()
    return index_generation.generation_collection_name(base, generation_id, "text") if generation_id else base


def _load_reusable_embeddings(client, base_collection_name: str) -> dict[tuple[str, str, str | None], list[float]]:
    active_generation = index_generation.active_generation_id()
    source_name = (
        index_generation.generation_collection_name(base_collection_name, active_generation, "text")
        if active_generation
        else base_collection_name
    )
    try:
        source = client.get_or_create_collection(name=source_name, metadata={"hnsw:space": "cosine"})
        result = source.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        return {}
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    raw_embeddings = result.get("embeddings")
    embeddings = list(raw_embeddings) if raw_embeddings is not None else []
    reusable: dict[tuple[str, str, str | None], list[float]] = {}
    for document, metadata, embedding in zip(documents, metadatas, embeddings):
        key = _embedding_reuse_key(str(document or ""), metadata or {})
        reusable.setdefault(key, [float(value) for value in embedding])
    return reusable


def _embedding_reuse_key(document: str, metadata: dict[str, Any]) -> tuple[str, str, str | None]:
    normalized = re.sub(r"\s+", " ", document).strip()
    chunk_hash = str(metadata.get("chunk_hash") or hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    granularity = str(metadata.get("granularity") or "chunk")
    fingerprint = str(metadata["embedding_fingerprint"]) if metadata.get("embedding_fingerprint") else None
    return chunk_hash, granularity, fingerprint


def search_chunks(query: str, intent: str, entities: list[str] | None = None, top_k: int = 2, min_score: float | None = None, candidate_k: int | None = None) -> list[dict]:
    rag_settings = load_rag_settings()
    min_score = rag_settings.min_score if min_score is None else min_score
    candidate_k = max(rag_settings.vector_candidate_k, top_k * 4) if candidate_k is None else candidate_k
    entities = entities or []
    allowed_document_ids = _allowed_document_ids()
    if allowed_document_ids is not None and not allowed_document_ids:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []
    query_embedding = embed_texts([query])[0]
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": max(candidate_k, top_k),
        "include": ["documents", "metadatas", "distances"],
    }
    if allowed_document_ids is not None:
        query_kwargs["where"] = {"document_id": {"$in": sorted(allowed_document_ids)}}
    result = collection.query(**query_kwargs)
    candidates: list[dict] = []
    for doc, meta, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
        if not _metadata_allowed(meta or {}, allowed_document_ids):
            continue
        vector_score = max(0.0, 1.0 - float(distance))
        intent_score = 1.0 if meta.get("category") == intent else 0.0
        entity_score = _entity_match_score(doc, entities)
        source_priority_score = _source_priority_score(meta.get("source_file", ""), intent)
        final_score = 0.65 * vector_score + 0.20 * intent_score + 0.10 * entity_score + 0.05 * source_priority_score
        if final_score >= min_score:
            candidates.append(_chunk_payload(doc, meta, final_score, "vector"))
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:candidate_k]


def search_all_chunks() -> list[dict]:
    allowed_document_ids = _allowed_document_ids()
    if allowed_document_ids is not None and not allowed_document_ids:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []
    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if allowed_document_ids is not None:
        get_kwargs["where"] = {"document_id": {"$in": sorted(allowed_document_ids)}}
    result = collection.get(**get_kwargs)
    return [_chunk_payload(doc, meta, 0.0, "all") for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))]


def search_chunks_by_keywords(query: str, intent: str, entities: list[str] | None = None, top_k: int = 2) -> list[dict]:
    chunks = search_all_chunks()
    entities = entities or []
    query_terms = set(entities + [intent])
    query_terms.update(term for term in ["欠费", "地址迁移", "带宽变更", "一票否决", "中断", "报修", "销户", "资费", "材料", "审核"] if term in query)
    candidates: list[dict] = []
    for chunk in chunks:
        doc = chunk.get("content", "")
        meta_category = chunk.get("category", "")
        score = 0.45
        score += 0.2 if meta_category == intent else 0
        score += min(0.3, sum(1 for term in query_terms if term and term in doc) * 0.08)
        score += _source_priority_score(chunk.get("source_file", ""), intent) * 0.05
        if score >= 0.35:
            candidates.append({**chunk, "score": round(score, 4), "retrieval_channel": "keyword"})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def _chunk_payload(doc: str, meta: dict, score: float, channel: str) -> dict:
    context = current_retrieval_context()
    return {
        "content": doc,
        "source_file": meta.get("source_file", ""),
        "source_path": meta.get("source_path", ""),
        "category": meta.get("category", ""),
        "section_title": meta.get("section_title", ""),
        "granularity": meta.get("granularity", "chunk"),
        "parent_chunk_id": meta.get("parent_chunk_id", ""),
        "document_id": meta.get("document_id", meta.get("doc_id", "")),
        "document_version": meta.get("document_version", ""),
        "effective_at": meta.get("effective_at", ""),
        "updated_at": meta.get("updated_at", ""),
        "acl_revision": meta.get("acl_revision", ""),
        "index_generation": context.generation_id if context else (meta.get("generation_id", "") or "legacy"),
        "score": round(score, 4),
        "retrieval_channel": channel,
    }


def _allowed_document_ids() -> frozenset[str] | None:
    context = current_retrieval_context()
    return context.allowed_document_ids if context else None


def _metadata_allowed(metadata: dict[str, Any], allowed_document_ids: frozenset[str] | None) -> bool:
    if allowed_document_ids is None:
        return True
    document_id = str(metadata.get("document_id") or metadata.get("doc_id") or "")
    return document_id in allowed_document_ids


def _entity_match_score(text: str, entities: list[str]) -> float:
    if not entities:
        return 0.0
    return min(1.0, sum(1 for entity in entities if entity in text) / max(1, len(entities)))


def _source_priority_score(source_file: str, intent: str) -> float:
    if not source_file or not intent:
        return 0.0
    return 1.0 if source_file in source_files_for_category(intent) else 0.0
