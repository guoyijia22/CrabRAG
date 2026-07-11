from __future__ import annotations

import hashlib
import json
from typing import Any

from chromadb.utils.batch_utils import create_batches

from services.rag_api.config import PROJECT_DIR, get_settings
from services.rag_api.document import doc_status
from services.rag_api.exceptions import IndexCollectionUnavailable
from services.rag_api import index_generation
from services.rag_api.security import current_retrieval_context
from services.rag_api.llm.siliconflow_client import embed_texts
from services.rag_api.vector.chroma_store import EMBEDDING_BATCH_SIZE
from services.rag_api.vector.chroma_store import _get_chroma_client
from services.rag_api.vector.embedding_reuse import iter_matching_embeddings, update_embedding_dimension

ENTITY_SUFFIX = "_graph_entities"
RELATIONSHIP_SUFFIX = "_graph_relationships"
GRAPH_VECTOR_MANIFEST_PATH = PROJECT_DIR / "data" / "ingest" / "graph_vector_manifest.json"


def index_graph_vectors(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    entity_chunks = [record for node in nodes if _node_id(node) for record in _entity_records(node)]
    relationship_chunks = [_relationship_record(edge) for edge in edges if _edge_source(edge) and _edge_target(edge)]
    _reset_and_add(entity_collection_name(), entity_chunks)
    _reset_and_add(relationship_collection_name(), relationship_chunks)
    return {
        "graph_entity_index_count": len(entity_chunks),
        "graph_relationship_index_count": len(relationship_chunks),
    }


def index_graph_vectors_incremental(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    entity_records = [record for node in nodes if _node_id(node) for record in _entity_records(node)]
    relationship_records = [_relationship_record(edge) for edge in edges if _edge_source(edge) and _edge_target(edge)]
    manifest = _load_manifest()
    manifest["entities"] = _sync_collection(entity_collection_name(), entity_records, manifest.get("entities", {}))
    manifest["relationships"] = _sync_collection(
        relationship_collection_name(),
        relationship_records,
        manifest.get("relationships", {}),
    )
    _save_manifest(manifest)
    return {
        "graph_entity_index_count": len(entity_records),
        "graph_relationship_index_count": len(relationship_records),
    }


def index_graph_vectors_generation(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    generation_id: str,
    *,
    full_rebuild: bool = False,
) -> dict[str, int]:
    entity_records = [record for node in nodes if _node_id(node) for record in _entity_records(node)]
    relationship_records = [_relationship_record(edge) for edge in edges if _edge_source(edge) and _edge_target(edge)]
    entity_name = index_generation.generation_collection_name(_base_collection_name(), generation_id, "graph_entity")
    relationship_name = index_generation.generation_collection_name(_base_collection_name(), generation_id, "graph_relationship")
    entity_stats = _build_generation_collection(
        entity_name, entity_records, generation_id, "graph_entity", full_rebuild=full_rebuild
    )
    relationship_stats = _build_generation_collection(
        relationship_name, relationship_records, generation_id, "graph_relationship", full_rebuild=full_rebuild
    )
    graph_dimensions = {
        int(stats.get("embedding_dimension") or 0)
        for stats in (entity_stats, relationship_stats)
        if int(stats.get("embedding_dimension") or 0) > 0
    }
    if len(graph_dimensions) > 1:
        raise RuntimeError(f"图谱 embedding 维度不一致：{sorted(graph_dimensions)}")
    return {
        "graph_entity_index_count": len(entity_records),
        "graph_relationship_index_count": len(relationship_records),
        "graph_reused_embedding_count": entity_stats["reused"] + relationship_stats["reused"],
        "graph_embedded_record_count": entity_stats["embedded"] + relationship_stats["embedded"],
        "graph_embedding_dimension": next(iter(graph_dimensions), 0),
    }


def search_graph_entities(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    return _search_collection(entity_collection_name(), query, top_k)


def search_graph_relationships(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    return _search_collection(relationship_collection_name(), query, top_k)


def entity_collection_name() -> str:
    context = current_retrieval_context()
    generation_id = context.generation_id if context and context.generation_id != "legacy" else index_generation.active_generation_id()
    if generation_id:
        return index_generation.generation_collection_name(_base_collection_name(), generation_id, "graph_entity")
    return f"{_base_collection_name()}{ENTITY_SUFFIX}"


def relationship_collection_name() -> str:
    context = current_retrieval_context()
    generation_id = context.generation_id if context and context.generation_id != "legacy" else index_generation.active_generation_id()
    if generation_id:
        return index_generation.generation_collection_name(_base_collection_name(), generation_id, "graph_relationship")
    return f"{_base_collection_name()}{RELATIONSHIP_SUFFIX}"


def _base_collection_name() -> str:
    return get_settings().collection_name


def _reset_and_add(collection_name: str, records: list[dict[str, Any]]) -> None:
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if not records:
        return
    documents = [record["document"] for record in records]
    embeddings = _embed_documents(documents)
    ids = [record["id"] for record in records]
    metadatas = [record["metadata"] for record in records]
    for batch_ids, batch_embeddings, batch_metadatas, batch_documents in create_batches(
        client,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    ):
        collection.add(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)


def _build_generation_collection(
    collection_name: str,
    records: list[dict[str, Any]],
    generation_id: str,
    kind: str,
    *,
    full_rebuild: bool = False,
) -> dict[str, int]:
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    embedding_fingerprint = doc_status.embedding_fingerprint(get_settings())
    pending_by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        record["metadata"] = {**record["metadata"], "embedding_fingerprint": embedding_fingerprint}
        record_hash = _record_hash(record)
        record["metadata"] = {**record["metadata"], "record_hash": record_hash, "generation_id": generation_id}
        pending_by_hash.setdefault(record_hash, []).append(record)

    write_ids: list[str] = []
    write_documents: list[str] = []
    write_embeddings: list[list[float]] = []
    write_metadatas: list[dict[str, Any]] = []
    write_batch_size = max(1, min(512, int(client.get_max_batch_size())))
    embedding_dimension = 0

    def flush() -> None:
        if not write_ids:
            return
        collection.upsert(
            ids=list(write_ids),
            documents=list(write_documents),
            embeddings=list(write_embeddings),
            metadatas=list(write_metadatas),
        )
        write_ids.clear()
        write_documents.clear()
        write_embeddings.clear()
        write_metadatas.clear()

    def queue(group: list[dict[str, Any]], embedding: list[float]) -> None:
        nonlocal embedding_dimension
        embedding_dimension = update_embedding_dimension(embedding_dimension, embedding)
        for record in group:
            write_ids.append(str(record["id"]))
            write_documents.append(str(record["document"]))
            write_embeddings.append(list(embedding))
            write_metadatas.append(record["metadata"])
            if len(write_ids) >= write_batch_size:
                flush()

    source_names = [] if full_rebuild else _generation_source_names(kind)

    def reusable_key(document: str, metadata: dict[str, Any]) -> str | None:
        if str(metadata.get("embedding_fingerprint") or "") != embedding_fingerprint:
            return None
        return str(metadata.get("record_hash") or _record_hash({"document": document, "metadata": metadata}))

    for record_hash, embedding in iter_matching_embeddings(
        client,
        source_names,
        set(pending_by_hash),
        reusable_key,
    ):
        queue(pending_by_hash.pop(str(record_hash)), embedding)

    missing_groups = list(pending_by_hash.values())
    for start in range(0, len(missing_groups), EMBEDDING_BATCH_SIZE):
        groups = missing_groups[start : start + EMBEDDING_BATCH_SIZE]
        generated = embed_texts([group[0]["document"] for group in groups])
        for group, embedding in zip(groups, generated):
            queue(group, embedding)
    flush()
    collection.modify(
        metadata={
            "embedding_fingerprint": embedding_fingerprint,
            "embedding_dimension": embedding_dimension,
        }
    )
    if collection.count() != len(records):
        raise RuntimeError(f"图谱索引代写入数量不一致：expected={len(records)}, actual={collection.count()}")
    embedded_record_count = sum(len(group) for group in missing_groups)
    return {
        "reused": len(records) - embedded_record_count,
        "embedded": embedded_record_count,
        "embedding_dimension": embedding_dimension,
    }


def _generation_source_names(kind: str) -> list[str]:
    state = index_generation.load_index_state()
    generation_ids = [
        str(value)
        for value in (state.get("active_generation"), state.get("previous_generation"))
        if value
    ]
    if generation_ids:
        return [
            index_generation.generation_collection_name(_base_collection_name(), generation_id, kind)
            for generation_id in generation_ids
        ]
    if kind == "graph_entity":
        return [f"{_base_collection_name()}{ENTITY_SUFFIX}"]
    return [f"{_base_collection_name()}{RELATIONSHIP_SUFFIX}"]


def _sync_collection(collection_name: str, records: list[dict[str, Any]], previous: dict[str, Any]) -> dict[str, str]:
    previous_hashes = {str(key): str(value) for key, value in (previous or {}).items()}
    current_hashes = {record["id"]: _record_hash(record) for record in records}
    removed_ids = sorted(set(previous_hashes) - set(current_hashes))
    changed_records = [record for record in records if previous_hashes.get(record["id"]) != current_hashes[record["id"]]]
    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if removed_ids:
        collection.delete(ids=removed_ids)
    if changed_records:
        _upsert_records(client, collection, changed_records)
    return current_hashes


def _upsert_records(client, collection, records: list[dict[str, Any]]) -> None:
    documents = [record["document"] for record in records]
    embeddings = _embed_documents(documents)
    ids = [record["id"] for record in records]
    metadatas = [record["metadata"] for record in records]
    for batch_ids, batch_embeddings, batch_metadatas, batch_documents in create_batches(
        client,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    ):
        collection.upsert(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)


def _search_collection(collection_name: str, query: str, top_k: int) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    context = current_retrieval_context()
    allowed_document_ids = context.allowed_document_ids if context else None
    if allowed_document_ids is not None and not allowed_document_ids:
        return []
    client = _get_chroma_client()
    governed_read = (
        context is not None and context.generation_id != "legacy"
    ) or (
        context is None and bool(index_generation.active_generation_id())
    )
    try:
        if governed_read:
            collection = client.get_collection(name=collection_name)
        else:
            collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    except Exception as exc:  # noqa: BLE001
        raise IndexCollectionUnavailable(f"活动索引集合不可用：{collection_name}") from exc
    if collection.count() == 0:
        return []
    query_embedding = embed_texts([query])[0]
    metadata = getattr(collection, "metadata", None) or {}
    expected_dimension = int(metadata.get("embedding_dimension") or 0)
    if expected_dimension and len(query_embedding) != expected_dimension:
        raise RuntimeError(
            f"向量模型维度已变化：index={expected_dimension}, query={len(query_embedding)}；请执行全量重建"
        )
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if allowed_document_ids is not None:
        query_kwargs["where"] = {"document_id": {"$in": sorted(allowed_document_ids)}}
    result = collection.query(**query_kwargs)
    hits: list[dict[str, Any]] = []
    for document, metadata, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
        if allowed_document_ids is not None and str((metadata or {}).get("document_id") or "") not in allowed_document_ids:
            continue
        score = round(max(0.0, 1.0 - float(distance)), 4)
        hits.append({**_normalize_metadata(metadata or {}), "document": document, "score": score})
    return hits


def _entity_records(node: dict[str, Any]) -> list[dict[str, Any]]:
    document_ids = sorted({str(item) for item in node.get("document_ids", []) or [] if item})
    return [_entity_record(node, document_id) for document_id in document_ids] if document_ids else [_entity_record(node, "")]


def _entity_record(node: dict[str, Any], document_id: str = "") -> dict[str, Any]:
    node_id = _node_id(node)
    label = str(node.get("label") or node_id)
    node_type = str(node.get("type") or "")
    category = str(node.get("category") or "")
    source_files = [str(item) for item in node.get("source_files", []) or [] if item]
    visible_source_files = source_files if len(node.get("document_ids", []) or []) <= 1 else []
    document = "\n".join(
        [
            f"实体：{label}",
            f"类型：{node_type}",
            f"分类：{category}",
            f"来源文件：{'、'.join(visible_source_files)}",
        ]
    )
    return {
        "id": f"entity::{node_id}::doc::{document_id}" if document_id else f"entity::{node_id}",
        "document": document,
        "metadata": _clean_metadata(
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "category": category,
                "source_files": visible_source_files,
                "document_id": document_id,
            }
        ),
    }


def _relationship_record(edge: dict[str, Any]) -> dict[str, Any]:
    source = _edge_source(edge)
    target = _edge_target(edge)
    label = str(edge.get("label") or edge.get("relation") or "")
    description = str(edge.get("description") or "")
    evidence = str(edge.get("evidence") or "")
    source_file = str(edge.get("source_file") or "")
    edge_id = str(edge.get("id") or f"{source}->{label}->{target}")
    document = "\n".join(
        [
            f"关系：{source} {label} {target}",
            f"描述：{description}",
            f"证据：{evidence}",
            f"来源文件：{source_file}",
        ]
    )
    return {
        "id": f"relationship::{edge_id}",
        "document": document,
        "metadata": _clean_metadata(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "label": label,
                "description": description,
                "evidence": evidence,
                "source_file": source_file,
                "document_id": str(edge.get("document_id") or ""),
                "confidence": edge.get("confidence", 0.8),
            }
        ),
    }


def _embed_documents(documents: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        embeddings.extend(embed_texts(documents[start : start + EMBEDDING_BATCH_SIZE]))
    return embeddings


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False)
    return cleaned


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    source_files = result.get("source_files")
    if isinstance(source_files, str):
        try:
            parsed = json.loads(source_files)
            result["source_files"] = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            result["source_files"] = [source_files] if source_files else []
    return result


def _load_manifest() -> dict[str, Any]:
    if not GRAPH_VECTOR_MANIFEST_PATH.exists():
        return {"entities": {}, "relationships": {}}
    try:
        payload = json.loads(GRAPH_VECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"entities": {}, "relationships": {}}
    if not isinstance(payload, dict):
        return {"entities": {}, "relationships": {}}
    return {
        "entities": payload.get("entities", {}) if isinstance(payload.get("entities"), dict) else {},
        "relationships": payload.get("relationships", {}) if isinstance(payload.get("relationships"), dict) else {},
    }


def _save_manifest(manifest: dict[str, Any]) -> None:
    GRAPH_VECTOR_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_VECTOR_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_hash(record: dict[str, Any]) -> str:
    metadata = {
        key: value
        for key, value in (record.get("metadata", {}) or {}).items()
        if key not in {"record_hash", "generation_id"}
    }
    payload = {"document": record.get("document", ""), "metadata": metadata}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("id") or node.get("label") or "").strip()


def _edge_source(edge: dict[str, Any]) -> str:
    return str(edge.get("source") or edge.get("from") or "").strip()


def _edge_target(edge: dict[str, Any]) -> str:
    return str(edge.get("target") or edge.get("to") or "").strip()
