from __future__ import annotations

import hashlib
import json
from typing import Any

from chromadb.utils.batch_utils import create_batches

from services.rag_api.config import PROJECT_DIR, get_settings
from services.rag_api import index_generation
from services.rag_api.security import current_retrieval_context
from services.rag_api.llm.siliconflow_client import embed_texts
from services.rag_api.vector.chroma_store import EMBEDDING_BATCH_SIZE
from services.rag_api.vector.chroma_store import _get_chroma_client

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


def index_graph_vectors_generation(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], generation_id: str) -> dict[str, int]:
    entity_records = [record for node in nodes if _node_id(node) for record in _entity_records(node)]
    relationship_records = [_relationship_record(edge) for edge in edges if _edge_source(edge) and _edge_target(edge)]
    entity_name = index_generation.generation_collection_name(_base_collection_name(), generation_id, "graph_entity")
    relationship_name = index_generation.generation_collection_name(_base_collection_name(), generation_id, "graph_relationship")
    entity_stats = _build_generation_collection(entity_name, entity_records, generation_id, "graph_entity")
    relationship_stats = _build_generation_collection(relationship_name, relationship_records, generation_id, "graph_relationship")
    return {
        "graph_entity_index_count": len(entity_records),
        "graph_relationship_index_count": len(relationship_records),
        "graph_reused_embedding_count": entity_stats["reused"] + relationship_stats["reused"],
        "graph_embedded_record_count": entity_stats["embedded"] + relationship_stats["embedded"],
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
) -> dict[str, int]:
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    reusable = _load_generation_embeddings(client, kind)
    missing: list[dict[str, Any]] = []
    embeddings_by_id: dict[str, list[float]] = {}
    for record in records:
        record_hash = _record_hash(record)
        record["metadata"] = {**record["metadata"], "record_hash": record_hash, "generation_id": generation_id}
        embedding = reusable.get(record_hash)
        if embedding is None:
            missing.append(record)
        else:
            embeddings_by_id[record["id"]] = embedding
    if missing:
        for record, embedding in zip(missing, _embed_documents([record["document"] for record in missing])):
            embeddings_by_id[record["id"]] = embedding
    if records:
        for batch_ids, batch_embeddings, batch_metadatas, batch_documents in create_batches(
            client,
            ids=[record["id"] for record in records],
            embeddings=[embeddings_by_id[record["id"]] for record in records],
            metadatas=[record["metadata"] for record in records],
            documents=[record["document"] for record in records],
        ):
            collection.upsert(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)
    if collection.count() != len(records):
        raise RuntimeError(f"图谱索引代写入数量不一致：expected={len(records)}, actual={collection.count()}")
    return {"reused": len(records) - len(missing), "embedded": len(missing)}


def _load_generation_embeddings(client, kind: str) -> dict[str, list[float]]:
    active_generation = index_generation.active_generation_id()
    if active_generation:
        source_name = index_generation.generation_collection_name(_base_collection_name(), active_generation, kind)
    elif kind == "graph_entity":
        source_name = f"{_base_collection_name()}{ENTITY_SUFFIX}"
    else:
        source_name = f"{_base_collection_name()}{RELATIONSHIP_SUFFIX}"
    try:
        source = client.get_or_create_collection(name=source_name, metadata={"hnsw:space": "cosine"})
        result = source.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        return {}
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    raw_embeddings = result.get("embeddings")
    embeddings = list(raw_embeddings) if raw_embeddings is not None else []
    reusable: dict[str, list[float]] = {}
    for document, metadata, embedding in zip(documents, metadatas, embeddings):
        metadata = metadata or {}
        record_hash = str(metadata.get("record_hash") or _record_hash({"document": document, "metadata": metadata}))
        reusable.setdefault(record_hash, [float(value) for value in embedding])
    return reusable


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
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        return []
    query_embedding = embed_texts([query])[0]
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
