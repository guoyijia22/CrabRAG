from __future__ import annotations

import json
from typing import Any

from chromadb.utils.batch_utils import create_batches

from services.rag_api.config import get_settings
from services.rag_api.llm.siliconflow_client import embed_texts
from services.rag_api.vector.chroma_store import EMBEDDING_BATCH_SIZE
from services.rag_api.vector.chroma_store import _get_chroma_client

ENTITY_SUFFIX = "_graph_entities"
RELATIONSHIP_SUFFIX = "_graph_relationships"


def index_graph_vectors(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    entity_chunks = [_entity_record(node) for node in nodes if _node_id(node)]
    relationship_chunks = [_relationship_record(edge) for edge in edges if _edge_source(edge) and _edge_target(edge)]
    _reset_and_add(entity_collection_name(), entity_chunks)
    _reset_and_add(relationship_collection_name(), relationship_chunks)
    return {
        "graph_entity_index_count": len(entity_chunks),
        "graph_relationship_index_count": len(relationship_chunks),
    }


def search_graph_entities(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    return _search_collection(entity_collection_name(), query, top_k)


def search_graph_relationships(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    return _search_collection(relationship_collection_name(), query, top_k)


def entity_collection_name() -> str:
    return f"{_base_collection_name()}{ENTITY_SUFFIX}"


def relationship_collection_name() -> str:
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


def _search_collection(collection_name: str, query: str, top_k: int) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        return []
    query_embedding = embed_texts([query])[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
    hits: list[dict[str, Any]] = []
    for document, metadata, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
        score = round(max(0.0, 1.0 - float(distance)), 4)
        hits.append({**_normalize_metadata(metadata or {}), "document": document, "score": score})
    return hits


def _entity_record(node: dict[str, Any]) -> dict[str, Any]:
    node_id = _node_id(node)
    label = str(node.get("label") or node_id)
    node_type = str(node.get("type") or "")
    category = str(node.get("category") or "")
    source_files = [str(item) for item in node.get("source_files", []) or [] if item]
    document = "\n".join(
        [
            f"实体：{label}",
            f"类型：{node_type}",
            f"分类：{category}",
            f"来源文件：{'、'.join(source_files)}",
        ]
    )
    return {
        "id": f"entity::{node_id}",
        "document": document,
        "metadata": _clean_metadata(
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "category": category,
                "source_files": source_files,
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


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("id") or node.get("label") or "").strip()


def _edge_source(edge: dict[str, Any]) -> str:
    return str(edge.get("source") or edge.get("from") or "").strip()


def _edge_target(edge: dict[str, Any]) -> str:
    return str(edge.get("target") or edge.get("to") or "").strip()
