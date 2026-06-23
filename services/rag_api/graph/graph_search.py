from __future__ import annotations

from services.rag_api.graph.entities import ENTITIES
from services.rag_api.graph.relations import RELATIONS


def extract_entities(query: str) -> list[str]:
    found: list[str] = []
    for entity, detail in ENTITIES.items():
        aliases = detail.get("aliases", [])
        if entity in query or any(alias in query for alias in aliases):
            found.append(entity)
    if "欠费" in query and "退订销户" not in found:
        found.append("退订销户")
    if ("材料" in query or "资质" in query) and ("企业客户" in query or "集团客户" in query) and "合规审核" not in found:
        found.append("合规审核")
    if ("地址迁移" in query or "带宽变更" in query) and "合规审核" not in found:
        found.append("合规审核")
    return _dedupe(found)


def find_relation_paths(entities: list[str], max_hops: int = 2) -> list[dict]:
    entity_set = set(entities)
    paths: list[dict] = []
    for rel in RELATIONS:
        if rel["from"] in entity_set and rel["to"] in entity_set:
            paths.append(_format_relation(rel))
        elif rel["from"] in entity_set or rel["to"] in entity_set:
            paths.append(_format_relation(rel))
    if max_hops >= 2:
        for first in RELATIONS:
            for second in RELATIONS:
                if first["to"] != second["from"]:
                    continue
                if first["from"] in entity_set and second["to"] in entity_set:
                    paths.append(
                        {
                            "path": f"{first['from']} -> {first['relation']} -> {first['to']} -> {second['relation']} -> {second['to']}",
                            "description": f"{first['description']} {second['description']}",
                        }
                    )
    return _dedupe_paths(paths)


def graph_relation_search(query: str, intent: str, top_k: int = 2) -> dict:
    entities = extract_entities(query)
    if intent and intent not in entities:
        entities.append(intent)
    relation_paths = find_relation_paths(entities)
    try:
        from services.rag_api.vector.chroma_store import search_chunks_by_keywords

        chunks = search_chunks_by_keywords(query, intent, entities, top_k=top_k)
    except Exception:
        chunks = []
    return {"entities": entities, "relation_paths": relation_paths[:top_k], "chunks": chunks[:top_k]}


def _format_relation(rel: dict) -> dict:
    return {"path": f"{rel['from']} -> {rel['relation']} -> {rel['to']}", "description": rel["description"]}


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _dedupe_paths(paths: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for item in paths:
        if item["path"] not in seen:
            seen.add(item["path"])
            result.append(item)
    return result
