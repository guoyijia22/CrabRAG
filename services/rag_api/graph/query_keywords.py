from __future__ import annotations

import re
from typing import Any

from services.rag_api.graph.entities import ENTITIES

RELATIONSHIP_HINTS = [
    "关系",
    "关联",
    "影响",
    "约束",
    "流程",
    "先后",
    "需要",
    "是否",
    "审核",
    "合规",
    "报告",
    "在哪",
    "哪里",
    "来源",
    "来源文件",
    "文件",
    "材料",
    "包含",
    "提及",
]


def split_graph_query_keywords(
    query: str,
    intent: str,
    entities: list[str] | None,
    *,
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query = (query or "").strip()
    nodes = nodes or []
    edges = edges or []
    entity_keywords = _entity_keywords(query, intent, entities or [], nodes)
    relationship_keywords = _relationship_keywords(query, intent, edges)
    fallback = False
    if not entity_keywords:
        entity_keywords = [query] if query else []
        fallback = True
    if not relationship_keywords:
        relationship_keywords = [query] if query else []
        fallback = True
    return {
        "entity_keywords": _dedupe(entity_keywords),
        "relationship_keywords": _dedupe(relationship_keywords),
        "fallback": fallback,
    }


def _entity_keywords(query: str, intent: str, entities: list[str], nodes: list[dict[str, Any]]) -> list[str]:
    keywords: list[str] = []
    for entity in entities:
        if entity and (entity in query or entity == intent):
            keywords.append(entity)
    for entity, detail in ENTITIES.items():
        aliases = detail.get("aliases", [])
        if entity in query or any(alias and alias in query for alias in aliases):
            keywords.append(entity)
    if intent:
        for node in nodes:
            node_id = str(node.get("id") or node.get("label") or "")
            if node_id and node_id == intent:
                keywords.append(node_id)
    for node in nodes:
        node_id = str(node.get("id") or node.get("label") or "").strip()
        label = str(node.get("label") or node_id).strip()
        if node_id and node_id in query:
            keywords.append(node_id)
        elif label and label in query:
            keywords.append(label)
    keywords.extend(_quoted_terms(query))
    return _dedupe(keywords)


def _relationship_keywords(query: str, intent: str, edges: list[dict[str, Any]]) -> list[str]:
    keywords: list[str] = []
    for hint in RELATIONSHIP_HINTS:
        if hint in query or hint == intent:
            keywords.append(hint)
    if "在哪" in query or "哪里" in query:
        keywords.append("来源文件")
    if intent and any(term in intent for term in ("审核", "流程", "规则", "关系", "分类")):
        keywords.append(intent)
    for edge in edges:
        label = str(edge.get("label") or edge.get("relation") or "").strip()
        description = str(edge.get("description") or edge.get("evidence") or "")
        if label and (label in query or any(hint in label for hint in keywords)):
            keywords.append(label)
        for hint in RELATIONSHIP_HINTS:
            if hint in description and hint in query:
                keywords.append(hint)
    keywords.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", query))
    return _dedupe(keywords)


def _quoted_terms(text: str) -> list[str]:
    return [term.strip() for term in re.findall(r"[“《\"']([^”》\"']{2,30})[”》\"']", text) if term.strip()]


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)
    return result
