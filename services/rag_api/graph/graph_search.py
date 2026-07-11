from __future__ import annotations

import json
import re
from typing import Any

from services.rag_api import index_generation
from services.rag_api.agent.prompts import build_keyword_extraction_prompt, detect_prompt_language
from services.rag_api.graph.entities import ENTITIES
from services.rag_api.graph.graph_vector_store import search_graph_entities, search_graph_relationships
from services.rag_api.graph.graph_store import KB_GRAPH_PATH as DEFAULT_KB_GRAPH_PATH
from services.rag_api.graph.graph_store import load_raw_graph
from services.rag_api.graph.query_keywords import split_graph_query_keywords
from services.rag_api.exceptions import IndexCollectionUnavailable
from services.rag_api.graph.relations import RELATIONS
from services.rag_api.llm.siliconflow_client import chat_completion
from services.rag_api.security import current_retrieval_context, filter_graph_by_permission

KB_GRAPH_PATH = DEFAULT_KB_GRAPH_PATH
COMMON_QUERY_TERMS = [
    "欠费",
    "地址迁移",
    "带宽变更",
    "一票否决",
    "中断",
    "报修",
    "销户",
    "资费",
    "材料",
    "审核",
    "合规",
    "合同",
    "资质",
    "企业客户",
    "个人客户",
    "政企专线",
    "业务变更",
]


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
    dynamic_result = _dynamic_graph_relation_search(query, intent, top_k=top_k)
    if dynamic_result is not None:
        return dynamic_result
    context = current_retrieval_context()
    if context is not None and context.generation_id != "legacy":
        return _governed_text_fallback(query, intent, top_k)
    if context is None and index_generation.active_generation_id():
        return {"entities": [], "relation_paths": [], "chunks": []}
    return _static_graph_relation_search(query, intent, top_k=top_k)


def _governed_text_fallback(query: str, intent: str, top_k: int) -> dict:
    entities = extract_entities(query)
    if intent and intent not in entities:
        entities.append(intent)
    try:
        from services.rag_api.vector.chroma_store import search_chunks_by_keywords

        chunks = search_chunks_by_keywords(query, intent, entities, top_k=top_k)
    except Exception:
        chunks = []
    return {"entities": entities, "relation_paths": [], "chunks": chunks[:top_k]}


def extract_query_keywords_with_llm(query: str) -> dict[str, Any]:
    language = detect_prompt_language(query)
    try:
        content = chat_completion(
            [{"role": "user", "content": build_keyword_extraction_prompt(query, language=language)}],
            temperature=0.0,
            max_tokens=300,
        )
        payload = _parse_keyword_json(content)
        return {
            "high_level_keywords": _string_list(payload.get("high_level_keywords")),
            "low_level_keywords": _string_list(payload.get("low_level_keywords")),
            "fallback": False,
            "language": language,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "high_level_keywords": [],
            "low_level_keywords": [],
            "fallback": True,
            "language": language,
            "error": str(exc),
        }


def _static_graph_relation_search(query: str, intent: str, top_k: int) -> dict:
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


def _dynamic_graph_relation_search(query: str, intent: str, top_k: int) -> dict | None:
    nodes, edges, graph_source = load_raw_graph(None if KB_GRAPH_PATH == DEFAULT_KB_GRAPH_PATH else KB_GRAPH_PATH)
    nodes, edges = filter_graph_by_permission(nodes, edges)
    if graph_source != "dynamic_graph" or not edges:
        return None
    keyword_info = split_graph_query_keywords(query, intent, [], nodes=nodes, edges=edges)
    trace: list[dict[str, Any]] = [{"node": "graph_query_keywords", "output": keyword_info}]
    vector_result = _dynamic_graph_vector_search(query, intent, nodes, edges, keyword_info, top_k)
    if vector_result is not None:
        vector_result["trace"] = trace + vector_result.get("trace", [])
        return vector_result
    llm_fallback_attempted = False
    if keyword_info.get("fallback"):
        llm_fallback_attempted = True
        fallback_result = _dynamic_graph_llm_keyword_search(query, intent, nodes, edges, top_k, trace)
        if fallback_result is not None:
            return fallback_result
    terms = _dedupe(keyword_info["entity_keywords"] + keyword_info["relationship_keywords"] + _query_terms(query, intent))
    matched_nodes = _matched_dynamic_nodes(nodes, query, intent, terms)
    relation_paths: list[dict[str, Any]] = []
    for edge in edges:
        score = _score_dynamic_edge(edge, matched_nodes, terms)
        if score <= 0:
            continue
        relation_paths.append(_format_dynamic_relation(edge, score, match_source="dynamic_literal"))
    if not relation_paths:
        if not llm_fallback_attempted:
            fallback_result = _dynamic_graph_llm_keyword_search(query, intent, nodes, edges, top_k, trace)
            if fallback_result is not None:
                return fallback_result
        return None
    relation_paths.sort(key=lambda item: item.get("score", 0), reverse=True)
    relation_paths = _dedupe_paths(relation_paths)
    entities = _dynamic_entities(matched_nodes, relation_paths)
    chunks = _search_chunks_for_relation_paths(query, intent, entities, relation_paths, top_k=top_k)
    trace.append(
        {
            "node": "graph_mix_search",
            "output": {
                "match_source": "dynamic_literal",
                "local_relation_paths": 0,
                "global_relation_paths": 0,
                "literal_relation_paths": len(relation_paths),
            },
        }
    )
    return {"entities": entities, "relation_paths": relation_paths[:top_k], "chunks": chunks[:top_k], "trace": trace}


def _dynamic_graph_llm_keyword_search(
    query: str,
    intent: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    top_k: int,
    base_trace: list[dict[str, Any]],
) -> dict | None:
    keyword_result = extract_query_keywords_with_llm(query)
    trace = base_trace + [{"node": "keyword_extraction_fallback", "output": keyword_result}]
    high_keywords = keyword_result.get("high_level_keywords") or []
    low_keywords = keyword_result.get("low_level_keywords") or []
    if keyword_result.get("fallback") or not high_keywords + low_keywords:
        return None
    keyword_info = {
        "entity_keywords": _dedupe(_string_list(low_keywords) or [query]),
        "relationship_keywords": _dedupe(_string_list(high_keywords) or [query]),
        "fallback": False,
        "source": "llm_keyword_extraction",
    }
    vector_result = _dynamic_graph_vector_search(query, intent, nodes, edges, keyword_info, top_k)
    if vector_result is not None:
        vector_result["trace"] = trace + vector_result.get("trace", [])
        return vector_result

    terms = _dedupe(keyword_info["entity_keywords"] + keyword_info["relationship_keywords"] + _query_terms(query, intent))
    matched_nodes = _matched_dynamic_nodes(nodes, query, intent, terms)
    relation_paths: list[dict[str, Any]] = []
    for edge in edges:
        score = _score_dynamic_edge(edge, matched_nodes, terms)
        if score <= 0:
            continue
        relation_paths.append(_format_dynamic_relation(edge, score, match_source="dynamic_literal_llm_keywords"))
    if not relation_paths:
        return None
    relation_paths.sort(key=lambda item: item.get("score", 0), reverse=True)
    relation_paths = _dedupe_paths(relation_paths)
    entities = _dynamic_entities(matched_nodes, relation_paths)
    chunks = _search_chunks_for_relation_paths(query, intent, entities, relation_paths, top_k=top_k)
    trace.append(
        {
            "node": "graph_mix_search",
            "output": {
                "match_source": "dynamic_literal_llm_keywords",
                "local_relation_paths": 0,
                "global_relation_paths": 0,
                "literal_relation_paths": len(relation_paths),
            },
        }
    )
    return {"entities": entities, "relation_paths": relation_paths[:top_k], "chunks": chunks[:top_k], "trace": trace}


def _format_relation(rel: dict) -> dict:
    return {"path": f"{rel['from']} -> {rel['relation']} -> {rel['to']}", "description": rel["description"]}


def _dynamic_graph_vector_search(
    query: str,
    intent: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    keyword_info: dict[str, Any],
    top_k: int,
) -> dict | None:
    candidate_k = max(top_k * 3, 6)
    vector_error = None
    try:
        entity_query = " ".join(keyword_info.get("entity_keywords") or [query])
        relationship_query = " ".join(keyword_info.get("relationship_keywords") or [query])
        entity_hits = search_graph_entities(entity_query, top_k=candidate_k)
        relationship_hits = search_graph_relationships(relationship_query, top_k=candidate_k)
    except IndexCollectionUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        entity_hits = []
        relationship_hits = []
        vector_error = str(exc)
    local_paths = _local_relation_paths_from_entity_hits(entity_hits, edges)
    global_paths = _global_relation_paths_from_relationship_hits(relationship_hits, edges)
    relation_paths = _round_robin_paths([local_paths, global_paths], top_k=max(top_k * 4, top_k + len(local_paths) + len(global_paths)))
    relation_paths = _dedupe_paths(relation_paths)
    if not relation_paths:
        return None
    matched_nodes = {str(hit.get("id") or hit.get("label") or "") for hit in entity_hits if hit.get("id") or hit.get("label")}
    entities = _dynamic_entities(matched_nodes, relation_paths)
    chunks = _search_chunks_for_relation_paths(query, intent, entities, relation_paths, top_k=top_k)
    return {
        "entities": entities,
        "relation_paths": relation_paths[:top_k],
        "chunks": chunks[:top_k],
        "trace": [
            {
                "node": "graph_mix_search",
                "output": {
                    "match_source": "graph_vector_mix",
                    "entity_candidates": len(entity_hits),
                    "relationship_candidates": len(relationship_hits),
                    "local_relation_paths": len(local_paths),
                    "global_relation_paths": len(global_paths),
                    "merged_relation_paths": len(relation_paths),
                    "vector_error": vector_error,
                },
            }
        ],
    }


def _local_relation_paths_from_entity_hits(entity_hits: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_by_entity = {str(hit.get("id") or hit.get("label") or ""): _float_value(hit.get("score"), 0.0) for hit in entity_hits}
    matched = {entity for entity in score_by_entity if entity}
    paths: list[dict[str, Any]] = []
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in matched and target not in matched:
            continue
        score = max(score_by_entity.get(source, 0.0), score_by_entity.get(target, 0.0), _float_value(edge.get("confidence"), 0.8))
        paths.append(_format_dynamic_relation(edge, min(1.0, score), match_source="entity_vector"))
    paths.sort(key=lambda item: item.get("score", 0), reverse=True)
    return paths


def _global_relation_paths_from_relationship_hits(relationship_hits: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_lookup = {_edge_key(edge): edge for edge in edges}
    paths: list[dict[str, Any]] = []
    for hit in relationship_hits:
        edge = edge_lookup.get(_edge_key(hit)) or {
            "source": hit.get("source", ""),
            "target": hit.get("target", ""),
            "label": hit.get("label", ""),
            "description": hit.get("description", ""),
            "evidence": hit.get("evidence", ""),
            "source_file": hit.get("source_file", ""),
            "document_id": hit.get("document_id", ""),
            "confidence": hit.get("confidence", 0.8),
        }
        paths.append(_format_dynamic_relation(edge, _float_value(hit.get("score"), 0.0), match_source="relationship_vector"))
    paths.sort(key=lambda item: item.get("score", 0), reverse=True)
    return paths


def _round_robin_paths(streams: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_len = max((len(items) for items in streams), default=0)
    for index in range(max_len):
        for items in streams:
            if index >= len(items):
                continue
            item = items[index]
            path = str(item.get("path") or "")
            if path in seen:
                continue
            seen.add(path)
            merged.append(item)
            if len(merged) >= top_k:
                return merged
    return merged


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source") or edge.get("from") or ""),
        str(edge.get("label") or edge.get("relation") or ""),
        str(edge.get("target") or edge.get("to") or ""),
    )


def _format_dynamic_relation(edge: dict[str, Any], score: float, *, match_source: str) -> dict[str, Any]:
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    relation = str(edge.get("label") or edge.get("relation") or "")
    return {
        "path": f"{source} -> {relation} -> {target}",
        "description": str(edge.get("description") or edge.get("evidence") or ""),
        "source_file": str(edge.get("source_file") or ""),
        "document_id": str(edge.get("document_id") or ""),
        "evidence": str(edge.get("evidence") or ""),
        "score": round(score, 4),
        "graph_source": "dynamic_graph",
        "match_source": match_source,
    }


def _query_terms(query: str, intent: str) -> list[str]:
    terms = extract_entities(query)
    if intent:
        terms.append(intent)
    terms.extend(term for term in COMMON_QUERY_TERMS if term in query or term == intent)
    terms.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", query))
    return _dedupe(terms)


def _matched_dynamic_nodes(nodes: list[dict[str, Any]], query: str, intent: str, terms: list[str]) -> set[str]:
    matched: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id") or node.get("label") or "")
        if not node_id:
            continue
        node_text = _item_text(node)
        if node_id in query or node_id == intent or any(term and term in node_text for term in terms):
            matched.add(node_id)
    return matched


def _score_dynamic_edge(edge: dict[str, Any], matched_nodes: set[str], terms: list[str]) -> float:
    source = str(edge.get("source") or "")
    target = str(edge.get("target") or "")
    edge_text = _item_text(edge)
    endpoint_hits = int(source in matched_nodes) + int(target in matched_nodes)
    term_hits = sum(1 for term in terms if term and term in edge_text)
    if endpoint_hits <= 0 and term_hits <= 0:
        return 0.0
    confidence = _float_value(edge.get("confidence"), 0.8)
    score = confidence * 0.5 + endpoint_hits * 0.12 + min(0.32, term_hits * 0.08)
    return min(1.0, score)


def _dynamic_entities(matched_nodes: set[str], relation_paths: list[dict[str, Any]]) -> list[str]:
    entities = list(matched_nodes)
    for item in relation_paths:
        parts = [part.strip() for part in str(item.get("path", "")).split("->") if part.strip()]
        for index, part in enumerate(parts):
            if index % 2 == 0:
                entities.append(part)
    return _dedupe(entities)


def _search_chunks_for_relation_paths(query: str, intent: str, entities: list[str], relation_paths: list[dict[str, Any]], top_k: int) -> list[dict]:
    try:
        from services.rag_api.vector.chroma_store import search_chunks_by_keywords

        chunks = search_chunks_by_keywords(query, intent, entities, top_k=max(top_k * 4, top_k + len(relation_paths)))
    except Exception:
        return []
    return _boost_relation_chunks(chunks, relation_paths)[:top_k]


def _boost_relation_chunks(chunks: list[dict], relation_paths: list[dict[str, Any]]) -> list[dict]:
    source_files = {str(item.get("source_file") or "") for item in relation_paths if item.get("source_file")}
    evidence_terms = _dedupe(
        [
            str(item.get("evidence") or "")
            for item in relation_paths
            if item.get("evidence")
        ]
        + [
            part.strip()
            for item in relation_paths
            for part in str(item.get("path", "")).split("->")
            if part.strip()
        ]
    )
    boosted: list[dict] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        source_file = str(chunk.get("source_file") or "")
        boost = 0.0
        if source_file in source_files:
            boost += 0.65
        if any(term and term in content for term in evidence_terms):
            boost += 0.25
        score = min(1.0, _float_value(chunk.get("score"), 0.0) + boost)
        boosted.append({**chunk, "score": round(score, 4)})
    boosted.sort(key=lambda item: item.get("score", 0), reverse=True)
    return boosted


def _item_text(item: dict[str, Any]) -> str:
    values: list[str] = []
    for value in item.values():
        if isinstance(value, list):
            values.extend(str(part) for part in value)
        elif isinstance(value, dict):
            values.extend(str(part) for part in value.values())
        else:
            values.append(str(value))
    return " ".join(values)


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_keyword_json(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        raise ValueError("missing keyword json")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("keyword json must be an object")
    return payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe([str(item).strip() for item in value if str(item).strip()])


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
