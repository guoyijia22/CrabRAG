from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR
from services.rag_api.document.categories import load_kb_categories
from services.rag_api.graph.entities import ENTITIES
from services.rag_api.graph.relations import RELATIONS
from services.rag_api.graph.schema_config import load_graph_schema

KB_GRAPH_PATH = PROJECT_DIR / "data" / "kb_graph.json"


def build_graph_payload() -> dict[str, Any]:
    schema = load_graph_schema()
    raw_nodes, raw_edges, graph_source = _load_raw_graph()
    nodes = [_node_payload(node, schema) for node in raw_nodes]
    edges = [_edge_payload(edge, schema) for edge in raw_edges]
    graph_source_files = _graph_source_files(nodes, edges)
    evidence_source_files = _evidence_source_files(edges)
    source_files = graph_source_files or _category_source_files()
    return {
        "schema": schema,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_file_count": len(source_files),
            "evidence_source_file_count": len(evidence_source_files),
            "graph_source": graph_source,
            "graph_source_label": _graph_source_label(graph_source),
        },
    }


def build_subgraph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema = load_graph_schema()
    relation_paths = payload.get("relation_paths") or []
    raw_nodes: dict[str, dict[str, Any]] = {}
    raw_edges: list[dict[str, Any]] = []
    for item in relation_paths:
        path = str(item.get("path", ""))
        parts = [part.strip() for part in path.split("->") if part.strip()]
        if len(parts) < 3:
            continue
        for index in range(0, len(parts) - 2, 2):
            source = parts[index]
            relation = parts[index + 1]
            target = parts[index + 2]
            raw_nodes.setdefault(source, {"id": source, "label": source, "type": "命中实体"})
            raw_nodes.setdefault(target, {"id": target, "label": target, "type": "命中实体"})
            raw_edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": relation,
                    "relation": relation,
                    "description": item.get("description", ""),
                    "evidence": item.get("evidence", ""),
                    "source_file": item.get("source_file", ""),
                    "graph_source": item.get("graph_source", "retrieved_path"),
                    "confidence": item.get("score", 0.8),
                }
            )
    nodes = [_node_payload(node, schema) for node in raw_nodes.values()]
    edges = [_edge_payload(edge, schema) for edge in raw_edges]
    return {
        "schema": schema,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_file_count": len({edge["properties"].get("source_file", "") for edge in edges if edge["properties"].get("source_file")}),
            "evidence_source_file_count": len({edge["properties"].get("source_file", "") for edge in edges if edge["properties"].get("source_file")}),
            "graph_source": "subgraph",
            "graph_source_label": _graph_source_label("subgraph"),
        },
    }


def _load_raw_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if KB_GRAPH_PATH.exists():
        try:
            data = json.loads(KB_GRAPH_PATH.read_text(encoding="utf-8"))
            nodes = _normalize_dynamic_nodes(data)
            edges = _normalize_dynamic_edges(data)
            if nodes or edges:
                return nodes, edges, "dynamic_graph"
        except (OSError, json.JSONDecodeError):
            pass
    return _static_nodes(), _static_edges(), "static_graph"


def _static_nodes() -> list[dict[str, Any]]:
    return [
        {
            "id": entity,
            "label": entity,
            "type": detail.get("type", ""),
            "category": "",
            "source_files": [],
            "evidence_count": 0,
        }
        for entity, detail in ENTITIES.items()
    ]


def _static_edges() -> list[dict[str, Any]]:
    return [
        {
            "source": rel.get("from", ""),
            "target": rel.get("to", ""),
            "label": rel.get("relation", ""),
            "relation": rel.get("relation", ""),
            "description": rel.get("description", ""),
            "evidence": "",
            "source_file": "",
            "graph_source": "static_graph",
            "confidence": 1.0,
        }
        for rel in RELATIONS
    ]


def _normalize_dynamic_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_nodes = data.get("nodes", [])
    result: list[dict[str, Any]] = []
    if isinstance(raw_nodes, list):
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id") or item.get("label") or item.get("name") or "").strip()
            if not node_id:
                continue
            result.append(
                {
                    "id": node_id,
                    "label": str(item.get("label") or node_id),
                    "type": item.get("type", ""),
                    "category": item.get("category", ""),
                    "source_files": item.get("source_files", []),
                    "evidence_count": item.get("evidence_count", 0),
                    **{key: value for key, value in item.items() if key not in {"id", "label"}},
                }
            )
    return result


def _normalize_dynamic_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_edges = data.get("edges") or data.get("relationships") or []
    result: list[dict[str, Any]] = []
    if isinstance(raw_edges, list):
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("from") or "").strip()
            target = str(item.get("target") or item.get("to") or "").strip()
            if not source or not target:
                continue
            label = str(item.get("label") or item.get("relation") or item.get("type") or "")
            result.append(
                {
                    "source": source,
                    "target": target,
                    "label": label,
                    "relation": label,
                    "description": item.get("description", ""),
                    "evidence": item.get("evidence", ""),
                    "source_file": item.get("source_file", ""),
                    "graph_source": item.get("graph_source", "dynamic_graph"),
                    "confidence": item.get("confidence", 0.8),
                    **{key: value for key, value in item.items() if key not in {"source", "from", "target", "to", "label", "relation", "type"}},
                }
            )
    return result


def _node_payload(raw: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    node_id = str(raw.get("id") or raw.get("label") or "")
    label = str(raw.get("label") or node_id)
    return {"id": node_id, "label": label, "properties": _properties(raw, schema.get("node_fields", []))}


def _edge_payload(raw: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    source = str(raw.get("source", ""))
    target = str(raw.get("target", ""))
    label = str(raw.get("label") or raw.get("relation") or "")
    edge_id = str(raw.get("id") or f"{source}->{target}->{label}")
    return {"id": edge_id, "source": source, "target": target, "label": label, "properties": _properties(raw, schema.get("edge_fields", []))}


def _properties(raw: dict[str, Any], fields: list[dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for field in fields:
        key = field.get("key")
        if not key:
            continue
        properties[key] = raw.get(key, _default_value(field.get("type")))
    return properties


def _default_value(field_type: str | None) -> Any:
    if field_type == "number":
        return 0
    if field_type == "list":
        return []
    if field_type == "boolean":
        return False
    return ""


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def _graph_source_files(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> set[str]:
    files: set[str] = set()
    for node in nodes:
        for value in _as_list(node.get("properties", {}).get("source_files")):
            if value:
                files.add(value)
    for edge in edges:
        for value in _as_list(edge.get("properties", {}).get("source_file")):
            if value:
                files.add(value)
    return files


def _evidence_source_files(edges: list[dict[str, Any]]) -> set[str]:
    files: set[str] = set()
    for edge in edges:
        props = edge.get("properties", {})
        if props.get("evidence") and props.get("source_file"):
            files.add(str(props["source_file"]))
    return files


def _category_source_files() -> set[str]:
    files: set[str] = set()
    for item in load_kb_categories().get("items", []):
        for source_file in item.get("source_files", []) or []:
            if source_file:
                files.add(str(source_file))
    return files


def _graph_source_label(graph_source: str) -> str:
    return {
        "static_graph": "内置基础图谱",
        "dynamic_graph": "知识库动态图谱",
        "subgraph": "问答命中子图",
    }.get(graph_source, graph_source or "-")
