from __future__ import annotations

from typing import Any

from services.rag_api.document.categories import load_kb_categories
from services.rag_api.graph.graph_store import KB_GRAPH_PATH, load_raw_graph
from services.rag_api.graph.schema_config import load_graph_schema


def build_graph_payload() -> dict[str, Any]:
    schema = load_graph_schema()
    raw_nodes, raw_edges, graph_source = load_raw_graph(KB_GRAPH_PATH)
    nodes = [_node_payload(node, schema) for node in raw_nodes]
    edges = [_edge_payload(edge, schema) for edge in raw_edges]
    graph_source_files = _graph_source_files(nodes, edges)
    evidence_source_files = _evidence_source_files(edges)
    source_files = graph_source_files or _category_source_files()
    if graph_source == "static_graph" and not source_files:
        nodes = []
        edges = []
        graph_source = "empty_graph"
        evidence_source_files = set()
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
        "empty_graph": "暂无知识图谱",
    }.get(graph_source, graph_source or "-")
