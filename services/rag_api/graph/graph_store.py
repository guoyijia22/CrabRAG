from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR
from services.rag_api.graph.entities import ENTITIES
from services.rag_api.graph.relations import RELATIONS

KB_GRAPH_PATH = PROJECT_DIR / "data" / "kb_graph.json"


def load_raw_graph(path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if path is None:
        from services.rag_api.security import pinned_artifact_path

        graph_path = pinned_artifact_path("graph.json", KB_GRAPH_PATH)
    else:
        graph_path = path
    if graph_path.exists():
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
            nodes = normalize_dynamic_nodes(data)
            edges = normalize_dynamic_edges(data)
            if nodes or edges:
                return nodes, edges, "dynamic_graph"
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass
    return static_nodes(), static_edges(), "static_graph"


def static_nodes() -> list[dict[str, Any]]:
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


def static_edges() -> list[dict[str, Any]]:
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


def normalize_dynamic_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
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


def normalize_dynamic_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
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
