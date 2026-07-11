from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from services.rag_api.graph.graph_store import KB_GRAPH_PATH


def build_and_save_kb_graph(
    category_payload: dict[str, Any],
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    path: Path | None = None,
) -> dict[str, Any]:
    return save_kb_graph(build_kb_graph(category_payload, documents, chunks), path=path)


def build_kb_graph(category_payload: dict[str, Any], documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    document_text = {str(doc.get("source_file") or ""): str(doc.get("content") or "") for doc in documents if doc.get("source_file")}
    document_ids = {
        str(doc.get("source_file") or ""): str(doc.get("document_id") or doc.get("doc_id") or "")
        for doc in documents
        if doc.get("source_file")
    }
    chunk_counts = Counter(str(chunk.get("metadata", {}).get("source_file") or "") for chunk in chunks)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    for item in category_payload.get("items", []) or []:
        category = str(item.get("name") or "").strip()
        if not category:
            continue
        source_files = [str(source_file) for source_file in item.get("source_files", []) or [] if source_file]
        _merge_node(
            nodes,
            {
                "id": category,
                "label": category,
                "type": "知识分类",
                "category": category,
                "source_files": source_files,
                "document_ids": [document_ids[source_file] for source_file in source_files if document_ids.get(source_file)],
                "document_sources": [
                    {"document_id": document_ids[source_file], "source_file": source_file}
                    for source_file in source_files
                    if document_ids.get(source_file)
                ],
                "evidence_count": int(item.get("chunk_count") or 0),
                "chunk_count": int(item.get("chunk_count") or 0),
                "document_count": int(item.get("document_count") or len(source_files)),
            },
        )
        for source_file in source_files:
            source_chunk_count = int(chunk_counts.get(source_file, 0))
            _merge_node(
                nodes,
                {
                    "id": source_file,
                    "label": source_file,
                    "type": "来源文件",
                    "category": category,
                    "source_files": [source_file],
                    "document_ids": [document_ids[source_file]] if document_ids.get(source_file) else [],
                    "document_sources": (
                        [{"document_id": document_ids[source_file], "source_file": source_file}]
                        if document_ids.get(source_file)
                        else []
                    ),
                    "evidence_count": source_chunk_count,
                    "chunk_count": source_chunk_count,
                    "document_count": 1,
                },
            )
            _merge_edge(
                edges,
                {
                    "source": category,
                    "target": source_file,
                    "label": "包含文件",
                    "relation": "包含文件",
                    "description": f"{category} 分类包含来源文件 {source_file}",
                    "evidence": source_file,
                    "source_file": source_file,
                    "document_id": document_ids.get(source_file, ""),
                    "graph_source": "generated_from_kb",
                    "confidence": 0.9,
                },
            )
            for topic in extract_source_topics(source_file, document_text.get(source_file, "")):
                _merge_node(
                    nodes,
                    {
                        "id": topic,
                        "label": topic,
                        "type": "主题实体",
                        "category": category,
                        "source_files": [source_file],
                        "document_ids": [document_ids[source_file]] if document_ids.get(source_file) else [],
                        "document_sources": (
                            [{"document_id": document_ids[source_file], "source_file": source_file}]
                            if document_ids.get(source_file)
                            else []
                        ),
                        "evidence_count": 1,
                    },
                )
                _merge_edge(
                    edges,
                    {
                        "source": source_file,
                        "target": topic,
                        "label": "提及主题",
                        "relation": "提及主题",
                        "description": f"{source_file} 提及主题 {topic}",
                        "evidence": source_file,
                        "source_file": source_file,
                        "document_id": document_ids.get(source_file, ""),
                        "graph_source": "generated_from_kb",
                        "confidence": 0.85,
                    },
                )
                _merge_edge(
                    edges,
                    {
                        "source": topic,
                        "target": category,
                        "label": "关联分类",
                        "relation": "关联分类",
                        "description": f"{topic} 关联到知识分类 {category}",
                        "evidence": source_file,
                        "source_file": source_file,
                        "document_id": document_ids.get(source_file, ""),
                        "graph_source": "generated_from_kb",
                        "confidence": 0.8,
                    },
                )

    return {
        "nodes": [_finalize_node(node) for node in nodes.values()],
        "edges": [_finalize_edge(edge) for edge in edges.values()],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "graph_source": "dynamic_graph",
    }


def save_kb_graph(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    graph_path = path or KB_GRAPH_PATH
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def extract_source_topics(source_file: str, text: str = "") -> list[str]:
    stem = Path(source_file).stem
    topics: list[str] = []
    topics.extend(_quoted_terms(stem))
    topics.extend(_quoted_terms(text[:1200]))
    if "一渠一表" in stem or "一渠一表" in text[:1200]:
        topics.append("一渠一表")
    cleaned = _clean_title_topic(stem)
    if cleaned and "“" not in cleaned and "”" not in cleaned and 2 <= len(cleaned) <= 24:
        topics.append(cleaned)
    return _dedupe(topics)


def _merge_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    node_id = str(node.get("id") or "").strip()
    if not node_id:
        return
    existing = nodes.setdefault(
        node_id,
        {
            **node,
            "source_files": set(node.get("source_files", []) or []),
            "document_ids": set(node.get("document_ids", []) or []),
            "document_sources": {
                str(item.get("document_id")): str(item.get("source_file"))
                for item in node.get("document_sources", []) or []
                if item.get("document_id")
            },
        },
    )
    existing["source_files"].update(node.get("source_files", []) or [])
    existing["document_ids"].update(node.get("document_ids", []) or [])
    existing["document_sources"].update(
        {
            str(item.get("document_id")): str(item.get("source_file"))
            for item in node.get("document_sources", []) or []
            if item.get("document_id")
        }
    )
    existing["evidence_count"] = max(int(existing.get("evidence_count") or 0), int(node.get("evidence_count") or 0))
    existing["chunk_count"] = max(int(existing.get("chunk_count") or 0), int(node.get("chunk_count") or 0))
    existing["document_count"] = max(int(existing.get("document_count") or 0), int(node.get("document_count") or 0))


def _merge_edge(edges: dict[tuple[str, str, str], dict[str, Any]], edge: dict[str, Any]) -> None:
    key = (str(edge.get("source") or ""), str(edge.get("label") or ""), str(edge.get("target") or ""))
    if all(key):
        edges.setdefault(key, edge)


def _finalize_node(node: dict[str, Any]) -> dict[str, Any]:
    result = dict(node)
    result["source_files"] = sorted(str(source_file) for source_file in result.get("source_files", []) if source_file)
    result["document_ids"] = sorted(str(document_id) for document_id in result.get("document_ids", []) if document_id)
    result["document_sources"] = [
        {"document_id": document_id, "source_file": source_file}
        for document_id, source_file in sorted((result.get("document_sources") or {}).items())
    ]
    if result.get("type") == "主题实体":
        result["evidence_count"] = max(int(result.get("evidence_count") or 0), len(result["source_files"]))
    return result


def _finalize_edge(edge: dict[str, Any]) -> dict[str, Any]:
    source = str(edge.get("source") or "")
    target = str(edge.get("target") or "")
    label = str(edge.get("label") or edge.get("relation") or "")
    return {"id": f"{source}->{label}->{target}", **edge}


def _quoted_terms(text: str) -> list[str]:
    terms = re.findall(r"[“《\"']([^”》\"']{2,30})[”》\"']", text)
    return [term.strip() for term in terms if term.strip()]


def _clean_title_topic(stem: str) -> str:
    value = re.sub(r"^V\d+(?:\.\d+)?", "", stem).strip()
    value = re.sub(r"[-_（(]?\d{4}(?:\d{2})?(?:\d{2})?[）)]?$", "", value).strip(" _-（）()")
    for prefix in ["关于", "基于"]:
        if value.startswith(prefix):
            value = value[len(prefix) :]
    for suffix in ["调研报告", "项目效益评价", "科技创新", "报告", "材料", "方案"]:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.strip(" _-（）()")


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return result
