from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR
from services.rag_api.llm.siliconflow_client import chat_completion

GRAPH_SCHEMA_CONFIG_PATH = PROJECT_DIR / "data" / "graph_schema_config.json"
GRAPH_SCHEMA_SUGGESTION_PATH = PROJECT_DIR / "data" / "graph_schema_suggestion.json"

ALLOWED_NODE_FIELDS = {
    "id",
    "label",
    "type",
    "category",
    "source_files",
    "evidence_count",
    "risk_level",
    "document_count",
    "chunk_count",
}
ALLOWED_EDGE_FIELDS = {
    "source",
    "target",
    "label",
    "relation",
    "description",
    "evidence",
    "source_file",
    "graph_source",
    "category",
    "confidence",
}
ALLOWED_FIELD_TYPES = {"string", "number", "list", "boolean"}
ALLOWED_VISUAL_KEYS = {"node_color_by", "edge_color_by", "node_size_by", "edge_width_by"}


def default_graph_schema() -> dict[str, Any]:
    return {
        "version": 1,
        "status": "default",
        "source": "default",
        "generated_at": "",
        "node_fields": [
            {"key": "id", "label": "实体ID", "type": "string"},
            {"key": "label", "label": "实体名称", "type": "string"},
            {"key": "type", "label": "实体类型", "type": "string"},
            {"key": "category", "label": "业务类别", "type": "string"},
            {"key": "source_files", "label": "来源文件", "type": "list"},
            {"key": "evidence_count", "label": "证据数量", "type": "number"},
        ],
        "edge_fields": [
            {"key": "source", "label": "起点实体", "type": "string"},
            {"key": "target", "label": "终点实体", "type": "string"},
            {"key": "label", "label": "关系名称", "type": "string"},
            {"key": "description", "label": "关系说明", "type": "string"},
            {"key": "evidence", "label": "原文证据", "type": "string"},
            {"key": "source_file", "label": "来源文件", "type": "string"},
            {"key": "graph_source", "label": "图谱来源", "type": "string"},
        ],
        "detail_fields": {
            "node": ["type", "category", "source_files", "evidence_count"],
            "edge": ["description", "evidence", "source_file", "graph_source"],
        },
        "visual_mappings": {
            "node_color_by": "type",
            "edge_color_by": "graph_source",
            "node_size_by": "evidence_count",
            "edge_width_by": "confidence",
        },
        "recommendation": "默认结构适合展示实体类型、业务类别、来源文件和关系证据。",
    }


def load_graph_schema() -> dict[str, Any]:
    if not GRAPH_SCHEMA_CONFIG_PATH.exists():
        return default_graph_schema()
    try:
        payload = json.loads(GRAPH_SCHEMA_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_graph_schema()
    return _validate_schema(payload, status="confirmed", source=payload.get("source", "config"))


def load_graph_schema_suggestion() -> dict[str, Any]:
    from services.rag_api.index_generation import active_artifact_path

    suggestion_path = active_artifact_path("graph_schema_suggestion.json", GRAPH_SCHEMA_SUGGESTION_PATH)
    if not suggestion_path.exists():
        fallback = default_graph_schema()
        fallback["status"] = "missing"
        fallback["source"] = "default"
        return fallback
    try:
        payload = json.loads(suggestion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_graph_schema()
    return _validate_schema(payload, status="suggested", source=payload.get("source", "suggestion"))


def save_graph_schema_suggestion(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    schema = _validate_schema(payload, status="suggested", source=payload.get("source", "llm"))
    suggestion_path = path or GRAPH_SCHEMA_SUGGESTION_PATH
    suggestion_path.parent.mkdir(parents=True, exist_ok=True)
    suggestion_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema


def save_graph_schema_config(payload: dict[str, Any]) -> dict[str, Any]:
    schema = _validate_schema(payload, status="confirmed", source=payload.get("source", "config"))
    GRAPH_SCHEMA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_SCHEMA_CONFIG_PATH.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema


def generate_graph_schema_suggestion(
    category_payload: dict[str, Any],
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    path: Path | None = None,
) -> dict[str, Any]:
    try:
        content = chat_completion(
            [
                {"role": "system", "content": _schema_system_prompt()},
                {"role": "user", "content": json.dumps(_safe_summary(category_payload, documents, chunks), ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=900,
        )
        parsed = _extract_json(content)
        parsed["source"] = "llm"
        return save_graph_schema_suggestion(parsed, path=path)
    except Exception:
        fallback = _fallback_suggestion(category_payload)
        fallback["source"] = "fallback"
        return save_graph_schema_suggestion(fallback, path=path)


def _schema_system_prompt() -> str:
    return (
        "你是本地知识图谱 API 结构设计助手。请根据知识库摘要，建议适合图谱节点、边、详情面板和视觉映射的 JSON 配置。"
        "只能输出 JSON，不要解释。字段 key 只能使用白名单："
        f"node_fields={sorted(ALLOWED_NODE_FIELDS)}, edge_fields={sorted(ALLOWED_EDGE_FIELDS)}, visual_mappings={sorted(ALLOWED_VISUAL_KEYS)}。"
        "JSON 必须包含 node_fields、edge_fields、detail_fields、visual_mappings、recommendation。"
    )


def _safe_summary(category_payload: dict[str, Any], documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "categories": [
            {
                "name": item.get("name", ""),
                "document_count": item.get("document_count", 0),
                "chunk_count": item.get("chunk_count", 0),
                "source_files": item.get("source_files", [])[:8],
            }
            for item in category_payload.get("items", [])[:12]
        ],
        "source_files": [doc.get("source_file", "") for doc in documents[:16]],
        "section_titles": [chunk.get("metadata", {}).get("section_title", "") for chunk in chunks[:24]],
        "chunk_count": len(chunks),
    }


def _fallback_suggestion(category_payload: dict[str, Any]) -> dict[str, Any]:
    schema = default_graph_schema()
    categories = [item.get("name", "") for item in category_payload.get("items", []) if item.get("name")]
    if any("合规" in name or "审核" in name for name in categories):
        schema["node_fields"].append({"key": "risk_level", "label": "风险等级", "type": "string"})
        schema["detail_fields"]["node"].append("risk_level")
    schema["recommendation"] = "根据知识库分类生成的兜底结构，建议展示实体类型、业务类别、来源文件、证据数量和关系来源。"
    return schema


def _extract_json(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError("missing json object")
    return json.loads(match.group(0))


def _validate_schema(payload: dict[str, Any], status: str, source: str) -> dict[str, Any]:
    base = default_graph_schema()
    node_fields = _sanitize_fields(payload.get("node_fields"), ALLOWED_NODE_FIELDS) or base["node_fields"]
    edge_fields = _sanitize_fields(payload.get("edge_fields"), ALLOWED_EDGE_FIELDS) or base["edge_fields"]
    detail_fields = _sanitize_detail_fields(payload.get("detail_fields"), node_fields, edge_fields)
    visual_mappings = _sanitize_visual_mappings(payload.get("visual_mappings"))
    return {
        "version": 1,
        "status": status,
        "source": source or "config",
        "generated_at": payload.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node_fields": node_fields,
        "edge_fields": edge_fields,
        "detail_fields": detail_fields,
        "visual_mappings": visual_mappings or base["visual_mappings"],
        "recommendation": str(payload.get("recommendation") or base["recommendation"])[:400],
    }


def _sanitize_fields(fields: Any, allowed_keys: set[str]) -> list[dict[str, str]]:
    if not isinstance(fields, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        if not isinstance(field, dict):
            continue
        key = str(field.get("key", "")).strip()
        if key not in allowed_keys or key in seen:
            continue
        label = str(field.get("label") or key).strip()[:32]
        field_type = str(field.get("type") or "string").strip()
        if field_type not in ALLOWED_FIELD_TYPES:
            field_type = "string"
        result.append({"key": key, "label": label, "type": field_type})
        seen.add(key)
    return result


def _sanitize_detail_fields(detail_fields: Any, node_fields: list[dict[str, str]], edge_fields: list[dict[str, str]]) -> dict[str, list[str]]:
    node_allowed = {field["key"] for field in node_fields}
    edge_allowed = {field["key"] for field in edge_fields}
    if not isinstance(detail_fields, dict):
        return {
            "node": [field["key"] for field in node_fields if field["key"] not in {"id", "label"}][:6],
            "edge": [field["key"] for field in edge_fields if field["key"] not in {"source", "target", "label"}][:6],
        }
    return {
        "node": _sanitize_key_list(detail_fields.get("node"), node_allowed)[:8],
        "edge": _sanitize_key_list(detail_fields.get("edge"), edge_allowed)[:8],
    }


def _sanitize_visual_mappings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    allowed_values = ALLOWED_NODE_FIELDS | ALLOWED_EDGE_FIELDS
    for key, raw in value.items():
        if key in ALLOWED_VISUAL_KEYS and str(raw) in allowed_values:
            result[key] = str(raw)
    return result


def _sanitize_key_list(value: Any, allowed: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        key = str(item)
        if key in allowed and key not in result:
            result.append(key)
    return result
