from __future__ import annotations

import json
import re
from typing import Any

from services.rag_api.app_settings import DEFAULT_KNOWLEDGE_BASE_NAME
from services.rag_api.config import read_app_config, write_knowledge_base_name
from services.rag_api.llm.siliconflow_client import chat_completion


def ensure_knowledge_base_name(category_payload: dict[str, Any], documents: list[dict[str, Any]], chunk_count: int) -> tuple[str, str]:
    current = read_app_config().get("knowledge_base_name", "")
    if current and current != DEFAULT_KNOWLEDGE_BASE_NAME:
        return current, "config"
    try:
        generated = _generate_name_with_llm(category_payload, documents, chunk_count)
        if generated:
            return write_knowledge_base_name(generated), "llm"
    except Exception:
        pass
    return write_knowledge_base_name(_fallback_name(category_payload)), "fallback"


def _generate_name_with_llm(category_payload: dict[str, Any], documents: list[dict[str, Any]], chunk_count: int) -> str:
    categories = [item.get("name", "") for item in category_payload.get("items", []) if item.get("name")]
    source_files = [doc.get("source_file", "") for doc in documents[:12]]
    prompt = {
        "task": "请根据知识库类别和文件名，为本地规范知识库取一个 8 到 16 个汉字的中文名称。只输出名称，不要解释。",
        "categories": categories,
        "source_files": source_files,
        "document_count": len(documents),
        "chunk_count": chunk_count,
    }
    content = chat_completion(
        [
            {"role": "system", "content": "你是知识库命名助手，只输出简洁中文名称。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=80,
    )
    return _sanitize_name(content)


def _fallback_name(category_payload: dict[str, Any]) -> str:
    names = [item.get("name", "") for item in category_payload.get("items", []) if item.get("name")]
    if any(name in names for name in ["客户准入", "办理流程", "资费咨询", "合规审核", "业务变更", "故障报修", "退订销户"]):
        return "政企专线规范知识库"
    if any("公司" in name or "法律" in name for name in names):
        return "公司法规知识库"
    return "通信行业规范知识库"


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^\u4e00-\u9fff]", "", value)
    if len(text) < 4:
        return ""
    return text[:16]
