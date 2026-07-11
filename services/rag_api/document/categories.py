from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from services.rag_api.config import PROJECT_DIR

KB_CATEGORIES_PATH = PROJECT_DIR / "data" / "kb_categories.json"

DEFAULT_CATEGORIES = ["客户准入", "办理流程", "资费咨询", "合规审核", "业务变更", "故障报修", "退订销户"]

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("客户准入", ["客户准入", "准入", "资质", "营业执照", "授权证明", "集团客户", "企业客户"]),
    ("办理流程", ["办理流程", "业务受理", "受理", "申请", "开通", "装机", "流程"]),
    ("资费咨询", ["资费", "套餐", "月费", "价格", "带宽资费", "费用", "计费"]),
    ("合规审核", ["合规", "一票否决", "审核", "红线", "管控", "禁止"]),
    ("业务变更", ["业务变更", "变更", "带宽变更", "地址迁移", "移机", "资源变更"]),
    ("故障报修", ["故障", "报修", "报障", "中断", "维修", "修复时限", "投诉"]),
    ("退订销户", ["退订", "销户", "合同终止", "取消业务", "欠费", "设备回收"]),
    ("公司法律法规", ["公司法", "中华人民共和国公司法", "股东", "董事", "监事", "公司登记", "法定代表人"]),
    ("投诉处理", ["投诉", "申诉", "争议", "客户投诉", "处理时限", "服务质量"]),
]


def infer_document_category(source_file: str, text: str = "") -> str:
    haystack = f"{source_file}\n{text[:1200]}"
    scores: list[tuple[str, int]] = []
    for category, keywords in CATEGORY_RULES:
        score = sum(1 for keyword in keywords if keyword and keyword in haystack)
        if score > 0:
            scores.append((category, score))
    if scores:
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[0][0]
    return _category_from_filename(source_file)


def save_kb_categories(documents: list[dict[str, Any]], chunks: list[dict[str, Any]], path: Path | None = None) -> dict[str, Any]:
    doc_categories = {doc["source_file"]: infer_document_category(doc["source_file"], doc.get("content", "")) for doc in documents}
    chunk_counter: Counter[str] = Counter()
    source_files: dict[str, set[str]] = defaultdict(set)
    document_counter: Counter[str] = Counter(doc_categories.values())
    keyword_hits: dict[str, list[str]] = defaultdict(list)

    for doc in documents:
        category = doc_categories[doc["source_file"]]
        source_files[category].add(doc["source_file"])
        keyword_hits[category].extend(_matched_keywords(doc["source_file"], doc.get("content", "")))

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        category = meta.get("category") or doc_categories.get(meta.get("source_file", ""), "合规审核")
        chunk_counter[category] += 1
        if meta.get("source_file"):
            source_files[category].add(meta["source_file"])

    items = []
    ordered_names = _ordered_category_names(set(document_counter) | set(chunk_counter))
    for name in ordered_names:
        items.append(
            {
                "name": name,
                "document_count": int(document_counter.get(name, 0)),
                "chunk_count": int(chunk_counter.get(name, 0)),
                "source_files": sorted(source_files.get(name, set())),
                "keyword_hits": sorted(set(keyword_hits.get(name, []))),
            }
        )

    payload = {
        "items": items,
        "categories": [item["name"] for item in items],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    categories_path = path or KB_CATEGORIES_PATH
    categories_path.parent.mkdir(parents=True, exist_ok=True)
    categories_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_kb_categories() -> dict[str, Any]:
    from services.rag_api.index_generation import active_artifact_path

    categories_path = active_artifact_path("categories.json", KB_CATEGORIES_PATH)
    if not categories_path.exists():
        return default_categories_payload()
    try:
        payload = json.loads(categories_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_categories_payload()
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return default_categories_payload()
    categories = [item.get("name", "") for item in items if item.get("name")]
    return {
        "items": items,
        "categories": categories,
        "generated_at": payload.get("generated_at", ""),
    }


def default_categories_payload() -> dict[str, Any]:
    return {
        "items": [],
        "categories": [],
        "generated_at": "",
    }


def get_category_names() -> list[str]:
    payload = load_kb_categories()
    names = [name for name in payload.get("categories", []) if name]
    return names


def source_files_for_category(category: str) -> list[str]:
    payload = load_kb_categories()
    for item in payload.get("items", []):
        if item.get("name") == category:
            return item.get("source_files", []) or []
    return []


def _category_from_filename(source_file: str) -> str:
    stem = Path(source_file).stem
    for token in ["规范", "办法", "规则", "条例", "指引", "手册", "制度"]:
        stem = stem.replace(token, "")
    stem = stem.strip(" _-0123456789")
    return stem[:16] if stem else "合规审核"


def _matched_keywords(source_file: str, text: str) -> list[str]:
    haystack = f"{source_file}\n{text[:1200]}"
    hits: list[str] = []
    for _, keywords in CATEGORY_RULES:
        hits.extend(keyword for keyword in keywords if keyword in haystack)
    return hits


def _ordered_category_names(names: set[str]) -> list[str]:
    ordered = [name for name in DEFAULT_CATEGORIES if name in names]
    ordered.extend(sorted(name for name in names if name not in DEFAULT_CATEGORIES))
    return ordered
