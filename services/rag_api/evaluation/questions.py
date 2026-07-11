from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from services.rag_api.document.categories import DEFAULT_CATEGORIES, load_kb_categories
from services.rag_api.evaluation.dataset import dataset_to_question_set, load_evaluation_dataset
from services.rag_api.graph.relations import RELATIONS
from services.rag_api.llm.siliconflow_client import chat_completion
from services.rag_api.vector.chroma_store import search_all_chunks

MIN_QUESTIONS = 8
MAX_QUESTIONS = 14


def generate_evaluation_question_set() -> dict[str, Any]:
    """Generate a reproducible evaluation question set for the current KB."""

    fixed_dataset = load_evaluation_dataset()
    if fixed_dataset is not None:
        return dataset_to_question_set(fixed_dataset)

    category_payload = load_kb_categories()
    chunks = _load_sample_chunks()
    relation_payload = _relation_summaries()

    try:
        generated = _generate_with_llm(category_payload, chunks, relation_payload)
        questions = _validate_questions(generated, category_payload)
        if questions:
            if len(questions) < MIN_QUESTIONS:
                questions = _dedupe_questions(questions + _fallback_questions(category_payload))
            return _payload("llm", questions, category_payload)
    except Exception:  # noqa: BLE001
        pass

    return _payload("fallback", _fallback_questions(category_payload), category_payload)


def _generate_with_llm(category_payload: dict[str, Any], chunks: list[dict], relations: list[dict]) -> Any:
    prompt = {
        "categories": [
            {
                "name": item.get("name", ""),
                "source_files": item.get("source_files", [])[:6],
                "chunk_count": item.get("chunk_count", 0),
            }
            for item in category_payload.get("items", [])
        ],
        "chunks": [
            {
                "category": item.get("category", ""),
                "source_file": item.get("source_file", ""),
                "section_title": item.get("section_title", ""),
                "summary": (item.get("content", "") or "")[:180],
            }
            for item in chunks
        ],
        "relations": relations,
    }
    response = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你是 RAG 系统评测题生成器。请只输出 JSON。"
                    "基于给定知识库分类、来源文件、片段摘要和图谱关系，生成 8-14 道评测题。"
                    "题目要覆盖分类规则题、图谱关系题、多轮追问题和 1 道知识库外兜底边界题。"
                    "每题字段必须包含 id、question、history、category、question_type、expected_intent、"
                    "expected_retrieval_modes、expect_references、expect_relation_paths、expected_source_files、source_category。"
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_tokens=2200,
    )
    text = response.get("content", "") if isinstance(response, dict) else str(response)
    return _parse_json(text)


def _validate_questions(raw: Any, category_payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw_items = raw.get("questions", [])
    else:
        raw_items = raw
    if not isinstance(raw_items, list):
        return []

    allowed_categories = _category_names(category_payload)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question or question in seen:
            continue
        seen.add(question)
        category = str(item.get("category") or item.get("expected_intent") or "").strip()
        if category not in allowed_categories and category not in DEFAULT_CATEGORIES:
            category = _closest_category(category, allowed_categories)
        normalized = {
            "id": str(item.get("id") or f"llm_q_{index}").strip(),
            "question": question,
            "history": _normalize_history(item.get("history", [])),
            "category": category,
            "question_type": str(item.get("question_type") or "单一规则").strip(),
            "expected_intent": str(item.get("expected_intent") or category).strip(),
            "expected_retrieval_modes": _normalize_modes(item.get("expected_retrieval_modes")),
            "expect_references": bool(item.get("expect_references", True)),
            "expect_relation_paths": bool(item.get("expect_relation_paths", False)),
            "expected_source_files": _normalize_str_list(item.get("expected_source_files")),
            "source_category": str(item.get("source_category") or category).strip(),
        }
        result.append(normalized)
        if len(result) >= MAX_QUESTIONS:
            break
    return result if len(result) >= 1 else []


def _fallback_questions(category_payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = category_payload.get("items", []) or []
    questions: list[dict[str, Any]] = []

    for item in items:
        category = item.get("name", "")
        if not category:
            continue
        questions.append(_category_question(category, item))
        if len(questions) >= 7:
            break

    questions.extend(_relation_questions(items))
    questions.append(_context_followup_question())
    questions.append(
        {
            "id": "q_boundary_no_match",
            "question": "政企专线能不能赠送手机套餐？",
            "history": [],
            "category": "知识库外问题",
            "question_type": "兜底边界",
            "expected_intent": "合规审核",
            "expected_retrieval_modes": ["vector", "graph", "hybrid"],
            "expect_references": False,
            "expect_relation_paths": False,
            "expected_source_files": [],
            "source_category": "知识库外问题",
        }
    )

    for default_category in DEFAULT_CATEGORIES:
        if len(questions) >= MIN_QUESTIONS:
            break
        questions.append(_category_question(default_category, {"name": default_category, "source_files": []}))

    return _dedupe_questions(questions)[:MAX_QUESTIONS]


def _category_question(category: str, item: dict[str, Any]) -> dict[str, Any]:
    source_files = list(item.get("source_files", []) or [])
    templates = {
        "客户准入": ("企业客户办理政企专线需要提供哪些材料？", "单一规则", ["vector", "hybrid"]),
        "办理流程": ("政企专线从申请到开通需要经过哪些步骤？", "流程步骤", ["vector"]),
        "资费咨询": ("政企专线资费套餐标准有哪些要求？", "资费标准", ["vector"]),
        "合规审核": ("哪些情况属于业务开办一票否决或合规审核红线？", "单一规则", ["vector", "hybrid"]),
        "业务变更": ("企业客户办理地址迁移或带宽变更时需要哪些审核？", "关联推理", ["graph", "hybrid"]),
        "故障报修": ("专线中断后的故障报修流程和时限要求是什么？", "关联推理", ["graph", "hybrid"]),
        "退订销户": ("客户存在欠费时能不能直接办理退订销户？", "关联推理", ["graph", "hybrid"]),
        "公司法律法规": ("公司法中关于公司治理或法定代表人的主要规定是什么？", "单一规则", ["vector"]),
        "投诉处理": ("客户投诉处理流程和处理时限要求是什么？", "流程步骤", ["vector", "hybrid"]),
    }
    question, question_type, modes = templates.get(category, (f"请概括{category}相关的主要规范要求。", "单一规则", ["vector"]))
    return {
        "id": f"q_category_{_slug(category)}",
        "question": question,
        "history": [],
        "category": category,
        "question_type": question_type,
        "expected_intent": category,
        "expected_retrieval_modes": modes,
        "expect_references": True,
        "expect_relation_paths": "关联推理" in question_type,
        "expected_source_files": source_files,
        "source_category": category,
    }


def _relation_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_by_category = {item.get("name", ""): item.get("source_files", []) or [] for item in items}
    return [
        {
            "id": "q_relation_change_audit",
            "question": "企业客户办理地址迁移时，是否需要重新进行合规审核？",
            "history": [],
            "category": "业务变更",
            "question_type": "关联推理",
            "expected_intent": "业务变更",
            "expected_retrieval_modes": ["graph", "hybrid"],
            "expect_references": True,
            "expect_relation_paths": True,
            "expected_source_files": source_by_category.get("业务变更", []) + source_by_category.get("合规审核", []),
            "source_category": "业务变更",
        },
        {
            "id": "q_relation_cancel_debt",
            "question": "客户存在欠费时能不能直接办理销户？",
            "history": [],
            "category": "退订销户",
            "question_type": "关联推理",
            "expected_intent": "退订销户",
            "expected_retrieval_modes": ["graph", "hybrid"],
            "expect_references": True,
            "expect_relation_paths": True,
            "expected_source_files": source_by_category.get("退订销户", []) + source_by_category.get("合规审核", []),
            "source_category": "退订销户",
        },
        {
            "id": "q_relation_fault_sla",
            "question": "专线中断的故障等级会影响报修时限吗？",
            "history": [],
            "category": "故障报修",
            "question_type": "关联推理",
            "expected_intent": "故障报修",
            "expected_retrieval_modes": ["graph", "hybrid"],
            "expect_references": True,
            "expect_relation_paths": True,
            "expected_source_files": source_by_category.get("故障报修", []) + source_by_category.get("资费咨询", []),
            "source_category": "故障报修",
        },
    ]


def _context_followup_question() -> dict[str, Any]:
    return {
        "id": "q_context_followup_dynamic",
        "question": "如果是地址迁移呢？",
        "history": [
            {"role": "user", "content": "企业客户办理政企专线需要提供哪些材料？"},
            {"role": "assistant", "content": "需要依据客户准入规范核验企业客户材料。"},
        ],
        "category": "业务变更",
        "question_type": "多轮追问",
        "expected_intent": "业务变更",
        "expected_retrieval_modes": ["graph", "hybrid"],
        "expect_references": True,
        "expect_relation_paths": True,
        "expected_source_files": [],
        "source_category": "业务变更",
    }


def _payload(mode: str, questions: list[dict[str, Any]], category_payload: dict[str, Any]) -> dict[str, Any]:
    questions = _dedupe_questions(questions)[:MAX_QUESTIONS]
    return {
        "question_generation": {
            "mode": mode,
            "fixed": False,
            "gate_eligible": False,
            "category_count": len(category_payload.get("items", []) or []),
            "question_count": len(questions),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "questions": questions,
    }


def _sample_chunks(chunks: list[dict]) -> list[dict]:
    by_category: dict[str, dict] = {}
    for chunk in chunks:
        category = chunk.get("category", "")
        if category and category not in by_category:
            by_category[category] = chunk
        if len(by_category) >= MAX_QUESTIONS:
            break
    return list(by_category.values()) or chunks[:MAX_QUESTIONS]


def _load_sample_chunks() -> list[dict]:
    try:
        return _sample_chunks(search_all_chunks())
    except Exception:  # noqa: BLE001
        return []


def _relation_summaries() -> list[dict]:
    return [
        {
            "from": item.get("from", ""),
            "relation": item.get("relation", ""),
            "to": item.get("to", ""),
            "description": item.get("description", ""),
        }
        for item in RELATIONS
    ]


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _normalize_history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict) and item.get("content"):
            result.append({"role": str(item.get("role") or "user"), "content": str(item.get("content"))})
    return result


def _normalize_modes(value: Any) -> list[str]:
    allowed = {"vector", "graph", "hybrid"}
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ["vector"]
    result = [str(item) for item in value if str(item) in allowed]
    return result or ["vector"]


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _category_names(category_payload: dict[str, Any]) -> list[str]:
    names = [item.get("name", "") for item in category_payload.get("items", []) if item.get("name")]
    return names or list(category_payload.get("categories", []) or DEFAULT_CATEGORIES)


def _closest_category(value: str, categories: list[str]) -> str:
    if not categories:
        return DEFAULT_CATEGORIES[0]
    if not value:
        return categories[0]
    for category in categories:
        if category in value or value in category:
            return category
    return categories[0]


def _dedupe_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in questions:
        question = item.get("question", "")
        if not question or question in seen:
            continue
        seen.add(question)
        result.append(item)
    return result


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", value, flags=re.UNICODE).strip("_") or "category"
