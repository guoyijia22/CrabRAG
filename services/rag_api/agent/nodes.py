from __future__ import annotations

import json
import re

from services.rag_api.agent.business_scope import check_business_scope
from services.rag_api.agent.prompts import TOOL_CHOICE_PROMPT, build_answer_prompt, build_classify_prompt
from services.rag_api.agent.state import QAState
from services.rag_api.agent.tools import VALID_TOOLS, dispatch_retrieval, tool_to_mode
from services.rag_api.app_settings import load_app_settings
from services.rag_api.document.categories import DEFAULT_CATEGORIES, get_category_names
from services.rag_api.exceptions import LLM_ERROR_MESSAGE, LLMServiceError, NO_MATCH_MESSAGE
from services.rag_api.graph.graph_search import extract_entities
from services.rag_api.llm.siliconflow_client import chat_completion
from services.rag_api.rag_settings import get_retrieval_top_k, load_rag_settings
from services.rag_api.retrieval.optimizations import rewrite_query_with_context

QUESTION_TYPES = ["单一规则", "流程步骤", "标准查询", "资费标准", "关联推理"]
GRAPH_CORE_INTENTS = ["业务变更", "合规审核", "故障报修", "退订销户"]
FALLBACK_ANSWER = NO_MATCH_MESSAGE


def classify_intent_node(state: QAState) -> QAState:
    question = state["question"]
    history = state.get("history", [])
    categories = get_category_names()
    rag_settings = load_rag_settings()
    effective_question, rewrite_trace = rewrite_query_with_context(question, history, rag_settings)
    trace = state.get("trace", []) + [{"node": "rag_settings", "output": rag_settings.model_dump()}]
    trace.append({"node": "context_rewrite", "output": rewrite_trace})

    scope = check_business_scope(effective_question, categories)
    trace.append({"node": "business_scope_check", "output": scope})
    if not scope["in_scope"]:
        return {
            **state,
            "effective_question": effective_question,
            "intent": "业务外",
            "question_type": "业务外",
            "retrieval_mode": "none",
            "entities": scope.get("matched_entities", []),
            "business_scope": scope,
            "trace": trace,
        }

    try:
        content = chat_completion(
            [
                {"role": "system", "content": build_classify_prompt(categories)},
                {"role": "user", "content": f"历史上下文：{json.dumps(history, ensure_ascii=False)}\n用户问题：{effective_question}"},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        parsed = _parse_json(content)
    except Exception:
        parsed = _heuristic_classify(effective_question, history, categories)
    parsed = _normalize_classification(parsed, effective_question, categories)
    trace.append(
        {
            "node": "classify_intent",
            "output": {
                "intent": parsed["intent"],
                "retrieval_mode": parsed["retrieval_mode"],
                "entities": parsed["entities"],
                "available_categories": categories,
            },
        }
    )
    return {**state, **parsed, "effective_question": effective_question, "business_scope": scope, "trace": trace}


def choose_retrieval_tool_node(state: QAState) -> QAState:
    if not state.get("business_scope", {}).get("in_scope", True):
        trace = state.get("trace", []) + [{"node": "agent_tool_choice", "output": {"selected_tool": "none", "retrieval_mode": "none", "reason": "查询范围外，跳过检索"}}]
        return {**state, "selected_tool": "none", "tool_choice_reason": "查询范围外，跳过检索", "retrieval_mode": "none", "trace": trace}
    try:
        content = chat_completion(
            [
                {"role": "system", "content": TOOL_CHOICE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": state.get("effective_question", state["question"]),
                            "intent": state.get("intent", ""),
                            "question_type": state.get("question_type", ""),
                            "candidate_retrieval_mode": state.get("retrieval_mode", ""),
                            "entities": state.get("entities", []),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )
        parsed = _parse_json(content)
        selected_tool = parsed.get("tool", "")
        reason = parsed.get("reason", "LLM 工具选择")
        if selected_tool not in VALID_TOOLS:
            raise ValueError("invalid tool")
    except Exception:
        selected_tool, reason = _heuristic_tool_choice(state)
    retrieval_mode = tool_to_mode(selected_tool)
    trace = state.get("trace", []) + [{"node": "agent_tool_choice", "output": {"selected_tool": selected_tool, "retrieval_mode": retrieval_mode, "reason": reason}}]
    return {**state, "selected_tool": selected_tool, "tool_choice_reason": reason, "retrieval_mode": retrieval_mode, "trace": trace}


def retrieve_node(state: QAState) -> QAState:
    top_k = get_retrieval_top_k()
    if not state.get("business_scope", {}).get("in_scope", True):
        trace = state.get("trace", []) + [{"node": "retrieve", "output": {"top_k": top_k, "mode": "none", "sources": []}}]
        return {**state, "retrieved_chunks": [], "relation_paths": [], "references": [], "trace": trace, "error": None}
    selected_tool = state.get("selected_tool") or _heuristic_tool_choice(state)[0]
    query = state.get("effective_question", state["question"])
    result = dispatch_retrieval(query, state["intent"], state.get("entities", []), selected_tool)
    chunks = result.get("chunks", [])[:top_k]
    trace = state.get("trace", []) + result.get("trace", [])
    trace.append({"node": "retrieve", "output": {"top_k": top_k, "mode": result.get("mode", ""), "sources": [chunk.get("source_file", "") for chunk in chunks]}})
    return {**state, "retrieved_chunks": chunks, "relation_paths": result.get("relation_paths", [])[:top_k], "references": chunks, "trace": trace, "error": result.get("error")}


def generate_answer_node(state: QAState) -> QAState:
    app_settings = load_app_settings()
    top_k = get_retrieval_top_k()
    if not state.get("business_scope", {}).get("in_scope", True):
        trace = state.get("trace", []) + [{"node": "generate_answer", "output": {"has_references": False, "response_type": "out_of_scope"}}]
        return {**state, "answer": app_settings.out_of_scope_response, "references": [], "trace": trace, "error": None}

    chunks = state.get("retrieved_chunks", [])[:top_k]
    if not chunks:
        trace = state.get("trace", []) + [{"node": "generate_answer", "output": {"has_references": False, "response_type": "no_match"}}]
        return {**state, "answer": app_settings.no_match_response, "references": [], "trace": trace, "error": state.get("error")}
    try:
        prompt = build_answer_prompt().format(
            question=state.get("effective_question", state["question"]),
            retrieved_chunks=json.dumps(chunks, ensure_ascii=False),
            relation_paths=json.dumps(state.get("relation_paths", []), ensure_ascii=False),
        )
        answer = chat_completion([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1200)
        if "【参考规范原文片段】" not in answer:
            answer = _format_answer_from_chunks(state, answer)
        error = None
    except LLMServiceError:
        answer = LLM_ERROR_MESSAGE
        error = LLM_ERROR_MESSAGE
    trace = state.get("trace", []) + [{"node": "generate_answer", "output": {"has_references": bool(chunks), "reference_count": len(chunks), "response_type": "normal"}}]
    return {**state, "answer": answer, "references": chunks, "trace": trace, "error": error}


def _parse_json(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError("missing json")
    return json.loads(match.group(0))


def _normalize_classification(parsed: dict, question: str, categories: list[str]) -> dict:
    fallback = _heuristic_classify(question, [], categories)
    intent = parsed.get("intent") if parsed.get("intent") in categories else _closest_category(parsed.get("intent", ""), categories) or fallback["intent"]
    question_type = parsed.get("question_type") if parsed.get("question_type") in QUESTION_TYPES else fallback["question_type"]
    retrieval_mode = parsed.get("retrieval_mode") if parsed.get("retrieval_mode") in ["vector", "graph", "hybrid"] else fallback["retrieval_mode"]
    entities = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
    for entity in extract_entities(question):
        if entity not in entities:
            entities.append(entity)
    if intent not in entities and intent in GRAPH_CORE_INTENTS:
        entities.append(intent)
    return {"intent": intent, "question_type": question_type, "retrieval_mode": retrieval_mode, "entities": _dedupe(entities)}


def _heuristic_classify(question: str, history: list[dict], categories: list[str] | None = None) -> dict:
    categories = categories or get_category_names()
    expanded = question
    if any(word in question for word in ["这个", "那这种情况", "如果是这样", "如果是"]) and history:
        expanded = f"{history[-1].get('content', '')} {question}"
    candidates = [
        ("业务变更", "关联推理", "hybrid", ["地址迁移", "带宽变更", "移机", "变更"], ["合规审核"]),
        ("退订销户", "关联推理", "hybrid", ["欠费", "销户", "退订", "合同终止"], ["合规审核"]),
        ("故障报修", "关联推理", "hybrid", ["故障", "中断", "报修", "修复"], []),
        ("资费咨询", "资费标准", "vector", ["资费", "套餐", "月费", "多少钱", "价格"], []),
        ("办理流程", "流程步骤", "vector", ["流程", "步骤", "申请", "开通", "受理"], []),
        ("合规审核", "单一规则", "vector", ["一票否决", "合规", "审核", "红线"], []),
        ("公司法律法规", "单一规则", "vector", ["公司法", "股东", "董事", "法定代表人"], []),
        ("投诉处理", "流程步骤", "vector", ["投诉", "申诉", "争议", "处理时限"], []),
        ("客户准入", "单一规则", "vector", ["客户", "材料", "资质", "营业执照", "授权"], []),
    ]
    for intent, question_type, retrieval_mode, keywords, extra_entities in candidates:
        if any(word in expanded for word in keywords):
            normalized_intent = intent if intent in categories else _closest_category(intent, categories) or (categories[0] if categories else DEFAULT_CATEGORIES[0])
            return {
                "intent": normalized_intent,
                "question_type": question_type,
                "retrieval_mode": retrieval_mode,
                "entities": extract_entities(expanded) + extra_entities,
            }
    default_intent = "客户准入" if "客户准入" in categories else (categories[0] if categories else DEFAULT_CATEGORIES[0])
    return {"intent": default_intent, "question_type": "单一规则", "retrieval_mode": "vector", "entities": extract_entities(expanded)}


def _heuristic_tool_choice(state: QAState) -> tuple[str, str]:
    question = state.get("effective_question", state.get("question", ""))
    question_type = state.get("question_type", "")
    intent = state.get("intent", "")
    if intent not in GRAPH_CORE_INTENTS and question_type in ["单一规则", "流程步骤", "资费标准", "标准查询"]:
        return "vector_rule_search", "启发式兜底：扩展类别或单一规则适合向量检索"
    if any(word in question for word in ["地址迁移", "带宽变更", "欠费", "合同", "设备回收", "故障等级", "时限"]):
        return "hybrid_search", "启发式兜底：问题涉及关系推理且需要原文证据"
    if question_type in ["资费标准", "流程步骤", "单一规则", "标准查询"]:
        return "vector_rule_search", "启发式兜底：单一规则、流程或标准查询"
    return "graph_relation_search_tool", "启发式兜底：关联推理问题"


def _format_answer_from_chunks(state: QAState, answer: str) -> str:
    refs = "\n\n".join(
        f"{i}. 来源：《{chunk.get('source_file', '')}》\n原文片段：{chunk.get('content', '')}"
        for i, chunk in enumerate(state.get("retrieved_chunks", [])[:get_retrieval_top_k()], start=1)
    )
    return f"【业务类别】\n{state.get('intent', '')}\n\n【答复】\n{answer.strip()}\n\n【合规提示】\n仅依据当前检索到的原文片段作答。\n\n【参考规范原文片段】\n{refs}"


def _closest_category(value: str, categories: list[str]) -> str:
    if not value:
        return ""
    for category in categories:
        if value in category or category in value:
            return category
    return ""


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
