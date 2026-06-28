from __future__ import annotations

from services.rag_api.document.categories import DEFAULT_CATEGORIES, get_category_names
from services.rag_api.graph.graph_search import extract_entities

QUESTION_TYPES = ["单一规则", "流程步骤", "标准查询", "资费标准", "关联推理"]
GRAPH_CORE_INTENTS = ["业务变更", "合规审核", "故障报修", "退订销户"]


def heuristic_classify(question: str, history: list[dict], categories: list[str] | None = None) -> dict:
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
            normalized_intent = intent if intent in categories else closest_category(intent, categories) or (categories[0] if categories else DEFAULT_CATEGORIES[0])
            return {
                "intent": normalized_intent,
                "question_type": question_type,
                "retrieval_mode": retrieval_mode,
                "entities": extract_entities(expanded) + extra_entities,
            }
    default_intent = "客户准入" if "客户准入" in categories else (categories[0] if categories else DEFAULT_CATEGORIES[0])
    return {"intent": default_intent, "question_type": "单一规则", "retrieval_mode": "vector", "entities": extract_entities(expanded)}


def heuristic_tool_choice(state: dict) -> tuple[str, str]:
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


def closest_category(value: str, categories: list[str]) -> str:
    if not value:
        return ""
    for category in categories:
        if value in category or category in value:
            return category
    return ""


def dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
