from __future__ import annotations

import json
import re

from services.rag_api.agent.business_scope import check_business_scope
from services.rag_api.agent.heuristics import (
    GRAPH_CORE_INTENTS,
    QUESTION_TYPES,
    closest_category as _closest_category,
    dedupe as _dedupe,
    heuristic_classify as _heuristic_classify,
    heuristic_tool_choice as _heuristic_tool_choice,
)
from services.rag_api.agent.prompts import build_answer_prompt, build_classify_prompt, build_tool_choice_prompt, detect_prompt_language
from services.rag_api.agent.rag_context import build_rag_context
from services.rag_api.agent.state import QAState
from services.rag_api.agent.tools import VALID_TOOLS, dispatch_retrieval, tool_to_mode
from services.rag_api.app_settings import load_app_settings
from services.rag_api.document.categories import get_category_names
from services.rag_api.exceptions import LLM_ERROR_MESSAGE, LLMServiceError, NO_MATCH_MESSAGE
from services.rag_api.graph.graph_search import extract_entities
from services.rag_api.llm.siliconflow_client import chat_completion
from services.rag_api.rag_settings import get_retrieval_top_k, load_rag_settings
from services.rag_api.retrieval.context_budget import apply_context_token_budget
from services.rag_api.retrieval.optimizations import rewrite_query_with_context

FALLBACK_ANSWER = NO_MATCH_MESSAGE


def classify_intent_node(state: QAState) -> QAState:
    question = state["question"]
    history = state.get("history", [])
    categories = get_category_names()
    rag_settings = load_rag_settings()
    effective_question, rewrite_trace = rewrite_query_with_context(question, history, rag_settings)
    prompt_language = detect_prompt_language(effective_question)
    trace = state.get("trace", []) + [{"node": "rag_settings", "output": rag_settings.model_dump()}]
    trace.append({"node": "prompt_language", "output": {"language": prompt_language}})
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
                {"role": "system", "content": build_classify_prompt(categories, language=prompt_language)},
                {"role": "user", "content": _classification_user_content(history, effective_question, prompt_language)},
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
        prompt_language = detect_prompt_language(state.get("effective_question", state["question"]))
        content = chat_completion(
            [
                {"role": "system", "content": build_tool_choice_prompt(prompt_language)},
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
    rag_settings = load_rag_settings()
    top_k = get_retrieval_top_k(rag_settings)
    if not state.get("business_scope", {}).get("in_scope", True):
        trace = state.get("trace", []) + [{"node": "generate_answer", "output": {"has_references": False, "response_type": "out_of_scope"}}]
        return {**state, "answer": app_settings.out_of_scope_response, "references": [], "trace": trace, "error": None}

    chunks = state.get("retrieved_chunks", [])[:top_k]
    if not chunks:
        trace = state.get("trace", []) + [{"node": "generate_answer", "output": {"has_references": False, "response_type": "no_match"}}]
        return {**state, "answer": app_settings.no_match_response, "references": [], "trace": trace, "error": state.get("error")}
    relation_paths = state.get("relation_paths", [])[:top_k]
    chunks, relation_paths, budget_trace = apply_context_token_budget(
        state.get("effective_question", state["question"]),
        chunks,
        relation_paths,
        rag_settings,
    )
    prompt_language = detect_prompt_language(state.get("effective_question", state["question"]))
    rag_context = build_rag_context(chunks, relation_paths)
    trace = state.get("trace", []) + [
        {"node": "context_token_budget", "output": budget_trace},
        {
            "node": "rag_context_builder",
            "output": {
                "prompt_language": prompt_language,
                "chunk_count": len(chunks),
                "relation_path_count": len(relation_paths),
                "context_length": len(rag_context),
            },
        },
    ]
    try:
        system_prompt = build_answer_prompt(language=prompt_language, context_data=rag_context)
        answer = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state.get("effective_question", state["question"])},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        if not _answer_has_reference_section(answer, prompt_language):
            answer = _format_answer_from_chunks({**state, "retrieved_chunks": chunks}, answer, language=prompt_language)
        error = None
    except LLMServiceError:
        answer = LLM_ERROR_MESSAGE
        error = LLM_ERROR_MESSAGE
    trace.append({"node": "generate_answer", "output": {"has_references": bool(chunks), "reference_count": len(chunks), "response_type": "normal"}})
    return {**state, "answer": answer, "retrieved_chunks": chunks, "relation_paths": relation_paths, "references": chunks, "trace": trace, "error": error}


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


def _classification_user_content(history: list[dict[str, str]], question: str, language: str) -> str:
    if language == "en":
        return f"Conversation history: {json.dumps(history, ensure_ascii=False)}\nUser question: {question}"
    return f"历史上下文：{json.dumps(history, ensure_ascii=False)}\n用户问题：{question}"


def _answer_has_reference_section(answer: str, language: str) -> bool:
    return "### References" in answer if language == "en" else "【参考知识库原文片段】" in answer


def _format_answer_from_chunks(state: QAState, answer: str, language: str | None = None) -> str:
    question = state.get("effective_question") or state.get("question") or ""
    active_language = language or (detect_prompt_language(question) if question else "zh")
    chunks = state.get("retrieved_chunks", [])[:get_retrieval_top_k()]
    if active_language == "en":
        refs = "\n".join(
            f"- [{i}] {chunk.get('source_file', '')}"
            for i, chunk in enumerate(chunks, start=1)
        )
        return f"## Category\n{state.get('intent', '')}\n\n## Answer\n{answer.strip()}\n\n### References\n{refs}"
    refs = "\n\n".join(
        f"{i}. 来源：《{chunk.get('source_file', '')}》\n原文片段：{chunk.get('content', '')}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return f"【业务类别】\n{state.get('intent', '')}\n\n【答复】\n{answer.strip()}\n\n【参考知识库原文片段】\n{refs}"
