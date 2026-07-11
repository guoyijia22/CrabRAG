from __future__ import annotations

from typing import Any, Literal

from services.rag_api.agent.business_scope import check_business_scope
from services.rag_api.agent.heuristics import heuristic_classify, heuristic_tool_choice
from services.rag_api.agent.tools import dispatch_retrieval, tool_to_mode
from services.rag_api.document.categories import get_category_names
from services.rag_api.llm.siliconflow_client import chat_completion  # noqa: F401
from services.rag_api.rag_settings import RagSettings, load_rag_settings, override_rag_settings, resolve_retrieval_top_k
from services.rag_api.security import PrincipalContext, build_retrieval_context, current_retrieval_context, use_retrieval_context

EvidenceMode = Literal["auto", "vector", "graph", "hybrid"]

MODE_TO_TOOL = {
    "vector": "vector_rule_search",
    "graph": "graph_relation_search_tool",
    "hybrid": "hybrid_search",
}

EVIDENCE_KEYS = [
    "content",
    "source_file",
    "source_path",
    "section_title",
    "category",
    "score",
    "rerank_score",
    "retrieval_channel",
]


def retrieve_evidence(
    question: str,
    *,
    top_k: int | None = None,
    mode: EvidenceMode = "auto",
    include_trace: bool = False,
    no_rerank: bool = False,
) -> dict[str, Any]:
    if current_retrieval_context() is not None:
        return _retrieve_evidence(
            question,
            top_k=top_k,
            mode=mode,
            include_trace=include_trace,
            no_rerank=no_rerank,
        )
    context = build_retrieval_context(PrincipalContext.anonymous())
    with use_retrieval_context(context):
        return _retrieve_evidence(
            question,
            top_k=top_k,
            mode=mode,
            include_trace=include_trace,
            no_rerank=no_rerank,
        )


def _retrieve_evidence(
    question: str,
    *,
    top_k: int | None = None,
    mode: EvidenceMode = "auto",
    include_trace: bool = False,
    no_rerank: bool = False,
) -> dict[str, Any]:
    effective_question = question.strip()
    if not effective_question:
        raise ValueError("question must not be empty")

    rag_settings = _settings_with_top_k(load_rag_settings(), top_k)
    top_k_decision = resolve_retrieval_top_k(effective_question, rag_settings)
    top_k_value = int(top_k_decision["effective_top_k"])
    categories = get_category_names()
    trace: list[dict[str, Any]] = [
        {"node": "rag_settings", "output": rag_settings.model_dump()},
        {"node": "context_rewrite", "output": {"enabled": False, "rewritten_query": effective_question, "reason": "disabled_for_evidence_cli"}},
    ]

    scope = check_business_scope(effective_question, categories)
    trace.append({"node": "business_scope_check", "output": scope})
    if not scope.get("in_scope", False):
        return _payload(
            question=question,
            effective_question=effective_question,
            retrieval_mode="none",
            evidence=[],
            relation_paths=[],
            warnings=["out_of_scope"],
            trace=trace,
            include_trace=include_trace,
            extra={"intent": "业务外", "question_type": "业务外", "entities": scope.get("matched_entities", []), "business_scope": scope},
        )

    classification = heuristic_classify(effective_question, [], categories)
    selected_tool, reason = _select_tool(mode, effective_question, classification)
    retrieval_mode = tool_to_mode(selected_tool)
    trace.append(
        {
            "node": "classify_intent",
            "output": {
                "intent": classification["intent"],
                "question_type": classification["question_type"],
                "retrieval_mode": classification["retrieval_mode"],
                "entities": classification["entities"],
                "strategy": "heuristic_only",
            },
        }
    )
    trace.append({"node": "agent_tool_choice", "output": {"selected_tool": selected_tool, "retrieval_mode": retrieval_mode, "reason": reason}})

    with override_rag_settings(rag_settings):
        result = dispatch_retrieval(
            effective_question,
            classification["intent"],
            classification.get("entities", []),
            selected_tool,
            allow_query_expansion=False,
            allow_rerank=not no_rerank,
        )

    chunks = result.get("chunks", [])[:top_k_value]
    relation_paths = result.get("relation_paths", [])[:top_k_value]
    trace.extend(result.get("trace", []))
    trace.append({"node": "retrieve", "output": {**top_k_decision, "top_k": top_k_value, "mode": result.get("mode", ""), "sources": [chunk.get("source_file", "") for chunk in chunks]}})

    error = result.get("error")
    if error and not chunks and not relation_paths:
        raise RuntimeError(error)

    warnings: list[str] = []
    if error:
        warnings.append("partial_retrieval_error")
    if not chunks:
        warnings.append("no_evidence")

    return _payload(
        question=question,
        effective_question=effective_question,
        retrieval_mode=result.get("mode", retrieval_mode),
        evidence=[_format_evidence(chunk) for chunk in chunks],
        relation_paths=relation_paths,
        warnings=warnings,
        trace=trace,
        include_trace=include_trace,
        extra={
            "intent": classification["intent"],
            "question_type": classification["question_type"],
            "entities": classification.get("entities", []),
            "selected_tool": selected_tool,
            "tool_choice_reason": reason,
        },
    )


def _settings_with_top_k(settings: RagSettings, top_k: int | None) -> RagSettings:
    if top_k is None:
        return settings
    return settings.model_copy(update={"top_k": max(1, min(10, top_k))})


def _select_tool(mode: EvidenceMode, question: str, classification: dict[str, Any]) -> tuple[str, str]:
    if mode != "auto":
        return MODE_TO_TOOL[mode], f"CLI 指定检索模式：{mode}"
    state = {
        "question": question,
        "effective_question": question,
        "intent": classification.get("intent", ""),
        "question_type": classification.get("question_type", ""),
        "retrieval_mode": classification.get("retrieval_mode", ""),
        "entities": classification.get("entities", []),
    }
    return heuristic_tool_choice(state)


def _payload(
    *,
    question: str,
    effective_question: str,
    retrieval_mode: str,
    evidence: list[dict[str, Any]],
    relation_paths: list[dict[str, Any]],
    warnings: list[str],
    trace: list[dict[str, Any]],
    include_trace: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "question": question,
        "effective_question": effective_question,
        "retrieval_mode": retrieval_mode,
        "evidence": evidence,
        "relation_paths": relation_paths,
        "warnings": warnings,
        **extra,
    }
    if include_trace:
        payload["trace"] = trace
    return payload


def _format_evidence(chunk: dict[str, Any]) -> dict[str, Any]:
    item = {key: chunk.get(key) for key in EVIDENCE_KEYS}
    item["content"] = item.get("content") or ""
    item["source_file"] = item.get("source_file") or ""
    item["source_path"] = item.get("source_path") or ""
    item["section_title"] = item.get("section_title") or ""
    item["category"] = item.get("category") or ""
    item["retrieval_channel"] = item.get("retrieval_channel") or ""
    return item
