from __future__ import annotations

from typing import Any

from services.rag_api.rag_settings import RagSettings


def apply_context_token_budget(
    question: str,
    chunks: list[dict[str, Any]],
    relation_paths: list[dict[str, Any]],
    settings: RagSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    max_tokens = max(100, int(settings.max_context_tokens or 6000))
    fixed_tokens = estimate_tokens(question) + 100
    relation_budget = max(40, min(int(max_tokens * 0.25), max_tokens - fixed_tokens))
    budgeted_paths, relation_used, relation_truncated = _fit_relation_paths(relation_paths, relation_budget)
    chunk_budget = max(60, max_tokens - fixed_tokens - relation_used)
    budgeted_chunks, chunk_used, chunk_truncated = _fit_chunks(chunks, chunk_budget)
    return (
        budgeted_chunks,
        budgeted_paths,
        {
            "max_context_tokens": max_tokens,
            "fixed_tokens": fixed_tokens,
            "relation_tokens": relation_used,
            "chunk_tokens": chunk_used,
            "input_chunk_count": len(chunks),
            "output_chunk_count": len(budgeted_chunks),
            "input_relation_path_count": len(relation_paths),
            "output_relation_path_count": len(budgeted_paths),
            "truncated": relation_truncated or chunk_truncated or len(budgeted_chunks) < len(chunks) or len(budgeted_paths) < len(relation_paths),
        },
    )


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars + max(1, other_chars // 4)


def _fit_relation_paths(paths: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], int, bool]:
    result: list[dict[str, Any]] = []
    used = 0
    truncated = False
    for path in paths:
        item = dict(path)
        cost = _relation_cost(item)
        remaining = budget - used
        if cost <= remaining:
            result.append(item)
            used += cost
            continue
        if not result and remaining > 20:
            item = _truncate_relation(item, remaining)
            result.append(item)
            used += _relation_cost(item)
            truncated = True
        break
    return result, used, truncated


def _fit_chunks(chunks: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], int, bool]:
    result: list[dict[str, Any]] = []
    used = 0
    truncated = False
    for chunk in chunks:
        item = dict(chunk)
        cost = estimate_tokens(str(item.get("content") or "")) + 16
        remaining = budget - used
        if cost <= remaining:
            result.append(item)
            used += cost
            continue
        if not result and remaining > 20:
            item["content"] = _truncate_text(str(item.get("content") or ""), max(1, remaining - 16))
            result.append(item)
            used += estimate_tokens(str(item.get("content") or "")) + 16
            truncated = True
        break
    return result, used, truncated


def _relation_cost(path: dict[str, Any]) -> int:
    return estimate_tokens(str(path.get("path") or "")) + estimate_tokens(str(path.get("description") or "")) + estimate_tokens(str(path.get("evidence") or "")) + 12


def _truncate_relation(path: dict[str, Any], budget: int) -> dict[str, Any]:
    result = dict(path)
    base = estimate_tokens(str(result.get("path") or "")) + 12
    remaining = max(1, budget - base)
    if result.get("description"):
        result["description"] = _truncate_text(str(result.get("description") or ""), remaining)
    elif result.get("evidence"):
        result["evidence"] = _truncate_text(str(result.get("evidence") or ""), remaining)
    return result


def _truncate_text(text: str, budget: int) -> str:
    if estimate_tokens(text) <= budget:
        return text
    output: list[str] = []
    used = 0
    for char in text:
        cost = estimate_tokens(char)
        if used + cost > max(1, budget - 1):
            break
        output.append(char)
        used += cost
    return "".join(output).rstrip() + "…"
