from __future__ import annotations

import re
from typing import Any

import requests

from services.rag_api.config import get_settings
from services.rag_api.llm import local_onnx_rerank
from services.rag_api.llm.siliconflow_client import chat_completion
from services.rag_api.rag_settings import RagSettings


def rewrite_query_with_context(query: str, history: list[dict[str, str]], settings: RagSettings) -> tuple[str, dict[str, Any]]:
    trace = {"enabled": settings.context_rewrite_enabled, "fallback": False, "rewritten_query": query}
    if not settings.context_rewrite_enabled or not history:
        return query, trace
    history_text = "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in history[-6:])
    prompt = (
        "根据对话历史和当前问题，生成一个独立的、包含所有必要信息的完整查询。"
        "只输出完整查询，不要解释。\n\n"
        f"对话历史：\n{history_text}\n\n当前问题：{query}\n\n完整查询："
    )
    try:
        rewritten = chat_completion([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300).strip()
        if rewritten:
            trace["rewritten_query"] = rewritten
            return rewritten, trace
    except Exception:
        trace["fallback"] = True
    return query, trace


def apply_query_expansion(query: str, settings: RagSettings) -> tuple[list[str], dict[str, Any]]:
    trace = {"enabled": settings.query_expansion_enabled, "fallback": False, "queries": [query]}
    if not settings.query_expansion_enabled:
        return [query], trace
    prompt = (
        "请根据用户原始问题，生成最多3个语义相近但表述不同的扩展查询。"
        "每个扩展查询单独一行，不要解释。\n\n"
        f"原始问题：{query}"
    )
    try:
        content = chat_completion([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
        expanded = [line.strip(" -\t\r\n") for line in content.splitlines() if line.strip()]
        queries = _dedupe([query] + expanded[:3])
        trace["queries"] = queries
        return queries, trace
    except Exception:
        trace["fallback"] = True
        return [query], trace


def keyword_search_candidates(query: str, chunks: list[dict], settings: RagSettings, limit: int) -> list[dict]:
    terms = _tokenize(query)
    candidates: list[dict] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        term_hits = sum(content.count(term) for term in terms if term)
        if term_hits <= 0:
            continue
        score = min(1.0, 0.35 + term_hits * 0.08)
        candidates.append({**chunk, "score": round(score, 4), "retrieval_channel": "bm25_keyword"})
    candidates.sort(key=lambda item: item.get("score", 0), reverse=True)
    return candidates[:limit]


def apply_rerank(query: str, chunks: list[dict], settings: RagSettings, top_k: int) -> tuple[list[dict], dict[str, Any]]:
    trace = {"enabled": settings.rerank_enabled, "provider": settings.rerank_provider, "fallback": False, "candidate_count": len(chunks)}
    if not settings.rerank_enabled or not chunks:
        return chunks[:top_k], trace
    if settings.rerank_provider == "local_onnx":
        return _apply_local_rerank(query, chunks, top_k, trace)
    app_settings = get_settings()
    url = f"{app_settings.rerank_base_url.rstrip('/')}/rerank"
    headers = {"Authorization": f"Bearer {app_settings.rerank_api_key or 'local'}", "Content-Type": "application/json"}
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": [chunk.get("content", "") for chunk in chunks],
        "top_n": min(top_k, len(chunks)),
        "return_documents": True,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=app_settings.request_timeout)
        response.raise_for_status()
        data = response.json()
        reranked: list[dict] = []
        for result in data.get("results", []):
            index = result.get("index", 0)
            if 0 <= index < len(chunks):
                reranked.append({**chunks[index], "score": result.get("relevance_score", chunks[index].get("score", 0)), "rerank_score": result.get("relevance_score")})
        if reranked:
            trace["returned_count"] = len(reranked)
            return reranked[:top_k], trace
    except Exception:
        trace["fallback"] = True
    return chunks[:top_k], trace


def _apply_local_rerank(query: str, chunks: list[dict], top_k: int, trace: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    app_settings = get_settings()
    documents = [chunk.get("content", "") for chunk in chunks]
    try:
        results = local_onnx_rerank.rerank_documents_local(
            query,
            documents,
            app_settings.local_rerank_model_dir,
            min(top_k, len(chunks)),
            app_settings.rerank_onnx_model_file,
        )
        reranked: list[dict] = []
        for result in results:
            index = int(result.get("index", 0))
            score = result.get("relevance_score")
            if 0 <= index < len(chunks):
                reranked.append({**chunks[index], "score": score if score is not None else chunks[index].get("score", 0), "rerank_score": score})
        if reranked:
            reranked = reranked[:top_k]
            trace["returned_count"] = len(reranked)
            return reranked, trace
    except Exception as exc:  # noqa: BLE001
        trace["fallback"] = True
        trace["error"] = str(exc)
    return chunks[:top_k], trace


def _tokenize(query: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", query)
    extra = [term for term in ["欠费", "地址迁移", "带宽变更", "一票否决", "中断", "报修", "销户", "资费", "材料", "审核"] if term in query]
    return _dedupe(terms + extra)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
