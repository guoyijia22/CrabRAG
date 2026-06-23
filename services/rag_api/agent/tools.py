from __future__ import annotations

from services.rag_api.graph.graph_search import graph_relation_search
from services.rag_api.rag_settings import get_retrieval_top_k, load_rag_settings
from services.rag_api.retrieval.optimizations import apply_query_expansion, apply_rerank, keyword_search_candidates, rrf_merge
from services.rag_api.vector.chroma_store import search_all_chunks, search_chunks

VALID_TOOLS = {"vector_rule_search", "graph_relation_search_tool", "hybrid_search"}


def vector_rule_search(query: str, intent: str, entities: list[str]) -> dict:
    rag_settings = load_rag_settings()
    top_k = get_retrieval_top_k(rag_settings)
    trace: list[dict] = []
    try:
        queries, expansion_trace = apply_query_expansion(query, rag_settings)
        trace.append({"node": "query_expansion", "output": expansion_trace})
        vector_results: list[dict] = []
        for expanded_query in queries:
            vector_results.extend(search_chunks(expanded_query, intent, entities, top_k=top_k))
        candidate_k = max(top_k * 4, rag_settings.vector_candidate_k if rag_settings.rag_param_tuning_enabled else top_k * 4)
        vector_results = _merge_chunks(vector_results, top_k=candidate_k)
        if rag_settings.hybrid_bm25_enabled:
            keyword_results = keyword_search_candidates(query, search_all_chunks(), rag_settings, limit=max(top_k * 4, rag_settings.vector_candidate_k))
            trace.append({"node": "hybrid_bm25", "output": {"enabled": True, "keyword_candidates": len(keyword_results), "vector_candidates": len(vector_results)}})
            vector_results = rrf_merge(vector_results, keyword_results, rag_settings, top_k=max(top_k * 4, rag_settings.vector_candidate_k))
        else:
            trace.append({"node": "hybrid_bm25", "output": {"enabled": False}})
        reranked, rerank_trace = apply_rerank(query, vector_results, rag_settings, top_k=top_k)
        trace.append({"node": "rerank", "output": rerank_trace})
        return {"mode": "vector", "chunks": reranked[:top_k], "relation_paths": [], "error": None, "trace": trace}
    except Exception as exc:  # noqa: BLE001
        return {"mode": "vector", "chunks": [], "relation_paths": [], "error": str(exc), "trace": trace}


def graph_relation_search_tool(query: str, intent: str, entities: list[str]) -> dict:
    top_k = get_retrieval_top_k()
    try:
        result = graph_relation_search(query, intent, top_k=top_k)
        return {"mode": "graph", "chunks": result["chunks"][:top_k], "relation_paths": result["relation_paths"][:top_k], "error": None, "trace": []}
    except Exception as exc:  # noqa: BLE001
        return {"mode": "graph", "chunks": [], "relation_paths": [], "error": str(exc), "trace": []}


def dispatch_retrieval(query: str, intent: str, entities: list[str], selected_tool: str) -> dict:
    top_k = get_retrieval_top_k()
    if selected_tool == "vector_rule_search":
        result = vector_rule_search(query, intent, entities)
    elif selected_tool == "graph_relation_search_tool":
        result = graph_relation_search_tool(query, intent, entities)
    else:
        graph_result = graph_relation_search_tool(query, intent, entities)
        vector_result = vector_rule_search(query, intent, entities)
        result = {
            "mode": "hybrid",
            "chunks": _merge_chunks(graph_result["chunks"] + vector_result["chunks"], top_k=top_k),
            "relation_paths": graph_result["relation_paths"][:top_k],
            "error": graph_result["error"] or vector_result["error"],
            "trace": vector_result.get("trace", []),
        }
    result["chunks"] = result.get("chunks", [])[:top_k]
    result["relation_paths"] = result.get("relation_paths", [])[:top_k]
    return result


def tool_to_mode(tool: str) -> str:
    return {"vector_rule_search": "vector", "graph_relation_search_tool": "graph", "hybrid_search": "hybrid"}.get(tool, "hybrid")


def _merge_chunks(chunks: list[dict], top_k: int) -> list[dict]:
    deduped: dict[str, dict] = {}
    for chunk in chunks:
        key = f"{chunk.get('source_file')}::{chunk.get('content', '')[:120]}"
        if key not in deduped or chunk.get("score", 0) > deduped[key].get("score", 0):
            deduped[key] = chunk
    return sorted(deduped.values(), key=lambda item: item.get("score", 0), reverse=True)[:top_k]
