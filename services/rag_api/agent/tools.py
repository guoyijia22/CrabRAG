from __future__ import annotations

from services.rag_api.graph.graph_search import graph_relation_search
from services.rag_api.rag_settings import get_retrieval_top_k, load_rag_settings
from services.rag_api.retrieval.optimizations import apply_query_expansion, apply_rerank, keyword_search_candidates
from services.rag_api.vector.chroma_store import search_all_chunks, search_chunks

VALID_TOOLS = {"vector_rule_search", "graph_relation_search_tool", "hybrid_search"}


def vector_rule_search(
    query: str,
    intent: str,
    entities: list[str],
    *,
    allow_query_expansion: bool = True,
    allow_rerank: bool = True,
    allow_keyword_search: bool = True,
) -> dict:
    rag_settings = load_rag_settings()
    top_k = get_retrieval_top_k(rag_settings)
    trace: list[dict] = []
    try:
        if allow_query_expansion:
            queries, expansion_trace = apply_query_expansion(query, rag_settings)
        else:
            queries = [query]
            expansion_trace = {"enabled": False, "fallback": False, "queries": [query], "reason": "disabled_for_evidence_cli"}
        trace.append({"node": "query_expansion", "output": expansion_trace})
        vector_results: list[dict] = []
        for expanded_query in queries:
            vector_results.extend(search_chunks(expanded_query, intent, entities, top_k=top_k))
        candidate_k = _candidate_count(rag_settings, top_k)
        vector_results = _merge_chunks(vector_results, top_k=candidate_k)
        if not allow_keyword_search:
            trace.append({"node": "hybrid_bm25", "output": {"enabled": False, "reason": "keyword_stream_used_by_hybrid_round_robin"}})
        else:
            trace.append(
                {
                    "node": "hybrid_bm25",
                    "output": {
                        "enabled": False,
                        "deprecated": True,
                        "reason": "keyword_stream_is_builtin_in_hybrid_round_robin",
                    },
                }
            )
        if allow_rerank:
            reranked, rerank_trace = apply_rerank(query, vector_results, rag_settings, top_k=top_k)
        else:
            reranked = vector_results[:top_k]
            rerank_trace = {"enabled": False, "provider": rag_settings.rerank_provider, "fallback": False, "candidate_count": len(vector_results), "reason": "disabled_by_cli_flag"}
        trace.append({"node": "rerank", "output": rerank_trace})
        return {"mode": "vector", "chunks": reranked[:top_k], "relation_paths": [], "error": None, "trace": trace}
    except Exception as exc:  # noqa: BLE001
        return {"mode": "vector", "chunks": [], "relation_paths": [], "error": str(exc), "trace": trace}


def graph_relation_search_tool(query: str, intent: str, entities: list[str]) -> dict:
    top_k = get_retrieval_top_k()
    try:
        result = graph_relation_search(query, intent, top_k=top_k)
        return {
            "mode": "graph",
            "chunks": result["chunks"][:top_k],
            "relation_paths": result["relation_paths"][:top_k],
            "error": None,
            "trace": result.get("trace", []),
        }
    except Exception as exc:  # noqa: BLE001
        return {"mode": "graph", "chunks": [], "relation_paths": [], "error": str(exc), "trace": []}


def dispatch_retrieval(query: str, intent: str, entities: list[str], selected_tool: str, *, allow_query_expansion: bool = True, allow_rerank: bool = True) -> dict:
    top_k = get_retrieval_top_k()
    if selected_tool == "vector_rule_search":
        result = vector_rule_search(query, intent, entities, allow_query_expansion=allow_query_expansion, allow_rerank=allow_rerank)
    elif selected_tool == "graph_relation_search_tool":
        result = graph_relation_search_tool(query, intent, entities)
    else:
        graph_result = graph_relation_search_tool(query, intent, entities)
        vector_result = vector_rule_search(
            query,
            intent,
            entities,
            allow_query_expansion=allow_query_expansion,
            allow_rerank=False,
            allow_keyword_search=False,
        )
        rag_settings = load_rag_settings()
        candidate_k = _candidate_count(rag_settings, top_k)
        keyword_error = None
        try:
            keyword_results = keyword_search_candidates(query, search_all_chunks(), rag_settings, limit=candidate_k)
        except Exception as exc:  # noqa: BLE001
            keyword_results = []
            keyword_error = str(exc)
        merged_candidates = _round_robin_merge_chunk_streams(
            [
                ("vector", vector_result["chunks"]),
                ("graph", graph_result["chunks"]),
                ("keyword", keyword_results),
            ],
            top_k=candidate_k,
        )
        trace = vector_result.get("trace", []) + graph_result.get("trace", []) + [
            {
                "node": "hybrid_round_robin",
                "output": {
                    "vector_candidates": len(vector_result.get("chunks", [])),
                    "graph_candidates": len(graph_result.get("chunks", [])),
                    "keyword_candidates": len(keyword_results),
                    "merged_candidates": len(merged_candidates),
                    "keyword_error": keyword_error,
                },
            }
        ]
        if allow_rerank:
            chunks, rerank_trace = apply_rerank(query, merged_candidates, rag_settings, top_k=top_k)
        else:
            chunks = merged_candidates[:top_k]
            rerank_trace = {
                "enabled": False,
                "provider": rag_settings.rerank_provider,
                "fallback": False,
                "candidate_count": len(merged_candidates),
                "reason": "disabled_by_cli_flag",
            }
        trace.append({"node": "hybrid_rerank", "output": rerank_trace})
        result = {
            "mode": "hybrid",
            "chunks": chunks,
            "relation_paths": graph_result["relation_paths"][:top_k],
            "error": graph_result["error"] or vector_result["error"],
            "trace": trace,
        }
    result["chunks"] = result.get("chunks", [])[:top_k]
    result["relation_paths"] = result.get("relation_paths", [])[:top_k]
    return result


def tool_to_mode(tool: str) -> str:
    return {"vector_rule_search": "vector", "graph_relation_search_tool": "graph", "hybrid_search": "hybrid"}.get(tool, "hybrid")


def _candidate_count(settings, top_k: int) -> int:
    return max(top_k * 4, int(getattr(settings, "vector_candidate_k", top_k * 4)))


def _merge_chunks(chunks: list[dict], top_k: int) -> list[dict]:
    deduped: dict[str, dict] = {}
    for chunk in chunks:
        key = f"{chunk.get('source_file')}::{chunk.get('content', '')[:120]}"
        if key not in deduped or chunk.get("score", 0) > deduped[key].get("score", 0):
            deduped[key] = chunk
    return sorted(deduped.values(), key=lambda item: item.get("score", 0), reverse=True)[:top_k]


def _round_robin_merge_chunk_streams(streams: list[tuple[str, list[dict]]], top_k: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    max_len = max((len(chunks) for _, chunks in streams), default=0)
    for index in range(max_len):
        for _, chunks in streams:
            if index >= len(chunks):
                continue
            chunk = chunks[index]
            key = f"{chunk.get('source_file')}::{chunk.get('content', '')[:120]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= top_k:
                return merged
    return merged
