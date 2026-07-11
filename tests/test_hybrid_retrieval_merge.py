from __future__ import annotations

from services.rag_api.rag_settings import RagSettings


def _chunk(content: str, source_file: str, score: float, channel: str) -> dict:
    return {
        "content": content,
        "source_file": source_file,
        "score": score,
        "retrieval_channel": channel,
    }


def test_hybrid_search_round_robins_vector_graph_and_keyword_chunks(monkeypatch):
    from services.rag_api.agent import tools

    monkeypatch.setattr(tools, "get_retrieval_top_k", lambda settings=None: 6)
    monkeypatch.setattr(tools, "load_rag_settings", lambda: RagSettings(top_k=6, rerank_enabled=False))
    monkeypatch.setattr(
        tools,
        "vector_rule_search",
        lambda *args, **kwargs: {
            "mode": "vector",
            "chunks": [
                _chunk("vector-1", "vector-1.md", 0.99, "vector"),
                _chunk("vector-2", "vector-2.md", 0.98, "vector"),
                _chunk("vector-3", "vector-3.md", 0.97, "vector"),
            ],
            "relation_paths": [],
            "error": None,
            "trace": [{"node": "vector", "output": {"ok": True}}],
        },
    )
    monkeypatch.setattr(
        tools,
        "keyword_search_candidates",
        lambda query, chunks, settings, limit: [
            _chunk("keyword-1", "keyword-1.md", 0.7, "bm25_keyword"),
            _chunk("keyword-2", "keyword-2.md", 0.6, "bm25_keyword"),
        ],
    )
    monkeypatch.setattr(tools, "search_all_chunks", lambda: [{"content": "all"}])
    monkeypatch.setattr(
        tools,
        "graph_relation_search_tool",
        lambda *args, **kwargs: {
            "mode": "graph",
            "chunks": [
                _chunk("graph-1", "graph-1.md", 0.2, "keyword"),
                _chunk("graph-2", "graph-2.md", 0.1, "keyword"),
            ],
            "relation_paths": [{"path": "A -> relates -> B"}],
            "error": None,
            "trace": [],
        },
    )

    result = tools.dispatch_retrieval("query", "intent", [], "hybrid_search", allow_rerank=False)

    assert [chunk["content"] for chunk in result["chunks"]] == [
        "vector-1",
        "graph-1",
        "keyword-1",
        "vector-2",
        "graph-2",
        "keyword-2",
    ]
    assert result["relation_paths"] == [{"path": "A -> relates -> B"}]
    assert any(item["node"] == "hybrid_round_robin" for item in result["trace"])


def test_hybrid_search_reranks_after_vector_graph_keyword_merge(monkeypatch):
    from services.rag_api.agent import tools

    captured: dict[str, object] = {}

    def fake_vector_rule_search(*args, **kwargs):
        captured["vector_allow_rerank"] = kwargs["allow_rerank"]
        captured["vector_allow_keyword_search"] = kwargs["allow_keyword_search"]
        return {
            "mode": "vector",
            "chunks": [
                _chunk("vector-1", "vector-1.md", 0.6, "vector"),
                _chunk("vector-2", "vector-2.md", 0.5, "vector"),
            ],
            "relation_paths": [],
            "error": None,
            "trace": [],
        }

    def fake_apply_rerank(query, chunks, settings, top_k):
        captured["rerank_input"] = [chunk["content"] for chunk in chunks]
        return [chunks[1], chunks[0]], {"enabled": True, "provider": "api", "candidate_count": len(chunks), "returned_count": 2}

    monkeypatch.setattr(tools, "get_retrieval_top_k", lambda settings=None: 2)
    monkeypatch.setattr(tools, "load_rag_settings", lambda: RagSettings(top_k=2, rerank_enabled=True))
    monkeypatch.setattr(tools, "vector_rule_search", fake_vector_rule_search)
    monkeypatch.setattr(
        tools,
        "graph_relation_search_tool",
        lambda *args, **kwargs: {
            "mode": "graph",
            "chunks": [_chunk("graph-1", "graph-1.md", 0.4, "keyword")],
            "relation_paths": [],
            "error": None,
            "trace": [],
        },
    )
    monkeypatch.setattr(
        tools,
        "keyword_search_candidates",
        lambda query, chunks, settings, limit: [_chunk("keyword-1", "keyword-1.md", 0.7, "bm25_keyword")],
    )
    monkeypatch.setattr(tools, "search_all_chunks", lambda: [{"content": "all"}])
    monkeypatch.setattr(tools, "apply_rerank", fake_apply_rerank)

    result = tools.dispatch_retrieval("query", "intent", [], "hybrid_search", allow_rerank=True)

    assert captured["vector_allow_rerank"] is False
    assert captured["vector_allow_keyword_search"] is False
    assert captured["rerank_input"] == ["vector-1", "graph-1", "keyword-1", "vector-2"]
    assert [chunk["content"] for chunk in result["chunks"]] == ["graph-1", "vector-1"]
    assert any(item["node"] == "hybrid_rerank" for item in result["trace"])


def test_vector_search_ignores_deprecated_bm25_toggle(monkeypatch):
    from services.rag_api.agent import tools

    monkeypatch.setattr(tools, "get_retrieval_top_k", lambda settings=None: 2)
    monkeypatch.setattr(
        tools,
        "load_rag_settings",
        lambda: RagSettings(top_k=2, hybrid_bm25_enabled=True, rerank_enabled=False),
    )
    monkeypatch.setattr(tools, "apply_query_expansion", lambda query, settings: ([query], {"enabled": False, "queries": [query]}))
    monkeypatch.setattr(
        tools,
        "search_chunks",
        lambda *args, **kwargs: [
            _chunk("vector-1", "vector-1.md", 0.9, "vector"),
            _chunk("vector-2", "vector-2.md", 0.8, "vector"),
        ],
    )
    monkeypatch.setattr(
        tools,
        "keyword_search_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("deprecated BM25 toggle must not alter vector search")),
    )

    result = tools.vector_rule_search("query", "intent", [], allow_rerank=False)

    assert result["error"] is None
    assert [chunk["content"] for chunk in result["chunks"]] == ["vector-1", "vector-2"]
    assert any(item["node"] == "hybrid_bm25" and item["output"].get("deprecated") for item in result["trace"])


def test_hybrid_candidate_count_uses_setting_without_param_gate(monkeypatch):
    from services.rag_api.agent import tools

    captured: dict[str, int] = {}

    monkeypatch.setattr(tools, "get_retrieval_top_k", lambda settings=None: 2)
    monkeypatch.setattr(
        tools,
        "load_rag_settings",
        lambda: RagSettings(top_k=2, vector_candidate_k=9, rag_param_tuning_enabled=False, rerank_enabled=False),
    )
    monkeypatch.setattr(
        tools,
        "vector_rule_search",
        lambda *args, **kwargs: {
            "mode": "vector",
            "chunks": [_chunk("vector-1", "vector-1.md", 0.9, "vector")],
            "relation_paths": [],
            "error": None,
            "trace": [],
        },
    )
    monkeypatch.setattr(
        tools,
        "graph_relation_search_tool",
        lambda *args, **kwargs: {"mode": "graph", "chunks": [], "relation_paths": [], "error": None, "trace": []},
    )
    monkeypatch.setattr(tools, "search_all_chunks", lambda: [{"content": "all"}])

    def fake_keyword_search(query, chunks, settings, limit):
        captured["limit"] = limit
        return [_chunk("keyword-1", "keyword-1.md", 0.7, "bm25_keyword")]

    monkeypatch.setattr(tools, "keyword_search_candidates", fake_keyword_search)

    result = tools.dispatch_retrieval("query", "intent", [], "hybrid_search", allow_rerank=False)

    assert captured["limit"] == 9
    assert [chunk["content"] for chunk in result["chunks"]] == ["vector-1", "keyword-1"]


def test_chunk_dedup_uses_stable_identity_instead_of_content_prefix():
    from services.rag_api.agent import tools

    shared_prefix = "x" * 120
    chunks = [
        {"chunk_id": "chunk-a", "source_file": "same.md", "content": f"{shared_prefix}A", "score": 0.9},
        {"chunk_id": "chunk-b", "source_file": "same.md", "content": f"{shared_prefix}B", "score": 0.8},
        {"chunk_id": "chunk-a", "source_file": "same.md", "content": "duplicate channel", "score": 0.7},
    ]

    merged = tools._round_robin_merge_chunk_streams([("vector", chunks)], top_k=3)

    assert [chunk["chunk_id"] for chunk in merged] == ["chunk-a", "chunk-b"]
