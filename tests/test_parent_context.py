from __future__ import annotations

from services.rag_api.rag_settings import RagSettings
from services.rag_api.retrieval.parent_context import apply_parent_context


def _child(
    chunk_id: str,
    document_id: str,
    parent_chunk_id: str,
    *,
    granularity: str = "paragraph",
    score: float = 0.8,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "parent_chunk_id": parent_chunk_id,
        "granularity": granularity,
        "content": f"child {chunk_id}",
        "source_file": f"{document_id}.md",
        "score": score,
        "retrieval_channel": "vector",
    }


def _parent(document_id: str, parent_chunk_id: str) -> dict:
    return {
        "chunk_id": f"{document_id}::document::stable",
        "document_id": document_id,
        "parent_chunk_id": parent_chunk_id,
        "granularity": "document",
        "content": f"parent {document_id}",
        "source_file": f"{document_id}.md",
        "score": 0,
        "retrieval_channel": "parent_lookup",
    }


def test_parent_context_is_disabled_by_default_without_collection_reads(monkeypatch):
    from services.rag_api.retrieval import parent_context

    chunks = [_child("child-a", "doc-a", "split-a")]
    monkeypatch.setattr(
        parent_context,
        "fetch_document_parent_chunks",
        lambda pairs: (_ for _ in ()).throw(AssertionError("disabled parent context must not read Chroma")),
    )

    expanded, trace = apply_parent_context(chunks, RagSettings())

    assert RagSettings().parent_context_enabled is False
    assert expanded == chunks
    assert trace == {
        "enabled": False,
        "expanded_count": 0,
        "missing_parent_count": 0,
        "dropped_unauthorized_count": 0,
        "deduplicated_count": 0,
        "reason": "disabled",
    }


def test_parent_context_expands_only_visible_same_document_children_and_preserves_identity(monkeypatch):
    from services.rag_api.retrieval import parent_context
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    captured: dict[str, set[tuple[str, str]]] = {}
    parent = _parent("doc-a", "split-a")

    def fake_fetch(pairs: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
        captured["pairs"] = pairs
        return {("doc-a", "split-a"): parent}

    monkeypatch.setattr(parent_context, "fetch_document_parent_chunks", fake_fetch)
    context = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-1",
    )
    chunks = [
        _child("child-high", "doc-a", "split-a", score=0.9),
        _child("child-low", "doc-a", "split-a", granularity="sentence", score=0.7),
        _child("child-missing", "doc-a", "split-missing", score=0.6),
        _child("restricted", "doc-b", "split-a", score=1.0),
        _child("document-original", "doc-a", "split-other", granularity="document", score=0.5),
    ]

    with use_retrieval_context(context):
        expanded, trace = apply_parent_context(chunks, RagSettings(parent_context_enabled=True))

    assert captured["pairs"] == {("doc-a", "split-a"), ("doc-a", "split-missing")}
    assert all(chunk["document_id"] == "doc-a" for chunk in expanded)
    assert [chunk["chunk_id"] for chunk in expanded] == [
        "doc-a::document::stable",
        "child-missing",
        "document-original",
    ]
    assert expanded[0]["content"] == "parent doc-a"
    assert expanded[0]["score"] == 0.9
    assert expanded[0]["retrieval_channel"] == "vector"
    assert expanded[0]["granularity"] == "document"
    assert expanded[0]["matched_chunk_id"] == "child-high"
    assert expanded[0]["matched_granularity"] == "paragraph"
    assert trace == {
        "enabled": True,
        "expanded_count": 2,
        "missing_parent_count": 1,
        "dropped_unauthorized_count": 1,
        "deduplicated_count": 1,
        "reason": "expanded",
    }


def test_parent_lookup_uses_current_collection_and_rechecks_permission_and_exact_pair(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context
    from services.rag_api.vector import chroma_store

    captured = {}

    class Collection:
        def get(self, **kwargs):
            captured.update(kwargs)
            return {
                "ids": ["parent-a", "parent-b", "parent-other"],
                "documents": ["parent A", "parent B", "other parent"],
                "metadatas": [
                    {
                        "chunk_id": "parent-a",
                        "document_id": "doc-a",
                        "parent_chunk_id": "split-a",
                        "granularity": "document",
                    },
                    {
                        "chunk_id": "parent-b",
                        "document_id": "doc-b",
                        "parent_chunk_id": "split-a",
                        "granularity": "document",
                    },
                    {
                        "chunk_id": "parent-other",
                        "document_id": "doc-a",
                        "parent_chunk_id": "split-other",
                        "granularity": "document",
                    },
                ],
            }

    monkeypatch.setattr(chroma_store, "get_collection", lambda: Collection())
    context = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-1",
    )

    with use_retrieval_context(context):
        result = chroma_store.fetch_document_parent_chunks({("doc-a", "split-a"), ("doc-b", "split-a")})

    assert set(result) == {("doc-a", "split-a")}
    assert result[("doc-a", "split-a")]["chunk_id"] == "parent-a"
    assert captured["include"] == ["documents", "metadatas"]
    assert {str(item) for item in captured["where"]["$and"]} == {
        str({"granularity": {"$eq": "document"}}),
        str({"document_id": {"$in": ["doc-a"]}}),
        str({"parent_chunk_id": {"$in": ["split-a"]}}),
    }


def test_vector_graph_and_hybrid_apply_parent_context_before_rerank_or_return(monkeypatch):
    from services.rag_api.agent import tools

    settings = RagSettings(top_k=2, parent_context_enabled=True, rerank_enabled=True)
    child = _child("child-a", "doc-a", "split-a")
    parent = _parent("doc-a", "split-a")
    applied: list[list[str]] = []
    rerank_inputs: list[list[str]] = []

    def fake_apply(chunks, active_settings):
        applied.append([chunk["chunk_id"] for chunk in chunks])
        return ([{**parent, "score": chunks[0]["score"]}] if chunks else []), {
            "enabled": True,
            "expanded_count": len(chunks),
            "missing_parent_count": 0,
            "dropped_unauthorized_count": 0,
            "deduplicated_count": 0,
            "reason": "expanded",
        }

    def fake_rerank(query, chunks, active_settings, top_k):
        rerank_inputs.append([chunk["chunk_id"] for chunk in chunks])
        return chunks[:top_k], {"enabled": True}

    monkeypatch.setattr(tools, "load_rag_settings", lambda: settings)
    monkeypatch.setattr(tools, "apply_parent_context", fake_apply)
    monkeypatch.setattr(tools, "apply_query_expansion", lambda query, active: ([query], {"enabled": False}))
    monkeypatch.setattr(tools, "apply_rerank", fake_rerank)
    monkeypatch.setattr(tools, "search_chunks", lambda *args, **kwargs: [child])
    monkeypatch.setattr(
        tools,
        "graph_relation_search",
        lambda *args, **kwargs: {"chunks": [child], "relation_paths": [], "trace": []},
    )

    vector = tools.dispatch_retrieval("query", "intent", [], "vector_rule_search")
    graph = tools.dispatch_retrieval("query", "intent", [], "graph_relation_search_tool")

    monkeypatch.setattr(
        tools,
        "vector_rule_search",
        lambda *args, **kwargs: {"mode": "vector", "chunks": [child], "relation_paths": [], "error": None, "trace": []},
    )
    monkeypatch.setattr(
        tools,
        "graph_relation_search_tool",
        lambda *args, **kwargs: {"mode": "graph", "chunks": [child], "relation_paths": [], "error": None, "trace": []},
    )
    monkeypatch.setattr(tools, "search_all_chunks", lambda: [])
    hybrid = tools.dispatch_retrieval("query", "intent", [], "hybrid_search")

    assert vector["chunks"][0]["chunk_id"] == "doc-a::document::stable"
    assert graph["chunks"][0]["chunk_id"] == "doc-a::document::stable"
    assert hybrid["chunks"][0]["chunk_id"] == "doc-a::document::stable"
    assert applied == [["child-a"], ["child-a"], ["child-a"]]
    assert rerank_inputs == [["doc-a::document::stable"], ["doc-a::document::stable"]]
    for result in (vector, graph, hybrid):
        assert any(item["node"] == "parent_context" for item in result["trace"])


def test_parent_context_evaluation_profile_uses_multi_vector_collection():
    from services.rag_api.evaluation.profiles import SWITCH_KEYS, build_evaluation_profiles

    profile = next(item for item in build_evaluation_profiles(RagSettings()) if item["id"] == "parent_context_enabled")

    assert "parent_context_enabled" in SWITCH_KEYS
    assert profile["settings"].parent_context_enabled is True
    assert profile["settings"].multi_vector_enabled is True
    assert profile["collection_name"] == "crabrag_eval_multi_vector"


def test_quality_metrics_credit_the_matched_child_identity_after_parent_expansion():
    from services.rag_api.evaluation.quality import calculate_quality_metrics

    reference = {
        **_parent("doc-a", "split-a"),
        "matched_chunk_id": "child-a",
        "matched_granularity": "sentence",
    }
    metrics = calculate_quality_metrics(
        [
            {
                "answer": "answer",
                "references": [reference],
                "expected": {"expect_references": True, "expected_chunk_ids": ["child-a"]},
            }
        ]
    )

    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["citation_precision"] == 1.0
