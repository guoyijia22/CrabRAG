from __future__ import annotations

from services.rag_api.rag_settings import RagSettings
from services.rag_api.retrieval.dedup import NEAR_DUPLICATE_JACCARD_THRESHOLD, apply_candidate_dedup


BASE_TEXT = "企业客户办理地址迁移需要提交营业执照、授权书和变更申请表。"
PUNCTUATION_VARIANT = "企业客户办理地址迁移，需要提交营业执照、授权书和变更申请表。"
UNRELATED_TEXT = "故障报修时限为四小时，客户应提供电路编号。"


def _chunk(
    chunk_id: str,
    content: str,
    score: float,
    *,
    document_id: str = "doc-a",
    source_file: str = "rules.md",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": content,
        "score": score,
        "document_id": document_id,
        "source_file": source_file,
        "retrieval_channel": "vector",
    }


def test_dedup_is_disabled_by_default_but_stable_identity_duplicates_are_always_removed():
    chunks = [
        _chunk("same", "lower score copy", 0.4),
        _chunk("near-a", BASE_TEXT, 0.8),
        _chunk("same", "higher score copy", 0.9),
        _chunk("near-b", PUNCTUATION_VARIANT, 0.7),
    ]

    result, trace = apply_candidate_dedup(chunks, RagSettings())

    assert RagSettings().dedup_enabled is False
    assert [chunk["chunk_id"] for chunk in result] == ["same", "near-a", "near-b"]
    assert result[0]["content"] == "higher score copy"
    assert trace == {
        "enabled": False,
        "input_count": 4,
        "output_count": 3,
        "removed_count": 1,
        "exact_removed_count": 1,
        "near_duplicate_removed_count": 0,
        "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        "reason": "disabled",
    }


def test_near_duplicate_dedup_keeps_highest_score_within_document_or_source_only():
    chunks = [
        _chunk("doc-a-low", BASE_TEXT, 0.7),
        _chunk("doc-a-high", PUNCTUATION_VARIANT, 0.9),
        _chunk("doc-a-other", UNRELATED_TEXT, 0.8),
        _chunk("doc-b-same", BASE_TEXT, 1.0, document_id="doc-b"),
        _chunk("source-low", BASE_TEXT, 0.6, document_id="", source_file="legacy.md"),
        _chunk("source-high", PUNCTUATION_VARIANT, 0.95, document_id="", source_file="legacy.md"),
    ]

    result, trace = apply_candidate_dedup(chunks, RagSettings(dedup_enabled=True))

    assert NEAR_DUPLICATE_JACCARD_THRESHOLD == 0.85
    assert [chunk["chunk_id"] for chunk in result] == [
        "doc-a-high",
        "doc-a-other",
        "doc-b-same",
        "source-high",
    ]
    assert trace == {
        "enabled": True,
        "input_count": 6,
        "output_count": 4,
        "removed_count": 2,
        "exact_removed_count": 0,
        "near_duplicate_removed_count": 2,
        "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        "reason": "near_duplicate_filter",
    }


def test_vector_graph_and_hybrid_apply_dedup_before_rerank_or_return(monkeypatch):
    from services.rag_api.agent import tools

    settings = RagSettings(top_k=2, dedup_enabled=True, rerank_enabled=True)
    candidates = [_chunk("candidate-a", BASE_TEXT, 0.8), _chunk("candidate-b", PUNCTUATION_VARIANT, 0.9)]
    kept = _chunk("kept", PUNCTUATION_VARIANT, 0.9)
    applied: list[list[str]] = []
    rerank_inputs: list[list[str]] = []
    deferred_flags: list[bool] = []

    def fake_dedup(chunks, active_settings):
        applied.append([chunk["chunk_id"] for chunk in chunks])
        return ([kept] if chunks else []), {
            "enabled": True,
            "input_count": len(chunks),
            "output_count": int(bool(chunks)),
            "removed_count": max(0, len(chunks) - 1),
            "exact_removed_count": 0,
            "near_duplicate_removed_count": max(0, len(chunks) - 1),
            "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "reason": "near_duplicate_filter",
        }

    def fake_rerank(query, chunks, active_settings, top_k):
        rerank_inputs.append([chunk["chunk_id"] for chunk in chunks])
        return chunks[:top_k], {"enabled": True}

    monkeypatch.setattr(tools, "load_rag_settings", lambda: settings)
    monkeypatch.setattr(tools, "apply_candidate_dedup", fake_dedup)
    monkeypatch.setattr(tools, "apply_query_expansion", lambda query, active: ([query], {"enabled": False}))
    monkeypatch.setattr(tools, "apply_rerank", fake_rerank)
    monkeypatch.setattr(tools, "search_chunks", lambda *args, **kwargs: list(candidates))
    monkeypatch.setattr(
        tools,
        "graph_relation_search",
        lambda *args, **kwargs: {"chunks": list(candidates), "relation_paths": [], "trace": []},
    )

    vector = tools.dispatch_retrieval("query", "intent", [], "vector_rule_search")
    graph = tools.dispatch_retrieval("query", "intent", [], "graph_relation_search_tool")

    def fake_vector(*args, **kwargs):
        deferred_flags.append(kwargs["allow_dedup"])
        return {"mode": "vector", "chunks": list(candidates), "relation_paths": [], "error": None, "trace": []}

    def fake_graph(*args, **kwargs):
        deferred_flags.append(kwargs["allow_dedup"])
        return {"mode": "graph", "chunks": list(candidates), "relation_paths": [], "error": None, "trace": []}

    monkeypatch.setattr(tools, "vector_rule_search", fake_vector)
    monkeypatch.setattr(tools, "graph_relation_search_tool", fake_graph)
    monkeypatch.setattr(tools, "search_all_chunks", lambda: [])
    hybrid = tools.dispatch_retrieval("query", "intent", [], "hybrid_search")

    assert [result["chunks"][0]["chunk_id"] for result in (vector, graph, hybrid)] == ["kept", "kept", "kept"]
    assert applied == [["candidate-b", "candidate-a"], ["candidate-a", "candidate-b"], ["candidate-a", "candidate-b"]]
    assert rerank_inputs == [["kept"], ["kept"]]
    assert deferred_flags == [False, False]
    for result in (vector, graph, hybrid):
        trace = next(item["output"] for item in result["trace"] if item["node"] == "candidate_dedup")
        assert trace["input_count"] == 2
        assert trace["output_count"] == 1
        assert trace["removed_count"] == 1


def test_dedup_evaluation_profile_and_approval_switch_are_registered():
    from services.rag_api.evaluation.profiles import SWITCH_KEYS, build_evaluation_profiles

    profile = next(item for item in build_evaluation_profiles(RagSettings()) if item["id"] == "dedup_enabled")

    assert "dedup_enabled" in SWITCH_KEYS
    assert profile["settings"].dedup_enabled is True
    assert profile["enabled_switches"] == ["dedup_enabled"]
