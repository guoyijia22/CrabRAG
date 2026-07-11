from __future__ import annotations

from services.rag_api.rag_settings import RagSettings, resolve_retrieval_top_k


SIMPLE_QUERY = "资费是多少？"
COMPLEX_QUERY = "请分别说明企业客户办理地址迁移和带宽变更时需要提交哪些材料、经过哪些审核步骤，以及两项业务对现有资费和服务时限分别有什么影响？"


def _chunks(count: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk-{index}",
            "content": f"证据 {index}",
            "source_file": f"source-{index}.md",
            "score": 1 - index / 100,
        }
        for index in range(count)
    ]


def test_dynamic_top_k_is_disabled_by_default_and_preserves_the_configured_value():
    settings = RagSettings(top_k=3)

    decision = resolve_retrieval_top_k(COMPLEX_QUERY, settings)

    assert settings.dynamic_top_k_enabled is False
    assert decision == {
        "enabled": False,
        "base_top_k": 3,
        "effective_top_k": 3,
        "increase": 0,
        "reason": "disabled",
    }


def test_dynamic_top_k_increases_only_for_complex_queries_and_never_exceeds_ten():
    enabled = RagSettings(top_k=2, dynamic_top_k_enabled=True)

    simple = resolve_retrieval_top_k(SIMPLE_QUERY, enabled)
    complex_decision = resolve_retrieval_top_k(COMPLEX_QUERY, enabled)
    capped = resolve_retrieval_top_k(COMPLEX_QUERY, enabled.model_copy(update={"top_k": 9}))

    assert simple["effective_top_k"] == 2
    assert simple["reason"] == "simple_query"
    assert complex_decision["effective_top_k"] == 4
    assert complex_decision["increase"] == 2
    assert complex_decision["reason"] != "simple_query"
    assert capped["effective_top_k"] == 10


def test_vector_graph_and_hybrid_use_the_same_effective_top_k_and_trace(monkeypatch):
    from services.rag_api.agent import tools

    settings = RagSettings(top_k=2, dynamic_top_k_enabled=True, rerank_enabled=False)
    expected_top_k = resolve_retrieval_top_k(COMPLEX_QUERY, settings)["effective_top_k"]
    captured: dict[str, int] = {}
    all_chunks = _chunks(8)

    monkeypatch.setattr(tools, "load_rag_settings", lambda: settings)
    monkeypatch.setattr(tools, "apply_query_expansion", lambda query, active: ([query], {"enabled": False, "queries": [query]}))

    def fake_search_chunks(query, intent, entities, top_k):
        captured["vector"] = top_k
        return all_chunks[:top_k]

    def fake_graph_search(query, intent, top_k):
        captured["graph"] = top_k
        return {
            "chunks": all_chunks[:top_k],
            "relation_paths": [{"path": str(index)} for index in range(top_k)],
            "trace": [],
        }

    monkeypatch.setattr(tools, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(tools, "graph_relation_search", fake_graph_search)
    monkeypatch.setattr(tools, "search_all_chunks", lambda: all_chunks)
    monkeypatch.setattr(tools, "keyword_search_candidates", lambda query, chunks, active, limit: chunks[:limit])

    vector = tools.dispatch_retrieval(COMPLEX_QUERY, "intent", [], "vector_rule_search", allow_query_expansion=False, allow_rerank=False)
    graph = tools.dispatch_retrieval(COMPLEX_QUERY, "intent", [], "graph_relation_search_tool", allow_rerank=False)
    hybrid = tools.dispatch_retrieval(COMPLEX_QUERY, "intent", [], "hybrid_search", allow_query_expansion=False, allow_rerank=False)

    assert captured == {"vector": expected_top_k, "graph": expected_top_k}
    assert len(vector["chunks"]) == expected_top_k
    assert len(graph["chunks"]) == expected_top_k
    assert len(graph["relation_paths"]) == expected_top_k
    assert len(hybrid["chunks"]) == expected_top_k
    for result in (vector, graph, hybrid):
        trace = next(item["output"] for item in result["trace"] if item["node"] == "dynamic_top_k")
        assert trace["base_top_k"] == 2
        assert trace["effective_top_k"] == expected_top_k
        assert trace["reason"] != "disabled"


def test_retrieve_and_generate_share_effective_top_k(monkeypatch):
    from services.rag_api.agent import nodes

    settings = RagSettings(top_k=2, dynamic_top_k_enabled=True)
    all_chunks = _chunks(6)
    state = {
        "question": COMPLEX_QUERY,
        "effective_question": COMPLEX_QUERY,
        "business_scope": {"in_scope": True},
        "intent": "业务变更",
        "entities": [],
        "selected_tool": "vector_rule_search",
        "trace": [],
    }
    monkeypatch.setattr(nodes, "load_rag_settings", lambda: settings)
    monkeypatch.setattr(
        nodes,
        "dispatch_retrieval",
        lambda *args, **kwargs: {
            "mode": "vector",
            "chunks": all_chunks,
            "relation_paths": [{"path": str(index)} for index in range(6)],
            "trace": [],
            "error": None,
        },
    )
    monkeypatch.setattr(nodes, "chat_completion", lambda *args, **kwargs: "模型答复")

    retrieved = nodes.retrieve_node(state)
    generated = nodes.generate_answer_node(retrieved)

    assert len(retrieved["retrieved_chunks"]) == 4
    assert len(retrieved["relation_paths"]) == 4
    assert len(generated["references"]) == 4
    assert "source-3.md" in generated["answer"]
    retrieve_trace = next(item["output"] for item in retrieved["trace"] if item["node"] == "retrieve")
    assert retrieve_trace["base_top_k"] == 2
    assert retrieve_trace["effective_top_k"] == 4


def test_evaluation_truncates_references_with_the_effective_top_k(monkeypatch):
    from services.rag_api.evaluation import runner

    settings = RagSettings(top_k=2, dynamic_top_k_enabled=True)
    profile = {"id": "dynamic", "settings": settings, "collection_name": None}
    question = {"id": "case-1", "question": SIMPLE_QUERY, "expect_references": True}
    monkeypatch.setattr(runner, "get_settings", lambda: type("Settings", (), {"use_local_models": False})())
    monkeypatch.setattr(
        runner,
        "run_qa",
        lambda state: {
            **state,
            "effective_question": COMPLEX_QUERY,
            "answer": "answer",
            "references": _chunks(6),
            "relation_paths": [{"path": str(index)} for index in range(6)],
            "trace": [],
            "error": None,
        },
    )

    case = runner._run_case("eval-1", profile, question)

    assert len(case["references"]) == 4
    assert len(case["relation_paths"]) == 4


def test_evidence_service_does_not_retruncate_dynamic_results_to_the_base_value(monkeypatch):
    from services.rag_api.retrieval import evidence_service

    settings = RagSettings(top_k=2, dynamic_top_k_enabled=True)
    monkeypatch.setattr(evidence_service, "load_rag_settings", lambda: settings)
    monkeypatch.setattr(evidence_service, "get_category_names", lambda: ["业务变更"])
    monkeypatch.setattr(evidence_service, "check_business_scope", lambda *args: {"in_scope": True, "matched_entities": []})
    monkeypatch.setattr(
        evidence_service,
        "heuristic_classify",
        lambda *args: {"intent": "业务变更", "question_type": "关联推理", "retrieval_mode": "vector", "entities": []},
    )
    monkeypatch.setattr(evidence_service, "heuristic_tool_choice", lambda state: ("vector_rule_search", "test"))
    monkeypatch.setattr(
        evidence_service,
        "dispatch_retrieval",
        lambda *args, **kwargs: {
            "mode": "vector",
            "chunks": _chunks(6),
            "relation_paths": [{"path": str(index)} for index in range(6)],
            "error": None,
            "trace": [],
        },
    )

    result = evidence_service._retrieve_evidence(
        COMPLEX_QUERY,
        mode="auto",
        include_trace=True,
        no_rerank=False,
    )

    assert len(result["evidence"]) == 4
    assert len(result["relation_paths"]) == 4
    retrieve_trace = next(item["output"] for item in result["trace"] if item["node"] == "retrieve")
    assert retrieve_trace["base_top_k"] == 2
    assert retrieve_trace["effective_top_k"] == 4
