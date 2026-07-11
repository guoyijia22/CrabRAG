from __future__ import annotations

from services.rag_api.rag_settings import RagSettings


def test_apply_context_token_budget_truncates_chunks_and_paths():
    from services.rag_api.retrieval.context_budget import apply_context_token_budget

    chunks = [
        {"content": "甲" * 500, "source_file": "first.md", "score": 0.9},
        {"content": "乙" * 500, "source_file": "second.md", "score": 0.8},
    ]
    relation_paths = [
        {"path": "A -> 关系 -> B", "description": "丙" * 200},
        {"path": "C -> 关系 -> D", "description": "丁" * 200},
    ]

    budgeted_chunks, budgeted_paths, trace = apply_context_token_budget(
        "问题",
        chunks,
        relation_paths,
        RagSettings(max_context_tokens=220),
    )

    assert len(budgeted_paths) == 1
    assert len(budgeted_chunks) == 1
    assert budgeted_chunks[0]["source_file"] == "first.md"
    assert len(budgeted_chunks[0]["content"]) < 500
    assert trace["truncated"] is True


def test_generate_answer_uses_budgeted_chunks_as_references(monkeypatch):
    from services.rag_api.agent import nodes

    captured: dict[str, str] = {}

    monkeypatch.setattr(nodes, "get_retrieval_top_k", lambda settings=None: 2)
    monkeypatch.setattr(nodes, "load_rag_settings", lambda: RagSettings(max_context_tokens=220, top_k=2))
    monkeypatch.setattr(
        nodes,
        "chat_completion",
        lambda messages, temperature, max_tokens: captured.setdefault("prompt", messages[0]["content"]) or "答复",
    )

    state = {
        "question": "问题",
        "effective_question": "问题",
        "business_scope": {"in_scope": True},
        "intent": "测试",
        "retrieved_chunks": [
            {"content": "甲" * 500, "source_file": "first.md", "score": 0.9},
            {"content": "乙" * 500, "source_file": "second.md", "score": 0.8},
        ],
        "relation_paths": [{"path": "A -> 关系 -> B", "description": "丙" * 200}],
        "trace": [],
    }

    result = nodes.generate_answer_node(state)

    assert len(result["references"]) == len(result["retrieved_chunks"]) == 1
    assert "second.md" not in captured["prompt"]
    assert any(item["node"] == "context_token_budget" for item in result["trace"])
