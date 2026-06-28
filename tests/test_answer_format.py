from __future__ import annotations

from services.rag_api.rag_settings import RagSettings


def test_answer_prompt_uses_general_reference_heading():
    from services.rag_api.agent.prompts import build_answer_prompt

    prompt = build_answer_prompt()
    forbidden_heading = "【" + "合规" + "提示】"
    old_reference_heading = "【参考" + "规范" + "原文片段】"

    assert forbidden_heading not in prompt
    assert old_reference_heading not in prompt
    assert "不得编造事实、规则、结论或来源" in prompt
    assert "【参考知识库原文片段】" in prompt


def test_fallback_answer_format_omits_compliance_hint_and_keeps_references(monkeypatch):
    from services.rag_api.agent import nodes

    monkeypatch.setattr(nodes, "get_retrieval_top_k", lambda settings=None: 2)
    state = {
        "intent": "通用知识库",
        "retrieved_chunks": [
            {"content": "原文证据", "source_file": "general.md"},
        ],
    }

    answer = nodes._format_answer_from_chunks(state, "这是答复")
    forbidden_heading = "【" + "合规" + "提示】"
    old_reference_heading = "【参考" + "规范" + "原文片段】"

    assert forbidden_heading not in answer
    assert old_reference_heading not in answer
    assert "【参考知识库原文片段】" in answer
    assert "来源：《general.md》" in answer
    assert "原文片段：原文证据" in answer


def test_generate_answer_node_autofills_general_answer_format(monkeypatch):
    from services.rag_api.agent import nodes

    monkeypatch.setattr(nodes, "get_retrieval_top_k", lambda settings=None: 1)
    monkeypatch.setattr(nodes, "load_rag_settings", lambda: RagSettings(top_k=1))
    monkeypatch.setattr(nodes, "chat_completion", lambda messages, temperature, max_tokens: "模型未按格式输出")
    state = {
        "question": "问题",
        "effective_question": "问题",
        "business_scope": {"in_scope": True},
        "intent": "通用知识库",
        "retrieved_chunks": [{"content": "原文证据", "source_file": "general.md"}],
        "relation_paths": [],
        "trace": [],
    }

    result = nodes.generate_answer_node(state)
    forbidden_heading = "【" + "合规" + "提示】"

    assert forbidden_heading not in result["answer"]
    assert "【参考知识库原文片段】" in result["answer"]
    assert "模型未按格式输出" in result["answer"]


def test_evaluation_retrieval_only_answer_uses_general_reference_heading():
    from services.rag_api.evaluation.runner import _format_retrieval_only_answer

    answer = _format_retrieval_only_answer(
        "通用知识库",
        [{"content": "原文证据", "source_file": "general.md"}],
    )
    old_reference_heading = "【参考" + "规范" + "原文片段】"

    assert old_reference_heading not in answer
    assert "【参考知识库原文片段】" in answer
    assert "来源：《general.md》" in answer
