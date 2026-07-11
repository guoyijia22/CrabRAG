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


def test_prompts_switch_between_chinese_and_english():
    from services.rag_api.agent.prompts import (
        build_answer_prompt,
        build_classify_prompt,
        build_tool_choice_prompt,
        detect_prompt_language,
    )

    assert detect_prompt_language("Where is the annual report?") == "en"
    assert detect_prompt_language("关于一渠一表的报告在哪") == "zh"

    classify_prompt = build_classify_prompt(["General Knowledge"], language="en")
    tool_prompt = build_tool_choice_prompt(language="en")
    answer_prompt = build_answer_prompt(language="en")

    assert "intent" in classify_prompt
    assert "retrieval_mode" in classify_prompt
    assert "You are CrabRAG" in classify_prompt
    assert "vector_rule_search" in tool_prompt
    assert "Only output JSON" in tool_prompt
    assert "## Category" in answer_prompt
    assert "### References" in answer_prompt
    assert "ONLY use the information in the provided Context" in answer_prompt

    zh_answer_prompt = build_answer_prompt(language="zh")
    assert "【业务类别】" in zh_answer_prompt
    assert "【参考知识库原文片段】" in zh_answer_prompt


def test_rag_context_assigns_reference_ids_and_keeps_source_text():
    from services.rag_api.agent.rag_context import build_rag_context

    context = build_rag_context(
        chunks=[
            {
                "content": "原文证据 A",
                "source_file": "general.md",
                "section_title": "第一节",
                "category": "通用知识库",
            },
            {
                "content": "Evidence B",
                "source_file": "manual.pdf",
                "section_title": "Chapter 2",
                "category": "Manual",
            },
        ],
        relation_paths=[
            {
                "path": "A -> relates_to -> B",
                "description": "A relates to B",
                "source_file": "manual.pdf",
                "evidence": "Evidence B",
            }
        ],
    )

    assert "Knowledge Graph Data (Entity)" in context
    assert "Knowledge Graph Data (Relationship)" in context
    assert "Document Chunks" in context
    assert '"reference_id": 1' in context
    assert '"reference_id": 2' in context
    assert "[1] general.md" in context
    assert "[2] manual.pdf" in context
    assert "原文证据 A" in context
    assert "Evidence B" in context


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


def test_generate_answer_node_preserves_partial_retrieval_error(monkeypatch):
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
        "error": "missing graph collection",
    }

    result = nodes.generate_answer_node(state)

    assert result["error"] == "missing graph collection"


def test_generate_answer_node_uses_structured_system_prompt_for_english(monkeypatch):
    from services.rag_api.agent import nodes

    captured = {}

    def fake_chat_completion(messages, temperature, max_tokens):
        captured["messages"] = messages
        return "## Category\nGeneral\n\n## Answer\nUse the manual.\n\n### References\n- [1] manual.pdf"

    monkeypatch.setattr(nodes, "get_retrieval_top_k", lambda settings=None: 1)
    monkeypatch.setattr(nodes, "load_rag_settings", lambda: RagSettings(top_k=1))
    monkeypatch.setattr(nodes, "chat_completion", fake_chat_completion)
    state = {
        "question": "Where is the manual?",
        "effective_question": "Where is the manual?",
        "business_scope": {"in_scope": True},
        "intent": "General",
        "retrieved_chunks": [{"content": "The manual is in manual.pdf.", "source_file": "manual.pdf"}],
        "relation_paths": [],
        "trace": [],
    }

    result = nodes.generate_answer_node(state)

    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1] == {"role": "user", "content": "Where is the manual?"}
    assert "Knowledge Graph Data" in captured["messages"][0]["content"]
    assert "Document Chunks" in captured["messages"][0]["content"]
    assert "### References" in result["answer"]
    assert any(item["node"] == "rag_context_builder" for item in result["trace"])


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
