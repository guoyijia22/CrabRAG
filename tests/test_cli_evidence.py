from __future__ import annotations

import json

import pytest

from services.rag_api.rag_settings import RagSettings


def test_retrieve_evidence_uses_heuristics_without_chat_llm(monkeypatch):
    from services.rag_api.retrieval import evidence_service

    def fail_chat(*args, **kwargs):
        raise AssertionError("chat LLM must not be called")

    captured = {}

    def fake_dispatch(query, intent, entities, selected_tool, *, allow_query_expansion, allow_rerank):
        captured.update(
            {
                "query": query,
                "intent": intent,
                "entities": entities,
                "selected_tool": selected_tool,
                "allow_query_expansion": allow_query_expansion,
                "allow_rerank": allow_rerank,
            }
        )
        return {
            "mode": "vector",
            "chunks": [
                {
                    "content": "企业客户申请资费套餐时需要提交营业执照。",
                    "source_file": "rules.md",
                    "source_path": "docs/rules.md",
                    "section_title": "资费材料",
                    "category": "资费咨询",
                    "score": 0.88,
                    "rerank_score": 0.91,
                    "retrieval_channel": "vector",
                }
            ],
            "relation_paths": [],
            "error": None,
            "trace": [{"node": "fake_retrieve", "output": {"ok": True}}],
        }

    monkeypatch.setattr(evidence_service, "chat_completion", fail_chat)
    monkeypatch.setattr(evidence_service, "get_category_names", lambda: ["资费咨询", "客户准入"])
    monkeypatch.setattr(
        evidence_service,
        "check_business_scope",
        lambda question, categories: {
            "in_scope": True,
            "score": 0.8,
            "matched_entities": [],
            "matched_keywords": ["资费"],
            "matched_categories": ["资费咨询"],
            "excluded_keywords": [],
        },
    )
    monkeypatch.setattr(evidence_service, "load_rag_settings", lambda: RagSettings(query_expansion_enabled=True, rerank_enabled=True, top_k=3))
    monkeypatch.setattr(evidence_service, "dispatch_retrieval", fake_dispatch)

    result = evidence_service.retrieve_evidence("企业客户资费套餐需要什么材料？", top_k=1, include_trace=True)

    assert result["ok"] is True
    assert result["question"] == "企业客户资费套餐需要什么材料？"
    assert result["retrieval_mode"] == "vector"
    assert result["evidence"][0]["content"].startswith("企业客户申请资费套餐")
    assert result["trace"]
    assert captured["allow_query_expansion"] is False
    assert captured["allow_rerank"] is True
    assert captured["selected_tool"] == "vector_rule_search"


def test_retrieve_evidence_no_rerank_flag_disables_rerank(monkeypatch):
    from services.rag_api.retrieval import evidence_service

    captured = {}

    def fake_dispatch(query, intent, entities, selected_tool, *, allow_query_expansion, allow_rerank):
        captured["allow_rerank"] = allow_rerank
        return {"mode": "vector", "chunks": [], "relation_paths": [], "error": None, "trace": []}

    monkeypatch.setattr(evidence_service, "get_category_names", lambda: ["资费咨询"])
    monkeypatch.setattr(evidence_service, "check_business_scope", lambda question, categories: {"in_scope": True, "matched_entities": []})
    monkeypatch.setattr(evidence_service, "load_rag_settings", lambda: RagSettings(rerank_enabled=True))
    monkeypatch.setattr(evidence_service, "dispatch_retrieval", fake_dispatch)

    result = evidence_service.retrieve_evidence("资费是多少？", no_rerank=True)

    assert result["ok"] is True
    assert result["warnings"] == ["no_evidence"]
    assert captured["allow_rerank"] is False


def test_cli_outputs_json_to_stdout(monkeypatch, capsys):
    from services.rag_api.cli import evidence as evidence_cli

    monkeypatch.setattr(
        evidence_cli,
        "retrieve_evidence",
        lambda **kwargs: {
            "ok": True,
            "question": kwargs["question"],
            "effective_question": kwargs["question"],
            "retrieval_mode": "vector",
            "evidence": [{"content": "证据", "source_file": "rules.md"}],
            "warnings": [],
        },
    )

    exit_code = evidence_cli.main(["--question", "资费是多少？", "--top-k", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["evidence"][0]["content"] == "证据"


def test_cli_outputs_errors_as_json_to_stderr(monkeypatch, capsys):
    from services.rag_api.cli import evidence as evidence_cli

    def fail(**kwargs):
        raise RuntimeError("本地向量模型文件缺失：model.onnx")

    monkeypatch.setattr(evidence_cli, "retrieve_evidence", fail)

    exit_code = evidence_cli.main(["--question", "资费是多少？"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "model.onnx" in payload["error"]


def test_cli_outputs_argument_errors_as_json_to_stderr(capsys):
    from services.rag_api.cli import evidence as evidence_cli

    exit_code = evidence_cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "question" in payload["error"]


def test_vector_search_can_disable_query_expansion_and_rerank(monkeypatch):
    from services.rag_api.agent import tools

    def fail_query_expansion(*args, **kwargs):
        raise AssertionError("query expansion must not be called")

    def fail_rerank(*args, **kwargs):
        raise AssertionError("rerank must not be called")

    monkeypatch.setattr(tools, "apply_query_expansion", fail_query_expansion)
    monkeypatch.setattr(tools, "apply_rerank", fail_rerank)
    monkeypatch.setattr(tools, "load_rag_settings", lambda: RagSettings(query_expansion_enabled=True, rerank_enabled=True, top_k=1))
    monkeypatch.setattr(
        tools,
        "search_chunks",
        lambda query, intent, entities, top_k: [
            {
                "content": "原始证据片段",
                "source_file": "rules.md",
                "score": 0.8,
            }
        ],
    )

    result = tools.vector_rule_search("资费是多少？", "资费咨询", [], allow_query_expansion=False, allow_rerank=False)

    assert result["chunks"][0]["content"] == "原始证据片段"
    assert result["trace"][0]["output"]["reason"] == "disabled_for_evidence_cli"
    assert result["trace"][-1]["output"]["reason"] == "disabled_by_cli_flag"


def test_crabrag_skill_points_to_cli_not_chat_endpoint():
    from pathlib import Path

    skill_text = Path("crabrag.skill").read_text(encoding="utf-8")

    assert "CRABRAG_HOME" in skill_text
    assert "crab-rag.bat" in skill_text
    assert "evidence[].content" in skill_text
    assert "/api/chat" in skill_text
    assert "CrabRAG" in skill_text
    assert "不要调用 CrabRAG 的 `/api/chat` 接口" in skill_text
    assert "DeepSeek V4 Flash" not in skill_text
