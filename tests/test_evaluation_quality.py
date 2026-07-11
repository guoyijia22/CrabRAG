from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.rag_api.evaluation import questions, runner
from services.rag_api.evaluation.dataset import (
    EvaluationDatasetError,
    dataset_to_question_set,
    load_evaluation_dataset,
)
from services.rag_api.evaluation.quality import calculate_quality_metrics, evaluate_quality_gate
from services.rag_api.llm.call_metrics import record_model_call
from services.rag_api.rag_settings import RagSettings


def _dataset_payload() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "crabrag-production",
        "dataset_version": "2026.07.1",
        "cases": [
            {
                "id": "case-materials",
                "question": "企业客户需要哪些材料？",
                "expected_document_ids": ["doc-materials"],
                "expect_references": True,
            }
        ],
    }


def test_fixed_dataset_loads_with_stable_fingerprint_and_question_set(tmp_path: Path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    payload = _dataset_payload()
    first_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    second_path.write_text(json.dumps({key: payload[key] for key in reversed(payload)}, ensure_ascii=False), encoding="utf-8")

    first = load_evaluation_dataset(first_path)
    second = load_evaluation_dataset(second_path)

    assert first is not None
    assert second is not None
    assert first["fingerprint"] == second["fingerprint"]
    assert len(first["fingerprint"]) == 64
    question_set = dataset_to_question_set(first)
    assert question_set["questions"] == first["cases"]
    assert question_set["question_generation"] == {
        "mode": "fixed",
        "fixed": True,
        "gate_eligible": True,
        "question_count": 1,
        "schema_version": 1,
        "dataset_id": "crabrag-production",
        "dataset_version": "2026.07.1",
        "dataset_fingerprint": first["fingerprint"],
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        (
            {
                "cases": [
                    {"id": "duplicate", "question": "问题一"},
                    {"id": "duplicate", "question": "问题二"},
                ]
            },
            "duplicate",
        ),
        ({"cases": [{"id": "missing-question"}]}, "question"),
    ],
)
def test_fixed_dataset_rejects_invalid_schema_and_case_identity(tmp_path: Path, change: dict, message: str):
    payload = _dataset_payload()
    payload.update(change)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match=message):
        load_evaluation_dataset(path)


def test_missing_fixed_dataset_keeps_dynamic_generation_but_disables_gate(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert load_evaluation_dataset(missing) is None

    monkeypatch.setattr(questions, "load_evaluation_dataset", lambda: None)
    monkeypatch.setattr(questions, "load_kb_categories", lambda: {"items": []})
    monkeypatch.setattr(questions, "_load_sample_chunks", lambda: [])
    monkeypatch.setattr(questions, "_relation_summaries", lambda: [])
    monkeypatch.setattr(questions, "_generate_with_llm", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))

    question_set = questions.generate_evaluation_question_set()

    assert question_set["question_generation"]["fixed"] is False
    assert question_set["question_generation"]["gate_eligible"] is False


def test_fixed_dataset_is_preferred_over_dynamic_question_generation(monkeypatch):
    dataset = load_evaluation_dataset_from_payload(_dataset_payload())
    monkeypatch.setattr(questions, "load_evaluation_dataset", lambda: dataset)
    monkeypatch.setattr(
        questions,
        "load_kb_categories",
        lambda: (_ for _ in ()).throw(AssertionError("fixed datasets must not inspect the knowledge base")),
    )

    question_set = questions.generate_evaluation_question_set()

    assert question_set["question_generation"]["mode"] == "fixed"
    assert question_set["questions"][0]["id"] == "case-materials"


def load_evaluation_dataset_from_payload(payload: dict) -> dict:
    """Exercise the public file loader without exposing a test-only production API."""

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "dataset.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        loaded = load_evaluation_dataset(path)
    assert loaded is not None
    return loaded


def test_production_quality_metrics_cover_retrieval_citations_leakage_latency_and_calls():
    cases = [
        {
            "answer": "有证据回答",
            "latency_ms": 100,
            "model_call_count": 2,
            "expected": {"expect_references": True, "expected_document_ids": ["doc-a", "doc-b"]},
            "references": [
                {"document_id": "doc-a", "acl_allowed": True, "status": "published"},
                {"document_id": "doc-x", "acl_allowed": False, "status": "published"},
            ],
        },
        {
            "answer": "另一个有证据回答",
            "latency_ms": 300,
            "model_call_count": 1,
            "expected": {"expect_references": True, "expected_document_ids": ["doc-c"]},
            "references": [
                {"document_id": "doc-x", "acl_allowed": True, "status": "published"},
                {"document_id": "doc-c", "acl_allowed": True, "status": "retired"},
            ],
        },
        {
            "answer": "没有证据仍然回答",
            "latency_ms": 200,
            "model_call_count": 0,
            "expected": {"expect_references": True, "expected_document_ids": ["doc-d"]},
            "references": [],
        },
        {
            "answer": "知识库无匹配内容",
            "latency_ms": 50,
            "model_call_count": 0,
            "expected": {"expect_references": False},
            "references": [],
        },
    ]

    metrics = calculate_quality_metrics(cases)

    assert metrics == {
        "recall_at_5": 0.5,
        "mrr_at_10": 0.5,
        "citation_precision": 0.5,
        "citation_coverage": 0.6667,
        "no_evidence_answer_rate": 0.3333,
        "acl_leakage_rate": 0.25,
        "invalid_content_leakage_rate": 0.25,
        "p95_latency_ms": 300,
        "model_call_count": 3,
    }


def _baseline_metrics() -> dict:
    return {
        "recall_at_5": 0.8,
        "mrr_at_10": 0.6,
        "citation_precision": 0.7,
        "citation_coverage": 0.75,
        "no_evidence_answer_rate": 0.1,
        "acl_leakage_rate": 0,
        "invalid_content_leakage_rate": 0,
        "p95_latency_ms": 100,
        "model_call_count": 20,
    }


def test_quality_gate_enforces_leakage_recall_latency_and_improvement_thresholds():
    baseline = _baseline_metrics()
    candidate = {**baseline, "recall_at_5": 0.78, "mrr_at_10": 0.61, "p95_latency_ms": 120}

    passed = evaluate_quality_gate(candidate, baseline, gate_eligible=True)

    assert passed["eligible"] is True
    assert passed["passed"] is True
    assert all(passed["checks"].values())

    failed = evaluate_quality_gate(
        {
            **candidate,
            "recall_at_5": 0.779,
            "p95_latency_ms": 121,
            "acl_leakage_rate": 0.01,
            "invalid_content_leakage_rate": 0.01,
            "mrr_at_10": baseline["mrr_at_10"],
        },
        baseline,
        gate_eligible=True,
    )
    assert failed["passed"] is False
    assert failed["checks"] == {
        "acl_leakage_zero": False,
        "invalid_content_leakage_zero": False,
        "recall_regression_within_limit": False,
        "p95_latency_within_limit": False,
        "primary_quality_improved": False,
    }


def test_dynamic_dataset_cannot_be_used_for_quality_gate():
    result = evaluate_quality_gate(_baseline_metrics(), _baseline_metrics(), gate_eligible=False)

    assert result == {
        "eligible": False,
        "passed": False,
        "checks": {},
        "reasons": ["fixed_dataset_required"],
    }


def test_evaluation_case_captures_model_call_count(monkeypatch):
    profile = {"id": "baseline", "settings": RagSettings(), "collection_name": None}
    question = {"id": "case-1", "question": "测试问题", "expect_references": False}

    def fake_run_qa(state: dict) -> dict:
        record_model_call("chat")
        record_model_call("chat")
        return {**state, "answer": "无匹配", "references": [], "relation_paths": [], "error": None}

    monkeypatch.setattr(runner, "run_qa", fake_run_qa)
    monkeypatch.setattr(runner, "get_settings", lambda: type("Settings", (), {"use_local_models": False})())

    case = runner._run_case("eval-1", profile, question)

    assert case["model_call_count"] == 2
    assert case["model_calls"]["chat_calls"] == 2


def test_quality_metrics_match_stable_chunk_ids_and_dataset_leakage_constraints():
    cases = [
        {
            "answer": "answer",
            "latency_ms": 10,
            "expected": {
                "expect_references": True,
                "expected_chunk_ids": ["chunk-a", "chunk-b"],
                "allowed_document_ids": ["doc-public"],
                "retired_document_ids": ["doc-retired"],
            },
            "references": [
                {"chunk_id": "chunk-a", "document_id": "doc-public", "source_file": "same.md"},
                {"chunk_id": "chunk-x", "document_id": "doc-restricted", "source_file": "same.md"},
                {"chunk_id": "chunk-b", "document_id": "doc-retired", "source_file": "same.md"},
            ],
        }
    ]

    metrics = calculate_quality_metrics(cases)

    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["citation_precision"] == 0.6667
    assert metrics["acl_leakage_rate"] == 0.6667
    assert metrics["invalid_content_leakage_rate"] == 0.3333


def test_evaluation_case_preserves_dataset_identity_and_leakage_constraints(monkeypatch):
    profile = {"id": "baseline", "settings": RagSettings(), "collection_name": None}
    question = {
        "id": "case-1",
        "question": "测试问题",
        "expected_chunk_ids": ["chunk-a"],
        "allowed_document_ids": ["doc-a"],
        "retired_document_ids": ["doc-old"],
    }
    monkeypatch.setattr(
        runner,
        "run_qa",
        lambda state: {
            **state,
            "answer": "answer",
            "references": [{"chunk_id": "chunk-a", "document_id": "doc-a"}],
            "relation_paths": [],
            "error": None,
        },
    )
    monkeypatch.setattr(runner, "get_settings", lambda: type("Settings", (), {"use_local_models": False})())

    case = runner._run_case("eval-1", profile, question)

    assert case["expected"]["expected_chunk_ids"] == ["chunk-a"]
    assert case["expected"]["allowed_document_ids"] == ["doc-a"]
    assert case["expected"]["retired_document_ids"] == ["doc-old"]
