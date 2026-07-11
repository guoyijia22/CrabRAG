from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from services.rag_api import main, rag_settings
from services.rag_api.evaluation import approval
from services.rag_api.rag_settings import RagSettings


def _passing_run(settings: RagSettings) -> dict:
    return {
        "run_id": "eval-1",
        "generation_id": "gen-1",
        "question_generation": {
            "fixed": True,
            "gate_eligible": True,
            "dataset_id": "quality",
            "dataset_version": "1",
            "dataset_fingerprint": "dataset-sha",
        },
        "profiles": [
            {
                "id": "rerank_enabled",
                "enabled_switches": ["rerank_enabled"],
                "settings": settings.model_dump(),
                "summary": {"quality_gate": {"eligible": True, "passed": True}},
            },
            {
                "id": "failed",
                "enabled_switches": ["query_expansion_enabled"],
                "settings": RagSettings(query_expansion_enabled=True).model_dump(),
                "summary": {"quality_gate": {"eligible": True, "passed": False}},
            },
        ],
    }


def test_passing_fixed_evaluation_records_exact_settings_approval(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(approval, "APPROVALS_PATH", tmp_path / "approvals.json")
    candidate = RagSettings(rerank_enabled=True)

    recorded = approval.record_quality_approvals(_passing_run(candidate))

    assert recorded == ["rerank_enabled"]
    assert approval.is_settings_approved(candidate, "gen-1") is True
    assert approval.is_settings_approved(candidate.model_copy(update={"top_k": 3}), "gen-1") is False
    assert approval.is_settings_approved(candidate, "gen-2") is False


def test_dynamic_evaluation_never_records_strategy_approval(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(approval, "APPROVALS_PATH", tmp_path / "approvals.json")
    payload = _passing_run(RagSettings(rerank_enabled=True))
    payload["question_generation"]["fixed"] = False

    assert approval.record_quality_approvals(payload) == []
    assert not (tmp_path / "approvals.json").exists()


def test_settings_update_rejects_new_strategy_without_matching_approval(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rag_settings, "SETTINGS_PATH", tmp_path / "rag_settings.json")
    monkeypatch.setattr(main, "get_settings", lambda: type("Settings", (), {"use_local_models": False})())
    monkeypatch.setattr(main.index_generation, "active_generation_id", lambda: "gen-1")
    monkeypatch.setattr(approval, "APPROVALS_PATH", tmp_path / "approvals.json")

    with pytest.raises(HTTPException) as exc_info:
        main.update_rag_settings(RagSettings(rerank_enabled=True))

    assert exc_info.value.status_code == 409
    assert "固定评测" in str(exc_info.value.detail)
    assert rag_settings.load_rag_settings().rerank_enabled is False


def test_settings_update_accepts_exact_approved_strategy(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rag_settings, "SETTINGS_PATH", tmp_path / "rag_settings.json")
    monkeypatch.setattr(main, "get_settings", lambda: type("Settings", (), {"use_local_models": False})())
    monkeypatch.setattr(main.index_generation, "active_generation_id", lambda: "gen-1")
    monkeypatch.setattr(approval, "APPROVALS_PATH", tmp_path / "approvals.json")
    candidate = RagSettings(rerank_enabled=True)
    approval.record_quality_approvals(_passing_run(candidate))

    saved = main.update_rag_settings(candidate)

    assert saved.rerank_enabled is True
