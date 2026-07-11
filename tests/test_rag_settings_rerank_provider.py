from __future__ import annotations

from fastapi.testclient import TestClient

from services.rag_api import config, main, rag_settings
from services.rag_api.main import app


def test_legacy_rag_settings_default_rerank_provider_api(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_settings, "SETTINGS_PATH", tmp_path / "rag_settings.json")
    (tmp_path / "rag_settings.json").write_text(
        '{"rerank_enabled": true, "rerank_model": "BAAI/bge-reranker-v2-m3"}',
        encoding="utf-8",
    )

    settings = rag_settings.load_rag_settings()

    assert settings.rerank_provider == "api"


def test_settings_api_forces_api_rerank_provider_when_remote_models(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_settings, "SETTINGS_PATH", tmp_path / "rag_settings.json")
    monkeypatch.setattr(main, "get_settings", lambda: config.Settings(use_local_models=False))
    client = TestClient(app)

    current = client.get("/api/settings")
    assert current.status_code == 200
    payload = current.json()
    assert payload["rerank_provider"] == "api"

    payload["rerank_enabled"] = True
    payload["rerank_provider"] = "local_onnx"
    updated = client.put("/api/settings", json=payload)

    assert updated.status_code == 200
    assert updated.json()["rerank_provider"] == "api"
    assert rag_settings.load_rag_settings().rerank_provider == "api"


def test_settings_api_preserves_rerank_model_when_local_models(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_settings, "SETTINGS_PATH", tmp_path / "rag_settings.json")
    monkeypatch.setattr(main, "get_settings", lambda: config.Settings(use_local_models=True))
    client = TestClient(app)

    payload = client.get("/api/settings").json()
    payload["rerank_enabled"] = True
    payload["rerank_provider"] = "local_onnx"
    payload["rerank_model"] = "BAAI/bge-reranker-v2-m3"
    updated = client.put("/api/settings", json=payload)

    assert updated.status_code == 200
    assert updated.json()["rerank_provider"] == "local_onnx"
    assert updated.json()["rerank_model"] == "BAAI/bge-reranker-v2-m3"
    assert rag_settings.load_rag_settings().rerank_model == "BAAI/bge-reranker-v2-m3"
