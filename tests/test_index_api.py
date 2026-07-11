from __future__ import annotations


def _configure_index(tmp_path, monkeypatch):
    from services.rag_api import index_generation

    root = tmp_path / "index"
    monkeypatch.setattr(index_generation, "INDEX_ROOT", root)
    monkeypatch.setattr(index_generation, "ACTIVE_INDEX_PATH", root / "active.json")
    monkeypatch.setattr(index_generation, "GENERATIONS_DIR", root / "generations")
    index_generation.publish_generation(
        "gen-1",
        {"permission_schema_version": 1, "stats": {"chunk_count": 2}, "warnings": []},
    )
    index_generation.publish_generation(
        "gen-2",
        {
            "permission_schema_version": 1,
            "next_activation_at": "2026-08-01T00:00:00Z",
            "stats": {"chunk_count": 3, "reused_embedding_count": 2, "embedded_chunk_count": 1},
            "warnings": [{"code": "AUTO_PUBLIC_DOCUMENT", "path": "a.txt"}],
        },
    )
    return index_generation


def test_index_status_returns_current_previous_cache_and_scheduler(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main

    _configure_index(tmp_path, monkeypatch)
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    monkeypatch.setattr(main.INDEX_SCHEDULER, "status", lambda: {"running": True, "next_activation_at": "2026-08-01T00:00:00Z"}, raising=False)

    response = TestClient(main.app).get(
        "/api/index/status",
        headers={"x-crabrag-internal-token": "trusted", "x-crabrag-subject": "admin", "x-crabrag-admin": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_generation"] == "gen-2"
    assert payload["previous_generation"] == "gen-1"
    assert payload["can_rollback"] is True
    assert payload["active"]["stats"]["reused_embedding_count"] == 2
    assert payload["scheduler"]["running"] is True
    assert "cache" in payload


def test_index_rollback_requires_admin_and_swaps_generations(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main

    index_generation = _configure_index(tmp_path, monkeypatch)
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    monkeypatch.setattr(main, "validate_generation_collections", lambda generation_id, manifest: None)
    client = TestClient(main.app)

    forbidden = client.post(
        "/api/index/rollback",
        headers={"x-crabrag-internal-token": "trusted", "x-crabrag-subject": "user"},
    )
    rolled_back = client.post(
        "/api/index/rollback",
        headers={"x-crabrag-internal-token": "trusted", "x-crabrag-subject": "admin", "x-crabrag-admin": "true"},
    )

    assert forbidden.status_code == 403
    assert rolled_back.status_code == 200
    assert index_generation.load_index_state()["active_generation"] == "gen-1"


def test_index_management_logs_and_evaluations_reject_non_admin(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main

    _configure_index(tmp_path, monkeypatch)
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    client = TestClient(main.app)
    headers = {"x-crabrag-internal-token": "trusted", "x-crabrag-subject": "user"}

    assert client.post("/api/ingest", headers=headers).status_code == 403
    assert client.get("/api/logs", headers=headers).status_code == 403
    assert client.get("/api/evaluations", headers=headers).status_code == 403
