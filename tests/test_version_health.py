from __future__ import annotations

from pathlib import Path


def test_version_has_single_repository_source_and_frontend_uses_it():
    from services.rag_api.version import SOFTWARE_VERSION

    assert Path("VERSION").read_text(encoding="utf-8").strip() == "1.1.0"
    assert SOFTWARE_VERSION == "1.1.0"
    vite = Path("apps/web/vite.config.ts").read_text(encoding="utf-8")
    header = Path("apps/web/src/components/AppHeader.tsx").read_text(encoding="utf-8")
    assert "VERSION" in vite
    assert "__CRABRAG_VERSION__" in vite
    assert "__CRABRAG_VERSION_LABEL__" in header


def test_health_keeps_legacy_fields_and_adds_build_and_model_capabilities(monkeypatch, tmp_path: Path):
    from services.rag_api import main
    from services.rag_api.config import Settings

    monkeypatch.setattr(main, "get_settings", lambda: Settings(docs_dirs=[], docs_dir=tmp_path, api_key=None))
    monkeypatch.setattr(main, "collection_status", lambda: {"count": 0})
    monkeypatch.setattr(main, "local_model_capabilities", lambda: {"available": False, "models": []})

    payload = main.health()

    for legacy in ("web", "rag_service", "docs_dir_exists", "docs_dir_has_files", "docs_dirs", "chroma", "llm_api", "active_generation", "index_scheduler"):
        assert legacy in payload
    assert payload["software_version"] == "1.1.0"
    assert payload["build"]["version"] == "1.1.0"
    assert set(payload["model_capabilities"]) == {"remote", "local"}
    assert payload["model_capabilities"]["remote"]["configured"] is False
    assert payload["model_capabilities"]["local"]["available"] is False
