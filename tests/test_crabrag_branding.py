from __future__ import annotations

from pathlib import Path

import pytest


def test_app_settings_default_ui_language_is_english():
    from services.rag_api.app_settings import AppSettings

    assert AppSettings().ui_language == "en"


def test_app_settings_rejects_unknown_ui_language():
    from pydantic import ValidationError
    from services.rag_api.app_settings import AppSettings

    with pytest.raises(ValidationError):
        AppSettings(ui_language="fr")


def test_public_app_config_includes_ui_language(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    settings_path = tmp_path / "data" / "app_settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        """
        {
          "system_name": "CrabRAG 通用基础查询",
          "knowledge_base_name": "自定义知识库",
          "ui_language": "zh",
          "knowledge_base_dirs": ["docs"]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", settings_path)

    payload = app_settings.read_public_app_config()

    assert payload["ui_language"] == "zh"


def test_crabrag_root_env_takes_priority_over_legacy_env(monkeypatch, tmp_path: Path):
    from services.rag_api import paths

    crabrag_root = tmp_path / "CrabRAG"
    legacy_root = tmp_path / "QueryBasePortableLab"
    monkeypatch.setenv("CRABRAG_ROOT", str(crabrag_root))
    monkeypatch.setenv("ELCQA_ROOT", str(legacy_root))

    assert paths.resolve_project_root(tmp_path / "fallback") == crabrag_root.resolve()


def test_crabrag_docs_dir_takes_priority_over_legacy_docs_env(monkeypatch, tmp_path: Path):
    from services.rag_api import app_settings

    crabrag_docs = tmp_path / "crab-docs"
    legacy_docs = tmp_path / "legacy-docs"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setenv("CRABRAG_DOCS_DIR", f"{crabrag_docs};{crabrag_docs}")
    monkeypatch.setenv("ELCQA_DOCS_DIR", str(legacy_docs))

    assert app_settings.default_knowledge_base_dirs() == [str(crabrag_docs.resolve())]


def test_old_default_system_name_migrates_to_crabrag(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    settings_path = tmp_path / "data" / "app_settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        """
        {
          "system_name": "QueryBasePortableLab 通用基础查询",
          "knowledge_base_name": "自定义知识库",
          "knowledge_base_dirs": ["docs"]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", settings_path)

    settings = app_settings.load_app_settings()

    assert settings.system_name == "CrabRAG 通用基础查询"
    assert settings.knowledge_base_name == "自定义知识库"


def test_custom_system_name_is_not_overwritten(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    settings_path = tmp_path / "data" / "app_settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        """
        {
          "system_name": "我的业务知识库助手",
          "knowledge_base_name": "自定义知识库",
          "knowledge_base_dirs": ["docs"]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", settings_path)

    settings = app_settings.load_app_settings()

    assert settings.system_name == "我的业务知识库助手"


def test_root_service_uses_crabrag_brand():
    from services.rag_api import main

    payload = main.root()

    assert payload["service"] == "crabrag-api"


def test_crabrag_skill_files_point_to_cli_not_chat_endpoint():
    uniclaw_skill = Path("crabrag.skill").read_text(encoding="utf-8")
    crabrag_skill = Path("skills/crabrag-rag/SKILL.md").read_text(encoding="utf-8")

    assert "CRABRAG_HOME" in uniclaw_skill
    assert "crab-rag.bat" in uniclaw_skill
    assert "evidence[].content" in uniclaw_skill
    assert "不要调用 CrabRAG 的 `/api/chat` 接口" in uniclaw_skill
    assert "CRABRAG_HOME" in crabrag_skill
    assert "crab-rag.bat" in crabrag_skill
    assert "不要调用 CrabRAG 的 `/api/chat` 接口" in crabrag_skill
    assert not Path("skills/query-base-rag/SKILL.md").exists()


def test_crabrag_cli_entry_exists_and_uses_crabrag_root():
    cli_text = Path("crab-rag.bat").read_text(encoding="utf-8")

    assert "CRABRAG_ROOT" in cli_text
    assert "services.rag_api.cli.evidence" in cli_text
    assert not Path("query-rag.bat").exists()


def test_default_web_port_is_3003():
    gateway_text = Path("server/gateway.js").read_text(encoding="utf-8")
    run_text = Path("run.ps1").read_text(encoding="utf-8")
    api_text = Path("services/rag_api/main.py").read_text(encoding="utf-8")
    readme_text = Path("README_PORTABLE.md").read_text(encoding="utf-8")

    assert "process.env.PORT ?? 3003" in gateway_text
    assert "[int]$WebPort = 3003" in run_text
    assert '$env:PORT = "$WebPort"' in run_text
    assert "http://127.0.0.1:$WebPort/" in run_text
    assert "http://127.0.0.1:3003" in api_text
    assert "http://127.0.0.1:3003" in readme_text


def test_frontend_top_nav_shows_app_version():
    bundle_text = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")
    css_text = Path("apps/web/dist/assets/index-C1LwUchF.css").read_text(encoding="utf-8")

    assert "className:`app-version`,children:`Ver1.00`" in bundle_text
    assert ".app-version{color:#a8a8a8" in css_text
