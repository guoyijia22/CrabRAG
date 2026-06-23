from __future__ import annotations

from pathlib import Path

from services.rag_api import config
from services.rag_api import model_api_settings as model_settings
from services.rag_api.llm import siliconflow_client
from services.rag_api.retrieval import optimizations
from services.rag_api.rag_settings import RagSettings


def _isolate_model_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model_settings, "MODEL_API_SETTINGS_PATH", tmp_path / "model_api_settings.json")
    config.get_settings.cache_clear()


def test_model_api_settings_save_split_chat_and_embedding_clients(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            clear_embedding_api_key=True,
            openai_compatible=True,
            embedding_openai_compatible=True,
        )
    )

    assert public.base_url == "https://chat.example/v1"
    assert public.chat_model == "deepseek-ai/DeepSeek-V3"
    assert public.embedding_base_url == "http://127.0.0.1:9997/v1"
    assert public.embedding_model == "BAAI/bge-m3"
    assert public.embedding_api_key_source == "missing"


def test_runtime_settings_use_separate_chat_and_embedding_clients(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    model_settings.save_model_api_settings(
        model_settings.ModelApiSettings(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="embedding-secret",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            embedding_openai_compatible=True,
        )
    )

    settings = config.get_settings()

    assert settings.api_key == "chat-secret"
    assert settings.base_url == "https://chat.example/v1"
    assert settings.chat_model == "deepseek-ai/DeepSeek-V3"
    assert settings.embedding_api_key == "embedding-secret"
    assert settings.embedding_base_url == "http://127.0.0.1:9997/v1"
    assert settings.embedding_model == "BAAI/bge-m3"


def test_legacy_model_settings_without_chat_model_fall_back_to_env(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    monkeypatch.setenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
    (tmp_path / "model_api_settings.json").write_text(
        '{"api_key":"chat-secret","base_url":"https://chat.example/v1","openai_compatible":true}',
        encoding="utf-8",
    )

    public = model_settings.public_model_api_settings()
    settings = config.get_settings()

    assert public.chat_model == "deepseek-ai/DeepSeek-V3"
    assert settings.chat_model == "deepseek-ai/DeepSeek-V3"


def test_embedding_client_uses_embedding_endpoint(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(siliconflow_client, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: config.Settings(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
        ),
    )
    siliconflow_client.get_embedding_client.cache_clear()

    siliconflow_client.get_embedding_client()

    assert captured["api_key"] == "local"
    assert captured["base_url"] == "http://127.0.0.1:9997/v1"


def test_rerank_uses_embedding_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.99}]}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(optimizations.requests, "post", fake_post)
    monkeypatch.setattr(
        optimizations,
        "get_settings",
        lambda: config.Settings(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            embedding_api_key="",
            embedding_base_url="http://127.0.0.1:9997/v1",
        ),
    )

    reranked, trace = optimizations.apply_rerank(
        "资费标准",
        [{"content": "候选片段", "source_file": "doc.txt", "score": 0.5}],
        RagSettings(rerank_enabled=True, rerank_model="BAAI/bge-reranker-v2-m3"),
        top_k=1,
    )

    assert captured["url"] == "http://127.0.0.1:9997/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer local"
    assert captured["json"]["documents"] == ["候选片段"]
    assert reranked[0]["rerank_score"] == 0.99
    assert trace["returned_count"] == 1
