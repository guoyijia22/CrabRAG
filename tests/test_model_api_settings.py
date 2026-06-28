from __future__ import annotations

from pathlib import Path

import pytest

from services.rag_api import config
from services.rag_api import model_api_settings as model_settings
from services.rag_api.llm import siliconflow_client
from services.rag_api.retrieval import optimizations
from services.rag_api.rag_settings import RagSettings


def _isolate_model_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model_settings, "MODEL_API_SETTINGS_PATH", tmp_path / "model_api_settings.json")
    config.get_settings.cache_clear()


def test_model_api_settings_remote_mode_forces_api_embedding_provider(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            embedding_provider="local_onnx",
            clear_embedding_api_key=True,
            openai_compatible=True,
            embedding_openai_compatible=True,
        )
    )

    assert public.base_url == "https://chat.example/v1"
    assert public.chat_model == "deepseek-ai/DeepSeek-V3"
    assert public.embedding_provider == "api"
    assert public.embedding_base_url == "http://127.0.0.1:9997/v1"
    assert public.embedding_model == "BAAI/bge-m3"
    assert public.embedding_api_key_source == "missing"
    assert public.embedding_onnx_model_file == "model.onnx"
    assert public.rerank_onnx_model_file == "model.onnx"


def test_model_api_settings_save_local_onnx_model_files(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            embedding_provider="local_onnx",
            embedding_onnx_model_file="model_int8.onnx",
            rerank_onnx_model_file="model_int8.onnx",
            openai_compatible=True,
            embedding_openai_compatible=True,
        )
    )

    assert public.embedding_onnx_model_file == "model_int8.onnx"
    assert public.rerank_onnx_model_file == "model_int8.onnx"
    settings = config.get_settings()
    assert settings.embedding_onnx_model_file == "model_int8.onnx"
    assert settings.rerank_onnx_model_file == "model_int8.onnx"


def test_model_api_settings_use_local_models_defaults_false(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    (tmp_path / "model_api_settings.json").write_text(
        '{"api_key":"chat-secret","base_url":"https://chat.example/v1","openai_compatible":true}',
        encoding="utf-8",
    )

    public = model_settings.public_model_api_settings()
    settings = config.get_settings()

    assert public.use_local_models is False
    assert settings.use_local_models is False


def test_model_api_settings_use_local_models_switches_qwen_runtime_defaults(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            use_local_models=True,
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            embedding_provider="api",
            rerank_onnx_model_file="model_q4.onnx",
            openai_compatible=True,
            embedding_openai_compatible=True,
        )
    )

    settings = config.get_settings()

    assert public.use_local_models is True
    assert public.embedding_provider == "local_onnx"
    assert public.embedding_model == "BAAI/bge-m3"
    assert public.embedding_onnx_model_file == "model_int8.onnx"
    assert public.rerank_onnx_model_file == "model_q4.onnx"
    assert settings.use_local_models is True
    assert settings.embedding_provider == "local_onnx"
    assert settings.embedding_model == "Qwen3-Embedding-0.6B-ONNX"
    assert settings.embedding_onnx_model_file == "model_int8.onnx"
    assert settings.rerank_onnx_model_file == "model_q4.onnx"
    assert settings.local_llm_model_dir.name == "Qwen3___5-0___8B-ONNX"
    assert settings.local_embedding_model_dir.name == "Qwen3-Embedding-0___6B-ONNX"
    assert settings.local_rerank_model_dir.name == "Qwen3-Reranker-0___6B-ONNX"


def test_local_models_preserve_saved_embedding_model_but_runtime_uses_qwen(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            use_local_models=True,
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
        )
    )

    saved = model_settings.load_model_api_settings()
    settings = config.get_settings()

    assert public.embedding_model == "BAAI/bge-m3"
    assert saved.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_provider == "local_onnx"
    assert settings.embedding_model == "Qwen3-Embedding-0.6B-ONNX"


def test_local_model_status_reports_missing_models_and_download_links(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(model_settings, "PROJECT_ROOT", tmp_path)

    public = model_settings.public_model_api_settings()
    status = public.local_model_status
    by_key = {item.key: item for item in status.models}

    assert status.base_dir == str(tmp_path / "runtime" / "models")
    assert status.missing_count == 3
    assert by_key["llm"].present is False
    assert by_key["llm"].name == "Qwen3.5-0.8B-ONNX"
    assert by_key["llm"].expected_dir.endswith(r"runtime\models\Qwen3___5-0___8B-ONNX")
    assert "config.json" in by_key["llm"].missing_files
    assert "onnx/*.onnx" in by_key["llm"].missing_files
    assert by_key["llm"].download_urls.zh == "https://www.modelscope.cn/models/onnx-community/Qwen3.5-0.8B-ONNX"
    assert by_key["llm"].download_urls.en == "https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX"
    assert by_key["embedding"].download_urls.zh == "https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX"
    assert by_key["embedding"].download_urls.en == "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX"
    assert by_key["rerank"].download_urls.zh == "https://www.modelscope.cn/models/onnx-community/Qwen3-Reranker-0.6B-ONNX"
    assert by_key["rerank"].download_urls.en == "https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX"


def test_local_model_status_detects_required_model_files(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(model_settings, "PROJECT_ROOT", tmp_path)

    for directory, onnx_file in [
        ("Qwen3___5-0___8B-ONNX", "decoder_model_merged_q4.onnx"),
        ("Qwen3-Embedding-0___6B-ONNX", "model_int8.onnx"),
        ("Qwen3-Reranker-0___6B-ONNX", "model_q4.onnx"),
    ]:
        model_dir = tmp_path / "runtime" / "models" / directory
        (model_dir / "onnx").mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (model_dir / name).write_text("{}", encoding="utf-8")
        (model_dir / "onnx" / onnx_file).write_bytes(b"onnx")

    status = model_settings.public_model_api_settings().local_model_status

    assert status.missing_count == 0
    assert all(item.present for item in status.models)
    assert all(item.missing_files == [] for item in status.models)


def test_model_settings_api_responses_include_local_model_status(tmp_path, monkeypatch):
    from services.rag_api import main

    _isolate_model_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(model_settings, "PROJECT_ROOT", tmp_path)

    get_payload = main.get_model_settings()
    put_payload = main.update_model_settings(model_settings.ModelApiSettingsUpdate(use_local_models=True))

    assert get_payload.local_model_status.missing_count == 3
    assert put_payload.local_model_status.missing_count == 3


def test_chat_model_settings_ignore_env_values(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    monkeypatch.setenv("API_KEY", "env-chat-secret")
    monkeypatch.setenv("API_BASE_URL", "https://env-chat.example/v1")
    monkeypatch.setenv("MODEL_NAME", "env-chat-model")

    public = model_settings.public_model_api_settings()
    settings = config.get_settings()

    assert public.api_key_set is False
    assert public.api_key_source == "missing"
    assert public.base_url == model_settings.DEFAULT_BASE_URL
    assert public.chat_model == model_settings.DEFAULT_CHAT_MODEL
    assert settings.api_key is None
    assert settings.base_url == model_settings.DEFAULT_BASE_URL
    assert settings.chat_model == model_settings.DEFAULT_CHAT_MODEL


def test_legacy_openai_compatible_false_is_treated_as_openai(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    (tmp_path / "model_api_settings.json").write_text(
        '{"api_key":"chat-secret","base_url":"https://chat.example/v1","chat_model":"deepseek-ai/DeepSeek-V3","openai_compatible":false}',
        encoding="utf-8",
    )

    public = model_settings.public_model_api_settings()
    settings = config.get_settings()

    assert public.openai_compatible is True
    assert settings.openai_compatible is True


def test_remote_mode_forces_api_embedding_provider(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            use_local_models=False,
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_provider="local_onnx",
            embedding_base_url="https://embedding.example/v1",
            embedding_model="BAAI/bge-m3",
            embedding_onnx_model_file="model_int8.onnx",
            rerank_onnx_model_file="model_q4.onnx",
        )
    )

    settings = config.get_settings()

    assert public.embedding_provider == "api"
    assert settings.embedding_provider == "api"


def test_model_api_settings_reject_invalid_local_onnx_model_file():
    with pytest.raises(ValueError):
        model_settings.ModelApiSettingsUpdate(embedding_onnx_model_file="../x.onnx")


def test_runtime_settings_remote_mode_ignores_legacy_local_embedding_provider(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    model_settings.save_model_api_settings(
        model_settings.ModelApiSettings(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_provider="local_onnx",
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
    assert settings.embedding_provider == "api"
    assert settings.embedding_api_key == "embedding-secret"
    assert settings.embedding_base_url == "http://127.0.0.1:9997/v1"
    assert settings.embedding_model == "BAAI/bge-m3"


def test_model_settings_update_clears_local_qwen_worker(monkeypatch):
    from services.rag_api import main
    from services.rag_api.llm import local_qwen_llm

    called = {"shutdown": 0}

    monkeypatch.setattr(main, "update_model_api_settings", lambda settings: model_settings.public_model_api_settings())
    monkeypatch.setattr(main.get_settings, "cache_clear", lambda: None)
    monkeypatch.setattr(siliconflow_client.get_chat_client, "cache_clear", lambda: None)
    monkeypatch.setattr(siliconflow_client.get_embedding_client, "cache_clear", lambda: None)
    monkeypatch.setattr(local_qwen_llm, "shutdown_local_qwen_worker", lambda: called.update(shutdown=called["shutdown"] + 1))

    main.update_model_settings(model_settings.ModelApiSettingsUpdate(use_local_models=True))

    assert called["shutdown"] == 1


def test_health_reports_local_qwen_onnx(monkeypatch, tmp_path: Path):
    from services.rag_api import main

    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: config.Settings(use_local_models=True, docs_dirs=[tmp_path], docs_dir=tmp_path),
    )
    monkeypatch.setattr(main, "collection_status", lambda: {"count": 0})

    payload = main.health()

    assert payload["llm_api"] == "local_qwen_onnx"


def test_legacy_rerank_settings_fall_back_to_embedding_client(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    (tmp_path / "model_api_settings.json").write_text(
        """
        {
          "api_key": "chat-secret",
          "base_url": "https://chat.example/v1",
          "chat_model": "deepseek-ai/DeepSeek-V3",
          "embedding_api_key": "embedding-secret",
          "embedding_base_url": "http://127.0.0.1:9997/v1",
          "embedding_model": "BAAI/bge-m3"
        }
        """,
        encoding="utf-8",
    )

    public = model_settings.public_model_api_settings()
    settings = config.get_settings()

    assert public.rerank_base_url == "http://127.0.0.1:9997/v1"
    assert public.rerank_api_key_source == "settings"
    assert settings.rerank_base_url == "http://127.0.0.1:9997/v1"
    assert settings.rerank_api_key == "embedding-secret"


def test_model_api_settings_save_and_clear_rerank_client(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)

    public = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="embedding-secret",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            rerank_api_key="rerank-secret",
            rerank_base_url="https://rerank.example/v1",
        )
    )

    assert public.rerank_base_url == "https://rerank.example/v1"
    assert public.rerank_api_key_source == "settings"
    assert config.get_settings().rerank_api_key == "rerank-secret"

    cleared = model_settings.update_model_api_settings(
        model_settings.ModelApiSettingsUpdate(
            api_key="chat-secret",
            base_url="https://chat.example/v1",
            chat_model="deepseek-ai/DeepSeek-V3",
            embedding_api_key="embedding-secret",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_model="BAAI/bge-m3",
            clear_rerank_api_key=True,
            rerank_base_url="https://rerank.example/v1",
        )
    )

    assert cleared.rerank_base_url == "https://rerank.example/v1"
    assert cleared.rerank_api_key_source == "settings"
    config.get_settings.cache_clear()
    assert config.get_settings().rerank_api_key == "embedding-secret"


def test_legacy_model_settings_without_chat_model_uses_default_not_env(tmp_path, monkeypatch):
    _isolate_model_settings(tmp_path, monkeypatch)
    monkeypatch.setenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
    (tmp_path / "model_api_settings.json").write_text(
        '{"api_key":"chat-secret","base_url":"https://chat.example/v1","openai_compatible":true}',
        encoding="utf-8",
    )

    public = model_settings.public_model_api_settings()
    settings = config.get_settings()

    assert public.chat_model == model_settings.DEFAULT_CHAT_MODEL
    assert public.embedding_provider == "api"
    assert settings.chat_model == model_settings.DEFAULT_CHAT_MODEL


def test_chat_client_ignores_legacy_openai_compatible_false(monkeypatch):
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
            openai_compatible=False,
        ),
    )
    siliconflow_client.get_chat_client.cache_clear()

    siliconflow_client.get_chat_client()

    assert captured["api_key"] == "chat-secret"
    assert captured["base_url"] == "https://chat.example/v1"


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


def test_embedding_client_ignores_legacy_openai_compatible_false(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(siliconflow_client, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: config.Settings(
            embedding_api_key="embedding-secret",
            embedding_base_url="http://127.0.0.1:9997/v1",
            embedding_openai_compatible=False,
        ),
    )
    siliconflow_client.get_embedding_client.cache_clear()

    siliconflow_client.get_embedding_client()

    assert captured["api_key"] == "embedding-secret"
    assert captured["base_url"] == "http://127.0.0.1:9997/v1"


def test_rerank_uses_rerank_endpoint(monkeypatch):
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
            rerank_api_key="rerank-secret",
            rerank_base_url="https://rerank.example/v1",
        ),
    )

    reranked, trace = optimizations.apply_rerank(
        "资费标准",
        [{"content": "候选片段", "source_file": "doc.txt", "score": 0.5}],
        RagSettings(rerank_enabled=True, rerank_model="BAAI/bge-reranker-v2-m3"),
        top_k=1,
    )

    assert captured["url"] == "https://rerank.example/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer rerank-secret"
    assert captured["json"]["documents"] == ["候选片段"]
    assert reranked[0]["rerank_score"] == 0.99
    assert trace["returned_count"] == 1
