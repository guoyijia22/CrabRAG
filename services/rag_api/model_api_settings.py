from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from services.rag_api.paths import resolve_env_file, resolve_project_root

PROJECT_ROOT = resolve_project_root(Path(__file__).resolve().parents[2])
MODEL_API_SETTINGS_PATH = PROJECT_ROOT / "data" / "model_api_settings.json"
ENV_PATH = resolve_env_file(PROJECT_ROOT, Path(r"D:\cd\.env"))

ApiKeySource = Literal["settings", "env", "missing"]
EmbeddingProvider = Literal["api", "local_onnx"]
OnnxModelFile = Literal["model.onnx", "model_fp16.onnx", "model_int8.onnx", "model_q4.onnx"]

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHAT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_PROVIDER: EmbeddingProvider = "api"
DEFAULT_ONNX_MODEL_FILE: OnnxModelFile = "model.onnx"
LOCAL_QWEN_LLM_MODEL = "Qwen3.5-0.8B-ONNX"
LOCAL_QWEN_LLM_MODEL_DIR_NAME = "Qwen3___5-0___8B-ONNX"
LOCAL_QWEN_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-ONNX"
LOCAL_QWEN_EMBEDDING_MODEL_DIR_NAME = "Qwen3-Embedding-0___6B-ONNX"
LOCAL_QWEN_EMBEDDING_ONNX_MODEL_FILE: OnnxModelFile = "model_int8.onnx"
LOCAL_QWEN_RERANK_MODEL = "Qwen3-Reranker-0.6B-ONNX"
LOCAL_QWEN_RERANK_MODEL_DIR_NAME = "Qwen3-Reranker-0___6B-ONNX"
LOCAL_QWEN_RERANK_ONNX_MODEL_FILE: OnnxModelFile = "model_q4.onnx"
LOCAL_MODEL_CONFIG_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
LOCAL_MODEL_DOWNLOAD_URLS = {
    "llm": {
        "zh": "https://www.modelscope.cn/models/onnx-community/Qwen3.5-0.8B-ONNX",
        "en": "https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX",
    },
    "embedding": {
        "zh": "https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX",
        "en": "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX",
    },
    "rerank": {
        "zh": "https://www.modelscope.cn/models/onnx-community/Qwen3-Reranker-0.6B-ONNX",
        "en": "https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX",
    },
}


class LocalModelDownloadUrls(BaseModel):
    zh: str
    en: str


class LocalModelStatusItem(BaseModel):
    key: Literal["llm", "embedding", "rerank"]
    name: str
    present: bool
    expected_dir: str
    required_files: list[str]
    missing_files: list[str]
    download_urls: LocalModelDownloadUrls


class LocalModelStatus(BaseModel):
    base_dir: str
    missing_count: int
    models: list[LocalModelStatusItem]


class ModelApiSettings(BaseModel):
    use_local_models: bool = False
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    openai_compatible: bool = True
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER
    embedding_api_key: str = ""
    embedding_base_url: str = DEFAULT_BASE_URL
    embedding_openai_compatible: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_onnx_model_file: OnnxModelFile = DEFAULT_ONNX_MODEL_FILE
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_onnx_model_file: OnnxModelFile = DEFAULT_ONNX_MODEL_FILE

    @field_validator("api_key", "base_url", "chat_model", "embedding_api_key", "embedding_base_url", "embedding_model", "rerank_api_key", "rerank_base_url", mode="after")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class PublicModelApiSettings(BaseModel):
    use_local_models: bool = False
    api_key_set: bool
    api_key_source: ApiKeySource
    api_key_hint: str = ""
    base_url: str = DEFAULT_BASE_URL
    openai_compatible: bool = True
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_api_key_set: bool
    embedding_api_key_source: ApiKeySource
    embedding_api_key_hint: str = ""
    embedding_provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER
    embedding_base_url: str = DEFAULT_BASE_URL
    embedding_openai_compatible: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_onnx_model_file: OnnxModelFile = DEFAULT_ONNX_MODEL_FILE
    rerank_api_key_set: bool
    rerank_api_key_source: ApiKeySource
    rerank_api_key_hint: str = ""
    rerank_base_url: str = DEFAULT_BASE_URL
    rerank_onnx_model_file: OnnxModelFile = DEFAULT_ONNX_MODEL_FILE
    local_model_status: LocalModelStatus


class ModelApiSettingsUpdate(BaseModel):
    use_local_models: bool = False
    api_key: str | None = Field(default=None, max_length=512)
    clear_api_key: bool = False
    base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1, max_length=300)
    openai_compatible: bool = True
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1, max_length=200)
    embedding_provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER
    embedding_api_key: str | None = Field(default=None, max_length=512)
    clear_embedding_api_key: bool = False
    embedding_base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1, max_length=300)
    embedding_openai_compatible: bool = True
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1, max_length=200)
    embedding_onnx_model_file: OnnxModelFile = DEFAULT_ONNX_MODEL_FILE
    rerank_api_key: str | None = Field(default=None, max_length=512)
    clear_rerank_api_key: bool = False
    rerank_base_url: str | None = Field(default=None, max_length=300)
    rerank_onnx_model_file: OnnxModelFile = DEFAULT_ONNX_MODEL_FILE

    @field_validator("api_key", "base_url", "chat_model", "embedding_api_key", "embedding_base_url", "embedding_model", "rerank_api_key", "rerank_base_url", mode="after")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


def load_model_api_settings() -> ModelApiSettings:
    if MODEL_API_SETTINGS_PATH.exists():
        try:
            return ModelApiSettings.model_validate_json(MODEL_API_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return ModelApiSettings()
    return ModelApiSettings()


def save_model_api_settings(settings: ModelApiSettings) -> ModelApiSettings:
    normalized = ModelApiSettings.model_validate(settings.model_dump())
    MODEL_API_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_API_SETTINGS_PATH.write_text(json.dumps(normalized.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def update_model_api_settings(update: ModelApiSettingsUpdate) -> PublicModelApiSettings:
    current = load_model_api_settings()
    api_key = current.api_key
    if update.clear_api_key:
        api_key = ""
    elif update.api_key:
        api_key = update.api_key
    embedding_api_key = current.embedding_api_key
    if update.clear_embedding_api_key:
        embedding_api_key = ""
    elif update.embedding_api_key:
        embedding_api_key = update.embedding_api_key
    rerank_api_key = current.rerank_api_key
    if update.clear_rerank_api_key:
        rerank_api_key = ""
    elif update.rerank_api_key:
        rerank_api_key = update.rerank_api_key
    rerank_base_url = update.rerank_base_url if update.rerank_base_url is not None else current.rerank_base_url
    embedding_provider: EmbeddingProvider = "api"
    embedding_model = update.embedding_model
    embedding_onnx_model_file = update.embedding_onnx_model_file
    rerank_onnx_model_file = update.rerank_onnx_model_file
    if update.use_local_models:
        embedding_provider = "local_onnx"
        embedding_onnx_model_file = LOCAL_QWEN_EMBEDDING_ONNX_MODEL_FILE
        rerank_onnx_model_file = LOCAL_QWEN_RERANK_ONNX_MODEL_FILE
    saved = save_model_api_settings(
        ModelApiSettings(
            use_local_models=update.use_local_models,
            api_key=api_key,
            base_url=update.base_url,
            openai_compatible=True,
            chat_model=update.chat_model,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_base_url=update.embedding_base_url,
            embedding_openai_compatible=update.embedding_openai_compatible,
            embedding_model=embedding_model,
            embedding_onnx_model_file=embedding_onnx_model_file,
            rerank_api_key=rerank_api_key,
            rerank_base_url=rerank_base_url,
            rerank_onnx_model_file=rerank_onnx_model_file,
        )
    )
    return public_model_api_settings(saved)


def public_model_api_settings(settings: ModelApiSettings | None = None) -> PublicModelApiSettings:
    active = settings or load_model_api_settings()
    source, hint = _key_source_and_hint(active.api_key, None)
    embedding_source, embedding_hint = _key_source_and_hint(active.embedding_api_key, _env_embedding_api_key())
    rerank_api_key, rerank_source, rerank_hint = _effective_rerank_api_key_with_source(active)
    return PublicModelApiSettings(
        use_local_models=effective_use_local_models(active),
        api_key_set=source != "missing",
        api_key_source=source,
        api_key_hint=hint,
        base_url=effective_base_url(active),
        openai_compatible=True,
        chat_model=effective_chat_model(active),
        embedding_api_key_set=embedding_source != "missing",
        embedding_api_key_source=embedding_source,
        embedding_api_key_hint=embedding_hint,
        embedding_provider=effective_embedding_provider(active),
        embedding_base_url=effective_embedding_base_url(active),
        embedding_openai_compatible=effective_embedding_openai_compatible(),
        embedding_model=(active.embedding_model or DEFAULT_EMBEDDING_MODEL) if effective_use_local_models(active) else effective_embedding_model(active),
        embedding_onnx_model_file=effective_embedding_onnx_model_file(active),
        rerank_api_key_set=rerank_api_key is not None,
        rerank_api_key_source=rerank_source,
        rerank_api_key_hint=rerank_hint,
        rerank_base_url=effective_rerank_base_url(active),
        rerank_onnx_model_file=effective_rerank_onnx_model_file(active),
        local_model_status=build_local_model_status(),
    )


def build_local_model_status() -> LocalModelStatus:
    base_dir = PROJECT_ROOT / "runtime" / "models"
    models = [
        _local_model_status_item(
            key="llm",
            name=LOCAL_QWEN_LLM_MODEL,
            model_dir=base_dir / LOCAL_QWEN_LLM_MODEL_DIR_NAME,
            required_files=[*LOCAL_MODEL_CONFIG_FILES, "onnx/*.onnx"],
            any_onnx=True,
        ),
        _local_model_status_item(
            key="embedding",
            name=LOCAL_QWEN_EMBEDDING_MODEL,
            model_dir=base_dir / LOCAL_QWEN_EMBEDDING_MODEL_DIR_NAME,
            required_files=[*LOCAL_MODEL_CONFIG_FILES, f"onnx/{LOCAL_QWEN_EMBEDDING_ONNX_MODEL_FILE}"],
            onnx_file=LOCAL_QWEN_EMBEDDING_ONNX_MODEL_FILE,
        ),
        _local_model_status_item(
            key="rerank",
            name=LOCAL_QWEN_RERANK_MODEL,
            model_dir=base_dir / LOCAL_QWEN_RERANK_MODEL_DIR_NAME,
            required_files=[*LOCAL_MODEL_CONFIG_FILES, f"onnx/{LOCAL_QWEN_RERANK_ONNX_MODEL_FILE}"],
            onnx_file=LOCAL_QWEN_RERANK_ONNX_MODEL_FILE,
        ),
    ]
    return LocalModelStatus(
        base_dir=str(base_dir),
        missing_count=sum(1 for item in models if not item.present),
        models=models,
    )


def _local_model_status_item(
    *,
    key: Literal["llm", "embedding", "rerank"],
    name: str,
    model_dir: Path,
    required_files: list[str],
    onnx_file: str | None = None,
    any_onnx: bool = False,
) -> LocalModelStatusItem:
    missing = [file_name for file_name in LOCAL_MODEL_CONFIG_FILES if not (model_dir / file_name).exists()]
    if any_onnx and not _has_any_onnx_file(model_dir):
        missing.append("onnx/*.onnx")
    if onnx_file and not _has_onnx_file(model_dir, onnx_file):
        missing.append(f"onnx/{onnx_file}")
    return LocalModelStatusItem(
        key=key,
        name=name,
        present=not missing,
        expected_dir=str(model_dir),
        required_files=required_files,
        missing_files=missing,
        download_urls=LocalModelDownloadUrls.model_validate(LOCAL_MODEL_DOWNLOAD_URLS[key]),
    )


def _has_any_onnx_file(model_dir: Path) -> bool:
    onnx_dir = model_dir / "onnx"
    return onnx_dir.exists() and any(path.is_file() for path in onnx_dir.glob("*.onnx"))


def _has_onnx_file(model_dir: Path, onnx_file: str) -> bool:
    return (model_dir / onnx_file).exists() or (model_dir / "onnx" / onnx_file).exists()


def effective_api_key() -> str | None:
    active = load_model_api_settings()
    return active.api_key or None


def effective_use_local_models(settings: ModelApiSettings | None = None) -> bool:
    return bool((settings or load_model_api_settings()).use_local_models)


def effective_base_url(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and active.base_url:
        return active.base_url
    return active.base_url or DEFAULT_BASE_URL


def effective_openai_compatible() -> bool:
    return True


def effective_chat_model(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and _settings_file_has_key("chat_model") and active.chat_model:
        return active.chat_model
    return active.chat_model or DEFAULT_CHAT_MODEL


def effective_embedding_api_key() -> str | None:
    active = load_model_api_settings()
    return active.embedding_api_key or _env_embedding_api_key()


def effective_embedding_provider(settings: ModelApiSettings | None = None) -> EmbeddingProvider:
    active = settings or load_model_api_settings()
    if effective_use_local_models(active):
        return "local_onnx"
    return "api"


def effective_embedding_base_url(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and active.embedding_base_url:
        return active.embedding_base_url
    return _env_embedding_base_url() or active.embedding_base_url or effective_base_url(active)


def effective_embedding_openai_compatible() -> bool:
    return True


def effective_embedding_model(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if effective_use_local_models(active):
        return LOCAL_QWEN_EMBEDDING_MODEL
    if MODEL_API_SETTINGS_PATH.exists() and _settings_file_has_key("embedding_model") and active.embedding_model:
        return active.embedding_model
    return _env_embedding_model() or active.embedding_model or DEFAULT_EMBEDDING_MODEL


def effective_embedding_onnx_model_file(settings: ModelApiSettings | None = None) -> OnnxModelFile:
    active = settings or load_model_api_settings()
    if effective_use_local_models(active):
        return LOCAL_QWEN_EMBEDDING_ONNX_MODEL_FILE
    return active.embedding_onnx_model_file or DEFAULT_ONNX_MODEL_FILE


def effective_rerank_api_key() -> str | None:
    return _effective_rerank_api_key_with_source(load_model_api_settings())[0]


def effective_rerank_base_url(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and _settings_file_has_key("rerank_base_url") and active.rerank_base_url:
        return active.rerank_base_url
    return _env_rerank_base_url() or active.rerank_base_url or effective_embedding_base_url(active)


def effective_rerank_onnx_model_file(settings: ModelApiSettings | None = None) -> OnnxModelFile:
    active = settings or load_model_api_settings()
    if effective_use_local_models(active):
        return LOCAL_QWEN_RERANK_ONNX_MODEL_FILE
    return active.rerank_onnx_model_file or DEFAULT_ONNX_MODEL_FILE


def _env_api_key() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["SILICONFLOW_API_KEY", "SILICON_FLOW_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_embedding_api_key() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["EMBEDDING_API_KEY", "RETRIEVAL_API_KEY", "SILICONFLOW_EMBEDDING_API_KEY"]:
        value = os.getenv(name)
        if value:
            return value
    return _env_api_key()


def _env_rerank_api_key() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["RERANK_API_KEY", "SILICONFLOW_RERANK_API_KEY"]:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_embedding_base_url() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["EMBEDDING_BASE_URL", "RETRIEVAL_BASE_URL", "SILICONFLOW_EMBEDDING_BASE_URL"]:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_rerank_base_url() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["RERANK_BASE_URL", "SILICONFLOW_RERANK_BASE_URL"]:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_embedding_model() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["SILICONFLOW_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL"]:
        value = os.getenv(name)
        if value:
            return value
    return None


def _key_source_and_hint(settings_key: str, env_key: str | None) -> tuple[ApiKeySource, str]:
    if settings_key:
        return "settings", _mask_api_key(settings_key)
    if env_key:
        return "env", "已通过环境变量配置"
    return "missing", ""


def _effective_rerank_api_key_with_source(settings: ModelApiSettings) -> tuple[str | None, ApiKeySource, str]:
    if settings.rerank_api_key:
        return settings.rerank_api_key, "settings", _mask_api_key(settings.rerank_api_key)
    env_rerank_api_key = _env_rerank_api_key()
    if env_rerank_api_key:
        return env_rerank_api_key, "env", "已通过环境变量配置"
    if settings.embedding_api_key:
        return settings.embedding_api_key, "settings", _mask_api_key(settings.embedding_api_key)
    env_embedding_api_key = _env_embedding_api_key()
    if env_embedding_api_key:
        return env_embedding_api_key, "env", "已通过环境变量配置"
    return None, "missing", ""


def _settings_file_has_key(key: str) -> bool:
    if not MODEL_API_SETTINGS_PATH.exists():
        return False
    try:
        payload = json.loads(MODEL_API_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and key in payload


def _mask_api_key(value: str) -> str:
    if len(value) <= 8:
        return "已配置"
    return f"{value[:4]}****{value[-4:]}"
