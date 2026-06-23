from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(os.getenv("ELCQA_ROOT") or Path(__file__).resolve().parents[2]).resolve()
MODEL_API_SETTINGS_PATH = PROJECT_ROOT / "data" / "model_api_settings.json"
ENV_PATH = Path(os.getenv("ELCQA_ENV_FILE") or (PROJECT_ROOT / "config" / ".env" if os.getenv("ELCQA_ROOT") else r"D:\cd\.env")).resolve()

ApiKeySource = Literal["settings", "env", "missing"]

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHAT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class ModelApiSettings(BaseModel):
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    openai_compatible: bool = True
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_api_key: str = ""
    embedding_base_url: str = DEFAULT_BASE_URL
    embedding_openai_compatible: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @field_validator("api_key", "base_url", "chat_model", "embedding_api_key", "embedding_base_url", "embedding_model", mode="after")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class PublicModelApiSettings(BaseModel):
    api_key_set: bool
    api_key_source: ApiKeySource
    api_key_hint: str = ""
    base_url: str = DEFAULT_BASE_URL
    openai_compatible: bool = True
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_api_key_set: bool
    embedding_api_key_source: ApiKeySource
    embedding_api_key_hint: str = ""
    embedding_base_url: str = DEFAULT_BASE_URL
    embedding_openai_compatible: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


class ModelApiSettingsUpdate(BaseModel):
    api_key: str | None = Field(default=None, max_length=512)
    clear_api_key: bool = False
    base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1, max_length=300)
    openai_compatible: bool = True
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL, min_length=1, max_length=200)
    embedding_api_key: str | None = Field(default=None, max_length=512)
    clear_embedding_api_key: bool = False
    embedding_base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1, max_length=300)
    embedding_openai_compatible: bool = True
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1, max_length=200)

    @field_validator("api_key", "base_url", "chat_model", "embedding_api_key", "embedding_base_url", "embedding_model", mode="after")
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
    saved = save_model_api_settings(
        ModelApiSettings(
            api_key=api_key,
            base_url=update.base_url,
            openai_compatible=update.openai_compatible,
            chat_model=update.chat_model,
            embedding_api_key=embedding_api_key,
            embedding_base_url=update.embedding_base_url,
            embedding_openai_compatible=update.embedding_openai_compatible,
            embedding_model=update.embedding_model,
        )
    )
    return public_model_api_settings(saved)


def public_model_api_settings(settings: ModelApiSettings | None = None) -> PublicModelApiSettings:
    active = settings or load_model_api_settings()
    source, hint = _key_source_and_hint(active.api_key, _env_api_key())
    embedding_source, embedding_hint = _key_source_and_hint(active.embedding_api_key, _env_embedding_api_key())
    return PublicModelApiSettings(
        api_key_set=source != "missing",
        api_key_source=source,
        api_key_hint=hint,
        base_url=effective_base_url(active),
        openai_compatible=active.openai_compatible,
        chat_model=effective_chat_model(active),
        embedding_api_key_set=embedding_source != "missing",
        embedding_api_key_source=embedding_source,
        embedding_api_key_hint=embedding_hint,
        embedding_base_url=effective_embedding_base_url(active),
        embedding_openai_compatible=active.embedding_openai_compatible,
        embedding_model=effective_embedding_model(active),
    )


def effective_api_key() -> str | None:
    active = load_model_api_settings()
    return active.api_key or _env_api_key()


def effective_base_url(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and active.base_url:
        return active.base_url
    return _env_base_url() or active.base_url or DEFAULT_BASE_URL


def effective_openai_compatible() -> bool:
    return load_model_api_settings().openai_compatible


def effective_chat_model(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and _settings_file_has_key("chat_model") and active.chat_model:
        return active.chat_model
    return _env_chat_model() or active.chat_model or DEFAULT_CHAT_MODEL


def effective_embedding_api_key() -> str | None:
    active = load_model_api_settings()
    return active.embedding_api_key or _env_embedding_api_key()


def effective_embedding_base_url(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and active.embedding_base_url:
        return active.embedding_base_url
    return _env_embedding_base_url() or active.embedding_base_url or effective_base_url(active)


def effective_embedding_openai_compatible() -> bool:
    return load_model_api_settings().embedding_openai_compatible


def effective_embedding_model(settings: ModelApiSettings | None = None) -> str:
    active = settings or load_model_api_settings()
    if MODEL_API_SETTINGS_PATH.exists() and _settings_file_has_key("embedding_model") and active.embedding_model:
        return active.embedding_model
    return _env_embedding_model() or active.embedding_model or DEFAULT_EMBEDDING_MODEL


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


def _env_base_url() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["SILICONFLOW_BASE_URL", "SILICON_FLOW_BASE_URL", "OPENAI_BASE_URL", "API_BASE_URL"]:
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


def _env_chat_model() -> str | None:
    load_dotenv(ENV_PATH)
    for name in ["SILICONFLOW_CHAT_MODEL", "OPENAI_MODEL", "CHAT_MODEL", "MODEL_NAME"]:
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
