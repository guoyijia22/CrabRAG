from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from services.rag_api.app_settings import (
    DEFAULT_KNOWLEDGE_BASE_NAME,
    DEFAULT_SYSTEM_NAME,
    read_public_app_config,
    update_common_questions,
    update_knowledge_base_name,
    update_system_name,
)
from services.rag_api.model_api_settings import (
    effective_api_key,
    effective_base_url,
    effective_chat_model,
    effective_embedding_api_key,
    effective_embedding_base_url,
    effective_embedding_model,
    effective_embedding_openai_compatible,
    effective_openai_compatible,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(os.getenv("ELCQA_ROOT") or ROOT_DIR).resolve()
LEGACY_ENV_PATH = Path(r"D:\cd\.env")
LEGACY_DOCS_DIR = Path(r"D:\cd\docs")
ENV_PATH = Path(os.getenv("ELCQA_ENV_FILE") or (PROJECT_DIR / "config" / ".env" if os.getenv("ELCQA_ROOT") else LEGACY_ENV_PATH)).resolve()
DEFAULT_DOCS_DIR = Path(os.getenv("ELCQA_DOCS_DIR") or (PROJECT_DIR / "docs" if os.getenv("ELCQA_ROOT") else LEGACY_DOCS_DIR)).resolve()


class Settings(BaseModel):
    docs_dir: Path = DEFAULT_DOCS_DIR
    chroma_dir: Path = PROJECT_DIR / "data" / "chroma"
    logs_dir: Path = PROJECT_DIR / "logs"
    collection_name: str = "enterprise_line_rules"
    api_key: str | None = None
    base_url: str = "https://api.siliconflow.cn/v1"
    openai_compatible: bool = True
    chat_model: str = "Qwen/Qwen3.5-9B"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_openai_compatible: bool = True
    embedding_model: str = "BAAI/bge-m3"
    request_timeout: float = 45.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ENV_PATH)
    return Settings(
        api_key=effective_api_key(),
        base_url=effective_base_url(),
        openai_compatible=effective_openai_compatible(),
        chat_model=effective_chat_model(),
        embedding_api_key=effective_embedding_api_key(),
        embedding_base_url=effective_embedding_base_url(),
        embedding_openai_compatible=effective_embedding_openai_compatible(),
        embedding_model=effective_embedding_model(),
    )


def read_app_config() -> dict:
    return read_public_app_config()


def write_common_questions(questions: list[str]) -> list[str]:
    return update_common_questions(questions)


def write_system_name(name: str) -> str:
    return update_system_name(name)


def write_knowledge_base_name(name: str) -> str:
    return update_knowledge_base_name(_normalize_knowledge_base_name(name))


def _normalize_knowledge_base_name(name: str) -> str:
    value = "".join(ch for ch in name.strip().replace("《", "").replace("》", "") if "\u4e00" <= ch <= "\u9fff")
    return value[:16] or DEFAULT_KNOWLEDGE_BASE_NAME
