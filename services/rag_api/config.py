from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from services.rag_api.app_settings import (
    DEFAULT_KNOWLEDGE_BASE_NAME,
    DEFAULT_SYSTEM_NAME,
    default_knowledge_base_dirs,
    effective_knowledge_base_dirs,
    read_public_app_config,
    update_common_questions,
    update_knowledge_base_name,
    update_system_name,
)
from services.rag_api.paths import resolve_env_file, resolve_project_root
from services.rag_api.model_api_settings import (
    effective_api_key,
    effective_base_url,
    effective_chat_model,
    effective_embedding_api_key,
    effective_embedding_base_url,
    effective_embedding_model,
    effective_embedding_onnx_model_file,
    effective_embedding_openai_compatible,
    effective_embedding_provider,
    effective_openai_compatible,
    effective_rerank_api_key,
    effective_rerank_base_url,
    effective_rerank_onnx_model_file,
    effective_use_local_models,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = resolve_project_root(ROOT_DIR)
LEGACY_ENV_PATH = Path(r"D:\cd\.env")
LEGACY_DOCS_DIR = Path(r"D:\cd\docs")
ENV_PATH = resolve_env_file(PROJECT_DIR, LEGACY_ENV_PATH)
DEFAULT_DOCS_DIRS = [Path(item).resolve() for item in default_knowledge_base_dirs()]
DEFAULT_DOCS_DIR = DEFAULT_DOCS_DIRS[0] if DEFAULT_DOCS_DIRS else LEGACY_DOCS_DIR.resolve()
LOCAL_QWEN_LLM_MODEL_DIR = PROJECT_DIR / "runtime" / "models" / "Qwen3___5-0___8B-ONNX"
LOCAL_QWEN_EMBEDDING_MODEL_DIR = PROJECT_DIR / "runtime" / "models" / "Qwen3-Embedding-0___6B-ONNX"
LOCAL_QWEN_RERANK_MODEL_DIR = PROJECT_DIR / "runtime" / "models" / "Qwen3-Reranker-0___6B-ONNX"


class Settings(BaseModel):
    docs_dir: Path = DEFAULT_DOCS_DIR
    docs_dirs: list[Path] = DEFAULT_DOCS_DIRS
    chroma_dir: Path = PROJECT_DIR / "data" / "chroma"
    logs_dir: Path = PROJECT_DIR / "logs"
    collection_name: str = "enterprise_line_rules"
    use_local_models: bool = False
    api_key: str | None = None
    base_url: str = "https://api.siliconflow.cn/v1"
    openai_compatible: bool = True
    chat_model: str = "Qwen/Qwen3.5-9B"
    embedding_provider: str = "api"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_openai_compatible: bool = True
    embedding_model: str = "BAAI/bge-m3"
    embedding_onnx_model_file: str = "model.onnx"
    rerank_api_key: str | None = None
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_onnx_model_file: str = "model.onnx"
    local_llm_model_dir: Path = LOCAL_QWEN_LLM_MODEL_DIR
    local_embedding_model_dir: Path = LOCAL_QWEN_EMBEDDING_MODEL_DIR
    local_rerank_model_dir: Path = LOCAL_QWEN_RERANK_MODEL_DIR
    request_timeout: float = 45.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ENV_PATH)
    docs_dirs = effective_knowledge_base_dirs()
    use_local_models = effective_use_local_models()
    return Settings(
        docs_dir=docs_dirs[0],
        docs_dirs=docs_dirs,
        use_local_models=use_local_models,
        api_key=effective_api_key(),
        base_url=effective_base_url(),
        openai_compatible=effective_openai_compatible(),
        chat_model=effective_chat_model(),
        embedding_provider=effective_embedding_provider(),
        embedding_api_key=effective_embedding_api_key(),
        embedding_base_url=effective_embedding_base_url(),
        embedding_openai_compatible=effective_embedding_openai_compatible(),
        embedding_model=effective_embedding_model(),
        embedding_onnx_model_file=effective_embedding_onnx_model_file(),
        rerank_api_key=effective_rerank_api_key(),
        rerank_base_url=effective_rerank_base_url(),
        rerank_onnx_model_file=effective_rerank_onnx_model_file(),
        local_embedding_model_dir=LOCAL_QWEN_EMBEDDING_MODEL_DIR,
        local_rerank_model_dir=LOCAL_QWEN_RERANK_MODEL_DIR,
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
