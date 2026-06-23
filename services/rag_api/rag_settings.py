from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(os.getenv("ELCQA_ROOT") or Path(__file__).resolve().parents[2]).resolve()
SETTINGS_PATH = PROJECT_ROOT / "data" / "rag_settings.json"
_SETTINGS_OVERRIDE: ContextVar["RagSettings | None"] = ContextVar("rag_settings_override", default=None)


class RagSettings(BaseModel):
    multi_vector_enabled: bool = False
    hybrid_bm25_enabled: bool = False
    query_expansion_enabled: bool = False
    rerank_enabled: bool = False
    context_rewrite_enabled: bool = False
    rag_param_tuning_enabled: bool = False
    chunk_size: int = Field(default=600, ge=200, le=1200)
    chunk_overlap: int = Field(default=100, ge=0, le=300)
    top_k: int = Field(default=2, ge=1, le=10)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    vector_candidate_k: int = Field(default=8, ge=2, le=50)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


def load_rag_settings() -> RagSettings:
    override = _SETTINGS_OVERRIDE.get()
    if override is not None:
        return override.model_copy(deep=True)
    if not SETTINGS_PATH.exists():
        return RagSettings()
    try:
        return RagSettings.model_validate_json(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return RagSettings()


def save_rag_settings(settings: RagSettings) -> RagSettings:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return settings


def reset_rag_settings() -> RagSettings:
    return save_rag_settings(RagSettings())


def get_retrieval_top_k(settings: RagSettings | None = None) -> int:
    active = settings or load_rag_settings()
    return active.top_k


@contextmanager
def override_rag_settings(settings: RagSettings):
    token = _SETTINGS_OVERRIDE.set(settings)
    try:
        yield
    finally:
        _SETTINGS_OVERRIDE.reset(token)
