from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from services.rag_api.paths import resolve_project_root

PROJECT_ROOT = resolve_project_root(Path(__file__).resolve().parents[2])
SETTINGS_PATH = PROJECT_ROOT / "data" / "rag_settings.json"
_SETTINGS_OVERRIDE: ContextVar["RagSettings | None"] = ContextVar("rag_settings_override", default=None)
RerankProvider = Literal["api", "local_onnx"]


class RagSettings(BaseModel):
    multi_vector_enabled: bool = False
    hybrid_bm25_enabled: bool = False
    query_expansion_enabled: bool = False
    rerank_enabled: bool = False
    context_rewrite_enabled: bool = False
    dynamic_top_k_enabled: bool = False
    parent_context_enabled: bool = False
    dedup_enabled: bool = False
    rag_param_tuning_enabled: bool = False
    chunk_size: int = Field(default=600, ge=200, le=1200)
    chunk_overlap: int = Field(default=100, ge=0, le=300)
    top_k: int = Field(default=2, ge=1, le=10)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    vector_candidate_k: int = Field(default=8, ge=2, le=50)
    max_context_tokens: int = Field(default=6000, ge=100, le=50000)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_provider: RerankProvider = "api"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


def load_rag_settings() -> RagSettings:
    override = _SETTINGS_OVERRIDE.get()
    if override is not None:
        return override.model_copy(deep=True)
    if not SETTINGS_PATH.exists():
        return RagSettings()
    try:
        settings = RagSettings.model_validate_json(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return RagSettings()
    from services.rag_api.evaluation.approval import effective_runtime_settings

    return effective_runtime_settings(settings)


def save_rag_settings(settings: RagSettings) -> RagSettings:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return settings


def reset_rag_settings() -> RagSettings:
    return save_rag_settings(RagSettings())


def get_retrieval_top_k(settings: RagSettings | None = None) -> int:
    active = settings or load_rag_settings()
    return active.top_k


def resolve_retrieval_top_k(query: str, settings: RagSettings) -> dict[str, int | bool | str]:
    base_top_k = max(1, min(10, int(settings.top_k)))
    if not settings.dynamic_top_k_enabled:
        return {
            "enabled": False,
            "base_top_k": base_top_k,
            "effective_top_k": base_top_k,
            "increase": 0,
            "reason": "disabled",
        }

    normalized = re.sub(r"\s+", "", query or "")
    reasons: list[str] = []
    if len(normalized) >= 36:
        reasons.append("long_query")
    clause_signals = re.findall(r"以及|同时|分别|并且|并|和|、|，|；|;", normalized)
    if len(clause_signals) >= 2:
        reasons.append("multiple_clauses")
    requirement_signals = re.findall(r"哪些|如何|为什么|是否|多少|步骤|条件|影响|区别|关系|时限|材料|审核|资费", normalized)
    if len(requirement_signals) >= 3:
        reasons.append("multiple_requirements")

    requested_increase = min(2, len(reasons))
    effective_top_k = min(10, base_top_k + requested_increase)
    increase = effective_top_k - base_top_k
    return {
        "enabled": True,
        "base_top_k": base_top_k,
        "effective_top_k": effective_top_k,
        "increase": increase,
        "reason": "+".join(reasons) if reasons else "simple_query",
    }


@contextmanager
def override_rag_settings(settings: RagSettings):
    token = _SETTINGS_OVERRIDE.set(settings)
    try:
        yield
    finally:
        _SETTINGS_OVERRIDE.reset(token)
