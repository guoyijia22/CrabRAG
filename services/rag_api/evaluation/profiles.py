from __future__ import annotations

from services.rag_api.rag_settings import RagSettings

SWITCH_KEYS = [
    "multi_vector_enabled",
    "hybrid_bm25_enabled",
    "query_expansion_enabled",
    "rerank_enabled",
    "context_rewrite_enabled",
    "rag_param_tuning_enabled",
]

PROFILE_SWITCHES = [
    ("multi_vector_enabled", "多向量检索", "生成 document / paragraph / sentence 多粒度索引"),
    ("hybrid_bm25_enabled", "混合检索", "BM25 关键词召回与向量召回 RRF 融合"),
    ("query_expansion_enabled", "查询扩展", "LLM 生成扩展查询提升召回覆盖"),
    ("rerank_enabled", "重排模型", "SiliconFlow Rerank API 对候选片段精排"),
    ("context_rewrite_enabled", "上下文意图重构", "多轮追问改写为独立查询"),
    ("rag_param_tuning_enabled", "核心参数调优", "启用 chunk、阈值、候选数和权重参数"),
]

COMBINATION_PROFILES = [
    (
        "hybrid_rerank",
        "混合检索 + 重排",
        "用 BM25 与向量融合扩大候选，再用重排模型提升片段精度",
        ["hybrid_bm25_enabled", "rerank_enabled"],
    ),
    (
        "query_hybrid",
        "查询扩展 + 混合检索",
        "扩展用户问题后同时走关键词和向量召回，偏召回覆盖",
        ["query_expansion_enabled", "hybrid_bm25_enabled"],
    ),
    (
        "query_hybrid_rerank",
        "查询扩展 + 混合检索 + 重排",
        "先扩展与融合召回，再对候选片段精排",
        ["query_expansion_enabled", "hybrid_bm25_enabled", "rerank_enabled"],
    ),
    (
        "context_hybrid",
        "上下文重构 + 混合检索",
        "将追问改写为独立问题，再用混合检索增强召回",
        ["context_rewrite_enabled", "hybrid_bm25_enabled"],
    ),
    (
        "context_query_hybrid_rerank",
        "上下文重构 + 查询扩展 + 混合检索 + 重排",
        "面向多轮追问的强召回与精排组合",
        ["context_rewrite_enabled", "query_expansion_enabled", "hybrid_bm25_enabled", "rerank_enabled"],
    ),
    (
        "multi_vector_hybrid",
        "多向量 + 混合检索",
        "多粒度索引叠加关键词召回，观察规范片段覆盖变化",
        ["multi_vector_enabled", "hybrid_bm25_enabled"],
    ),
    (
        "multi_vector_hybrid_rerank",
        "多向量 + 混合检索 + 重排",
        "多粒度候选召回后精排，偏综合质量",
        ["multi_vector_enabled", "hybrid_bm25_enabled", "rerank_enabled"],
    ),
    (
        "param_hybrid_rerank",
        "参数调优 + 混合检索 + 重排",
        "验证候选数量、阈值和权重调优后的精排效果",
        ["rag_param_tuning_enabled", "hybrid_bm25_enabled", "rerank_enabled"],
    ),
    (
        "multi_vector_param_hybrid_rerank",
        "多向量 + 参数调优 + 混合检索 + 重排",
        "同时调整索引粒度、参数和精排，不启用额外查询改写",
        ["multi_vector_enabled", "rag_param_tuning_enabled", "hybrid_bm25_enabled", "rerank_enabled"],
    ),
    (
        "all_enabled",
        "全配置开启",
        "6 项 RAG 优化全部启用，用于观察最高增强配置的综合效果",
        SWITCH_KEYS,
    ),
]

EVALUATION_COLLECTIONS = {
    "multi_vector_enabled": "enterprise_line_rules_eval_multi_vector",
    "rag_param_tuning_enabled": "enterprise_line_rules_eval_param_tuning",
    "multi_vector_param_tuning": "enterprise_line_rules_eval_multi_vector_param_tuning",
}


def build_evaluation_profiles(current_settings: RagSettings) -> list[dict]:
    profiles = [
        _profile(
            profile_id="baseline_all_off",
            name="全关闭基线",
            description="关闭 6 项优化的基线配置",
            settings=_base_settings(current_settings),
            profile_type="baseline",
            enabled_switches=[],
        )
    ]
    for switch, name, description in PROFILE_SWITCHES:
        settings = _settings_with_switches(current_settings, [switch])
        profiles.append(
            _profile(
                profile_id=switch,
                name=name,
                description=description,
                settings=settings,
                profile_type="single",
                enabled_switches=[switch],
            )
        )
    for profile_id, name, description, switches in COMBINATION_PROFILES:
        settings = _settings_with_switches(current_settings, switches)
        profiles.append(
            _profile(
                profile_id=profile_id,
                name=name,
                description=description,
                settings=settings,
                profile_type="combination",
                enabled_switches=switches,
            )
        )
    return profiles


def serialize_profile(profile: dict, include_settings: bool = False) -> dict:
    payload = {key: value for key, value in profile.items() if key != "settings"}
    if include_settings:
        payload["settings"] = profile["settings"].model_dump()
    return payload


def evaluation_collection_names(profiles: list[dict]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for profile in profiles:
        collection_name = profile.get("collection_name")
        if collection_name and collection_name not in seen:
            seen.add(collection_name)
            names.append(collection_name)
    return names


def _profile(
    *,
    profile_id: str,
    name: str,
    description: str,
    settings: RagSettings,
    profile_type: str,
    enabled_switches: list[str],
) -> dict:
    normalized_switches = _normalize_switches(enabled_switches)
    return {
        "id": profile_id,
        "name": name,
        "description": description,
        "profile_type": profile_type,
        "enabled_switches": normalized_switches,
        "settings": settings,
        "collection_name": _collection_name_for(normalized_switches),
    }


def _settings_with_switches(current_settings: RagSettings, switches: list[str]) -> RagSettings:
    updates = {switch: switch in switches for switch in SWITCH_KEYS}
    return current_settings.model_copy(update=updates)


def _base_settings(current_settings: RagSettings) -> RagSettings:
    return current_settings.model_copy(update={switch: False for switch in SWITCH_KEYS})


def _collection_name_for(enabled_switches: list[str]) -> str | None:
    has_multi_vector = "multi_vector_enabled" in enabled_switches
    has_param_tuning = "rag_param_tuning_enabled" in enabled_switches
    if has_multi_vector and has_param_tuning:
        return EVALUATION_COLLECTIONS["multi_vector_param_tuning"]
    if has_multi_vector:
        return EVALUATION_COLLECTIONS["multi_vector_enabled"]
    if has_param_tuning:
        return EVALUATION_COLLECTIONS["rag_param_tuning_enabled"]
    return None


def _normalize_switches(enabled_switches: list[str]) -> list[str]:
    selected = set(enabled_switches)
    return [switch for switch in SWITCH_KEYS if switch in selected]
