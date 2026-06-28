from __future__ import annotations

from services.rag_api.rag_settings import RagSettings

SWITCH_KEYS = [
    "query_expansion_enabled",
    "rerank_enabled",
    "context_rewrite_enabled",
    "multi_vector_enabled",
]

DEPRECATED_SWITCH_KEYS = [
    "hybrid_bm25_enabled",
    "rag_param_tuning_enabled",
]

PROFILE_SWITCHES = [
    ("query_expansion_enabled", "查询扩展", "生成等价查询，增强文本向量召回覆盖"),
    ("rerank_enabled", "统一重排", "对融合后的候选片段统一精排"),
    ("context_rewrite_enabled", "多轮追问重写", "多轮聊天中将追问改写为独立查询"),
    ("multi_vector_enabled", "多粒度文本索引", "重建知识库时生成 document / paragraph / sentence 多粒度索引"),
]

COMBINATION_PROFILES = [
    (
        "query_expansion_rerank",
        "查询扩展 + 统一重排",
        "扩展文本召回后，对融合候选统一精排",
        ["query_expansion_enabled", "rerank_enabled"],
    ),
    (
        "context_rewrite_rerank",
        "多轮追问重写 + 统一重排",
        "面向多轮追问，先改写查询再统一精排候选",
        ["context_rewrite_enabled", "rerank_enabled"],
    ),
    (
        "multi_vector_rerank",
        "多粒度文本索引 + 统一重排",
        "使用多粒度文本索引扩大候选，再统一精排",
        ["multi_vector_enabled", "rerank_enabled"],
    ),
    (
        "all_enhancements",
        "全增强配置",
        "启用查询扩展、统一重排、多轮追问重写和多粒度文本索引",
        SWITCH_KEYS,
    ),
]

EVALUATION_COLLECTIONS = {
    "multi_vector_enabled": "crabrag_eval_multi_vector",
}


def build_evaluation_profiles(current_settings: RagSettings) -> list[dict]:
    profiles = [
        _profile(
            profile_id="baseline_new_retrieval",
            name="标准新链路基线",
            description="使用默认 vector / graph / keyword 召回、动态图谱和 token 预算裁剪，关闭可选增强",
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
    updates.update({switch: False for switch in DEPRECATED_SWITCH_KEYS})
    return current_settings.model_copy(update=updates)


def _base_settings(current_settings: RagSettings) -> RagSettings:
    updates = {switch: False for switch in SWITCH_KEYS}
    updates.update({switch: False for switch in DEPRECATED_SWITCH_KEYS})
    return current_settings.model_copy(update=updates)


def _collection_name_for(enabled_switches: list[str]) -> str | None:
    if "multi_vector_enabled" in enabled_switches:
        return EVALUATION_COLLECTIONS["multi_vector_enabled"]
    return None


def _normalize_switches(enabled_switches: list[str]) -> list[str]:
    selected = set(enabled_switches)
    return [switch for switch in SWITCH_KEYS if switch in selected]
