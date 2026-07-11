from __future__ import annotations

from services.rag_api.evaluation.profiles import build_evaluation_profiles
from services.rag_api.rag_settings import RagSettings


def test_evaluation_profiles_follow_new_retrieval_chain_terms():
    profiles = build_evaluation_profiles(RagSettings())
    ids = [profile["id"] for profile in profiles]
    names = [profile["name"] for profile in profiles]
    descriptions = " ".join(profile["description"] for profile in profiles)
    enabled_switches = {switch for profile in profiles for switch in profile["enabled_switches"]}

    assert ids == [
        "baseline_new_retrieval",
        "query_expansion_enabled",
        "rerank_enabled",
        "context_rewrite_enabled",
        "dynamic_top_k_enabled",
        "parent_context_enabled",
        "multi_vector_enabled",
        "query_expansion_rerank",
        "context_rewrite_rerank",
        "multi_vector_rerank",
        "all_enhancements",
    ]
    assert names[0] == "标准新链路基线"
    assert "hybrid_bm25_enabled" not in enabled_switches
    assert "rag_param_tuning_enabled" not in enabled_switches
    assert "dynamic_top_k_enabled" in enabled_switches
    assert "parent_context_enabled" in enabled_switches
    assert "混合检索" not in descriptions
    assert "BM25" not in descriptions
    assert "6 项优化" not in descriptions
