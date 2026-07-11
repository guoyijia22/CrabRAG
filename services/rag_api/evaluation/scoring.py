from __future__ import annotations

from statistics import mean

from services.rag_api.app_settings import DEFAULT_NO_MATCH_RESPONSE, load_app_settings
from services.rag_api.evaluation.quality import calculate_quality_metrics

QUALITY_WEIGHTS = {
    "success_rate": 0.30,
    "fallback_correct_rate": 0.15,
    "source_hit_rate": 0.15,
    "graph_path_coverage": 0.15,
    "retrieval_mode_match_rate": 0.10,
    "avg_reference_score": 0.10,
    "intent_match_rate": 0.05,
}


def score_case(case: dict) -> dict:
    references = case.get("references", [])
    relation_paths = case.get("relation_paths", [])
    scores = [float(item.get("score", 0) or 0) for item in references]
    answer = case.get("answer", "") or ""
    error = case.get("error")
    configured_no_match = load_app_settings().no_match_response
    has_no_match_answer = configured_no_match in answer or DEFAULT_NO_MATCH_RESPONSE in answer
    fallback = has_no_match_answer or not references
    expected = case.get("expected", {}) or {}
    expected_intent = expected.get("expected_intent", "")
    expected_modes = expected.get("expected_retrieval_modes", []) or []
    expected_sources = expected.get("expected_source_files", []) or []
    expect_references = bool(expected.get("expect_references", True))
    expect_relation_paths = bool(expected.get("expect_relation_paths", False))
    trace = case.get("trace", [])
    intent_match = not expected_intent or case.get("intent") == expected_intent
    retrieval_mode_match = not expected_modes or case.get("retrieval_mode") in expected_modes
    source_hit = _source_hit(references, expected_sources)
    graph_path_hit = bool(relation_paths) if expect_relation_paths else True
    fallback_correct = has_no_match_answer and not references if not expect_references else not fallback
    success = fallback_correct if not expect_references else (error is None and bool(references) and not fallback)
    return {
        "success": success,
        "has_references": bool(references),
        "reference_count": len(references),
        "avg_reference_score": round(mean(scores), 4) if scores else 0,
        "source_count": len({item.get("source_file", "") for item in references if item.get("source_file")}),
        "latency_ms": int(case.get("latency_ms", 0) or 0),
        "fallback": fallback,
        "answer_length": len(answer),
        "trace_activations": _trace_activations(trace),
        "trace_fallbacks": _trace_fallbacks(trace),
        "intent_match": intent_match,
        "retrieval_mode_match": retrieval_mode_match,
        "source_hit": source_hit,
        "graph_path_hit": graph_path_hit,
        "context_rewrite_triggered": _context_rewrite_triggered(trace),
        "fallback_correct": fallback_correct,
    }


def score_profile(profile_id: str, cases: list[dict]) -> dict:
    metrics = [case.get("metrics") or score_case(case) for case in cases]
    count = len(metrics)
    if count == 0:
        return {
            "profile_id": profile_id,
            "question_count": 0,
            "success_rate": 0,
            "fallback_rate": 0,
            "error_rate": 0,
            "avg_latency_ms": 0,
            "avg_reference_count": 0,
            "avg_reference_score": 0,
            "avg_answer_length": 0,
            "trace_activations": {},
            "trace_fallbacks": {},
            "intent_match_rate": 0,
            "retrieval_mode_match_rate": 0,
            "source_hit_rate": 0,
            "graph_path_coverage": 0,
            "fallback_correct_rate": 0,
            "quality_score": 0,
            **calculate_quality_metrics([]),
        }
    trace_activations: dict[str, int] = {}
    trace_fallbacks: dict[str, int] = {}
    for item in metrics:
        for node, value in item.get("trace_activations", {}).items():
            trace_activations[node] = trace_activations.get(node, 0) + value
        for node, value in item.get("trace_fallbacks", {}).items():
            trace_fallbacks[node] = trace_fallbacks.get(node, 0) + value
    summary = {
        "profile_id": profile_id,
        "question_count": count,
        "success_rate": round(sum(1 for item in metrics if item["success"]) / count, 4),
        "fallback_rate": round(sum(1 for item in metrics if item["fallback"]) / count, 4),
        "error_rate": round(sum(1 for case in cases if case.get("error")) / count, 4),
        "avg_latency_ms": int(mean(item["latency_ms"] for item in metrics)),
        "avg_reference_count": round(mean(item["reference_count"] for item in metrics), 4),
        "avg_reference_score": _mean_nonzero([item["avg_reference_score"] for item in metrics]),
        "avg_answer_length": int(mean(item["answer_length"] for item in metrics)),
        "trace_activations": trace_activations,
        "trace_fallbacks": trace_fallbacks,
        "intent_match_rate": round(sum(1 for item in metrics if item.get("intent_match")) / count, 4),
        "retrieval_mode_match_rate": round(sum(1 for item in metrics if item.get("retrieval_mode_match")) / count, 4),
        "source_hit_rate": round(sum(1 for item in metrics if item.get("source_hit")) / count, 4),
        "graph_path_coverage": round(sum(1 for item in metrics if item.get("graph_path_hit")) / count, 4),
        "fallback_correct_rate": round(sum(1 for item in metrics if item.get("fallback_correct")) / count, 4),
        **calculate_quality_metrics(cases),
    }
    summary["quality_score"] = _quality_score(summary)
    return summary


def attach_baseline_deltas(profiles: list[dict]) -> None:
    if not profiles:
        return
    baseline = profiles[0].get("summary", {})
    best_profile = _best_profile(profiles)
    for profile in profiles:
        summary = profile.get("summary", {})
        summary["quality_score"] = _quality_score(summary)
        summary["delta"] = {
            "success_rate": round(summary.get("success_rate", 0) - baseline.get("success_rate", 0), 4),
            "fallback_rate": round(summary.get("fallback_rate", 0) - baseline.get("fallback_rate", 0), 4),
            "avg_latency_ms": int(summary.get("avg_latency_ms", 0) - baseline.get("avg_latency_ms", 0)),
            "avg_reference_score": round(summary.get("avg_reference_score", 0) - baseline.get("avg_reference_score", 0), 4),
            "source_hit_rate": round(summary.get("source_hit_rate", 0) - baseline.get("source_hit_rate", 0), 4),
            "graph_path_coverage": round(summary.get("graph_path_coverage", 0) - baseline.get("graph_path_coverage", 0), 4),
        }
        summary["recommendation"] = "效果最好" if profile is best_profile else _recommendation(summary)
        if profile is best_profile:
            summary["best_reason"] = _best_reason(profile)


def build_overall_summary(profiles: list[dict], question_generation: dict | None = None) -> dict:
    best = _best_profile(profiles)
    return {
        "best_profile": best.get("id", ""),
        "best_profile_name": best.get("name", ""),
        "best_reason": best.get("summary", {}).get("best_reason") or _best_reason(best),
        "profile_count": len(profiles),
        "diagnostics": build_diagnostic_summary(profiles, question_generation or {}),
    }


def decorate_evaluation_run(payload: dict) -> dict:
    profiles = payload.get("profiles", [])
    if profiles:
        attach_baseline_deltas(profiles)
        payload["summary"] = build_overall_summary(profiles, payload.get("question_generation", {}))
    return payload


def build_diagnostic_summary(profiles: list[dict], question_generation: dict) -> dict:
    best = _best_profile(profiles) if profiles else {}
    cases = best.get("cases", []) or []
    return {
        "question_generation_mode": question_generation.get("mode", ""),
        "question_generation_fallback": question_generation.get("mode") == "fallback",
        "best_profile": best.get("id", ""),
        "intent_mismatches": _case_summaries(cases, lambda case: not case.get("metrics", {}).get("intent_match", True)),
        "fallback_failures": _case_summaries(cases, lambda case: not case.get("metrics", {}).get("fallback_correct", True)),
        "graph_path_misses": _case_summaries(cases, lambda case: not case.get("metrics", {}).get("graph_path_hit", True)),
    }


def _trace_activations(trace: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in trace:
        node = item.get("node", "")
        output = item.get("output", {}) or {}
        if output.get("enabled") is True or node in {"agent_tool_choice", "retrieve", "generate_answer", "business_scope_check"}:
            result[node] = result.get(node, 0) + 1
    return result


def _trace_fallbacks(trace: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in trace:
        node = item.get("node", "")
        output = item.get("output", {}) or {}
        if output.get("fallback") is True:
            result[node] = result.get(node, 0) + 1
    return result


def _source_hit(references: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    actual_sources = [item.get("source_file", "") for item in references if item.get("source_file")]
    for actual in actual_sources:
        for expected in expected_sources:
            if expected and (expected == actual or expected in actual or actual in expected):
                return True
    return False


def _context_rewrite_triggered(trace: list[dict]) -> bool:
    for item in trace:
        if item.get("node") != "context_rewrite":
            continue
        output = item.get("output", {}) or {}
        if output.get("enabled") is True and not output.get("fallback"):
            return True
    return False


def _recommendation(summary: dict) -> str:
    delta = summary.get("delta", {})
    if delta.get("success_rate", 0) > 0 or delta.get("avg_reference_score", 0) >= 0.05:
        return "建议适用"
    if delta.get("success_rate", 0) < 0 or delta.get("fallback_rate", 0) > 0.1:
        return "不建议适用/有风险"
    return "效果接近"


def _quality_score(summary: dict) -> float:
    score = 0.0
    for key, weight in QUALITY_WEIGHTS.items():
        score += min(1.0, max(0.0, float(summary.get(key, 0) or 0))) * weight
    return round(score, 4)


def _best_profile(profiles: list[dict]) -> dict:
    if not profiles:
        return {}
    return max(
        profiles,
        key=lambda item: (
            _quality_score(item.get("summary", {})),
            item.get("summary", {}).get("success_rate", 0),
            item.get("summary", {}).get("fallback_correct_rate", 0),
            item.get("summary", {}).get("source_hit_rate", 0),
            item.get("summary", {}).get("graph_path_coverage", 0),
            item.get("summary", {}).get("avg_reference_score", 0),
            -int(item.get("summary", {}).get("avg_latency_ms", 0) or 0),
        ),
    )


def _best_reason(profile: dict) -> str:
    summary = profile.get("summary", {})
    return (
        f"质量优先综合分 {summary.get('quality_score', _quality_score(summary))}，"
        f"成功率 {summary.get('success_rate', 0):.2f}，"
        f"来源命中率 {summary.get('source_hit_rate', 0):.2f}，"
        f"图谱路径覆盖率 {summary.get('graph_path_coverage', 0):.2f}。"
    )


def _case_summaries(cases: list[dict], predicate) -> list[dict]:
    result = []
    for case in cases:
        if not predicate(case):
            continue
        expected = case.get("expected", {}) or {}
        result.append(
            {
                "question_id": case.get("question_id", ""),
                "question": case.get("question", ""),
                "expected_intent": expected.get("expected_intent", ""),
                "actual_intent": case.get("intent", ""),
                "retrieval_mode": case.get("retrieval_mode", ""),
            }
        )
    return result


def _mean_nonzero(values: list[float]) -> float:
    usable = [value for value in values if value > 0]
    return round(mean(usable), 4) if usable else 0
