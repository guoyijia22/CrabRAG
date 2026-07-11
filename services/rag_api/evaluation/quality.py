from __future__ import annotations

import math

_HIGHER_IS_BETTER = (
    "recall_at_5",
    "mrr_at_10",
    "citation_precision",
    "citation_coverage",
)
_INACTIVE_STATUSES = {"deleted", "inactive", "retired", "tombstoned"}
_EPSILON = 1e-12


def calculate_quality_metrics(cases: list[dict]) -> dict:
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    expected_evidence_cases = 0
    covered_cases = 0
    no_evidence_answers = 0
    relevant_citations = 0
    citation_count = 0
    acl_leaks = 0
    invalid_content_leaks = 0
    latencies: list[int] = []
    model_call_count = 0

    for case in cases:
        references = list(case.get("references") or [])
        expected = case.get("expected") or case
        relevant_ids = _expected_reference_ids(expected)
        expect_references = bool(expected.get("expect_references", True))
        relevance = [_reference_is_relevant(reference, relevant_ids) for reference in references]

        if relevant_ids:
            top_five_ids = {_reference_id(reference) for reference in references[:5]}
            recall_values.append(len(relevant_ids & top_five_ids) / len(relevant_ids))
            reciprocal_ranks.append(_reciprocal_rank(relevance[:10]))
        if expect_references:
            expected_evidence_cases += 1
            if any(relevance):
                covered_cases += 1
            if str(case.get("answer") or "").strip() and not references:
                no_evidence_answers += 1

        citation_count += len(references)
        relevant_citations += sum(1 for item in relevance if item)
        acl_leaks += sum(1 for reference in references if _is_acl_leak(reference))
        invalid_content_leaks += sum(1 for reference in references if _is_invalid_content(reference))
        latencies.append(max(0, int(case.get("latency_ms", 0) or 0)))
        model_call_count += max(0, int(case.get("model_call_count", 0) or 0))

    return {
        "recall_at_5": _average(recall_values),
        "mrr_at_10": _average(reciprocal_ranks),
        "citation_precision": _ratio(relevant_citations, citation_count),
        "citation_coverage": _ratio(covered_cases, expected_evidence_cases),
        "no_evidence_answer_rate": _ratio(no_evidence_answers, expected_evidence_cases),
        "acl_leakage_rate": _ratio(acl_leaks, citation_count),
        "invalid_content_leakage_rate": _ratio(invalid_content_leaks, citation_count),
        "p95_latency_ms": _percentile_95(latencies),
        "model_call_count": model_call_count,
    }


def evaluate_quality_gate(candidate: dict, baseline: dict, *, gate_eligible: bool) -> dict:
    if not gate_eligible:
        return {
            "eligible": False,
            "passed": False,
            "checks": {},
            "reasons": ["fixed_dataset_required"],
        }

    baseline_recall = float(baseline.get("recall_at_5", 0) or 0)
    baseline_latency = float(baseline.get("p95_latency_ms", 0) or 0)
    checks = {
        "acl_leakage_zero": float(candidate.get("acl_leakage_rate", 0) or 0) == 0,
        "invalid_content_leakage_zero": float(candidate.get("invalid_content_leakage_rate", 0) or 0) == 0,
        "recall_regression_within_limit": float(candidate.get("recall_at_5", 0) or 0) + _EPSILON >= baseline_recall - 0.02,
        "p95_latency_within_limit": float(candidate.get("p95_latency_ms", 0) or 0) <= baseline_latency * 1.2 + _EPSILON,
        "primary_quality_improved": _primary_quality_improved(candidate, baseline),
    }
    return {
        "eligible": True,
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
    }


def attach_quality_gates(profiles: list[dict], *, gate_eligible: bool) -> None:
    if not profiles:
        return
    baseline = profiles[0].get("summary", {})
    baseline["quality_gate"] = {
        "eligible": gate_eligible,
        "passed": None,
        "checks": {},
        "reasons": ["baseline_reference"],
    }
    for profile in profiles[1:]:
        summary = profile.get("summary", {})
        summary["quality_gate"] = evaluate_quality_gate(summary, baseline, gate_eligible=gate_eligible)


def _expected_reference_ids(expected: dict) -> set[str]:
    values = expected.get("expected_document_ids") or expected.get("expected_source_files") or []
    if isinstance(values, str):
        values = [values]
    return {str(value).strip() for value in values if str(value).strip()}


def _reference_id(reference: dict) -> str:
    metadata = reference.get("metadata") or {}
    return str(
        reference.get("document_id")
        or metadata.get("document_id")
        or reference.get("source_file")
        or metadata.get("source_file")
        or ""
    ).strip()


def _reference_is_relevant(reference: dict, relevant_ids: set[str]) -> bool:
    if isinstance(reference.get("relevant"), bool):
        return reference["relevant"]
    return bool(relevant_ids and _reference_id(reference) in relevant_ids)


def _reciprocal_rank(relevance: list[bool]) -> float:
    for index, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1 / index
    return 0.0


def _is_acl_leak(reference: dict) -> bool:
    metadata = reference.get("metadata") or {}
    return any(reference.get(field, metadata.get(field)) is False for field in ("acl_allowed", "permission_allowed", "authorized"))


def _is_invalid_content(reference: dict) -> bool:
    metadata = reference.get("metadata") or {}
    if any(reference.get(field, metadata.get(field)) is False for field in ("is_active", "is_valid")):
        return True
    status = str(reference.get("status") or metadata.get("status") or "").lower()
    return status in _INACTIVE_STATUSES


def _primary_quality_improved(candidate: dict, baseline: dict) -> bool:
    if any(float(candidate.get(key, 0) or 0) > float(baseline.get(key, 0) or 0) + _EPSILON for key in _HIGHER_IS_BETTER):
        return True
    return float(candidate.get("no_evidence_answer_rate", 0) or 0) + _EPSILON < float(
        baseline.get("no_evidence_answer_rate", 0) or 0
    )


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0


def _percentile_95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
