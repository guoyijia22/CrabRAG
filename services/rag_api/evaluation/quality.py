from __future__ import annotations

import math

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
            top_five_ids = set().union(*(_reference_ids(reference) for reference in references[:5])) if references else set()
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
        acl_leaks += sum(1 for reference in references if _is_acl_leak(reference, expected))
        invalid_content_leaks += sum(1 for reference in references if _is_invalid_content(reference, expected))
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
        "citation_precision_non_regression": float(candidate.get("citation_precision", 0) or 0) + _EPSILON
        >= float(baseline.get("citation_precision", 0) or 0),
        "citation_coverage_non_regression": float(candidate.get("citation_coverage", 0) or 0) + _EPSILON
        >= float(baseline.get("citation_coverage", 0) or 0),
        "no_evidence_answer_non_regression": float(candidate.get("no_evidence_answer_rate", 0) or 0)
        <= float(baseline.get("no_evidence_answer_rate", 0) or 0) + _EPSILON,
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
    field, prefix = next(
        (
            (field_name, field_prefix)
            for field_name, field_prefix in (
                ("expected_chunk_ids", "chunk"),
                ("expected_document_ids", "document"),
                ("expected_source_files", "source"),
            )
            if expected.get(field_name)
        ),
        ("expected_source_files", "source"),
    )
    values = expected.get(field) or []
    if isinstance(values, str):
        values = [values]
    return {f"{prefix}:{str(value).strip()}" for value in values if str(value).strip()}


def _reference_ids(reference: dict) -> set[str]:
    metadata = reference.get("metadata") or {}
    values = {
        "chunk": reference.get("chunk_id") or metadata.get("chunk_id"),
        "matched_chunk": reference.get("matched_chunk_id") or metadata.get("matched_chunk_id"),
        "document": reference.get("document_id") or metadata.get("document_id"),
        "source": reference.get("source_file") or metadata.get("source_file"),
    }
    return {
        f"{'chunk' if kind == 'matched_chunk' else kind}:{str(value).strip()}"
        for kind, value in values.items()
        if str(value or "").strip()
    }


def _reference_is_relevant(reference: dict, relevant_ids: set[str]) -> bool:
    if isinstance(reference.get("relevant"), bool):
        return reference["relevant"]
    return bool(relevant_ids & _reference_ids(reference))


def _reciprocal_rank(relevance: list[bool]) -> float:
    for index, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1 / index
    return 0.0


def _is_acl_leak(reference: dict, expected: dict) -> bool:
    metadata = reference.get("metadata") or {}
    decisions = [reference.get(field, metadata.get(field)) for field in ("acl_allowed", "permission_allowed", "authorized")]
    if any(isinstance(value, bool) for value in decisions):
        return any(value is False for value in decisions)
    if "allowed_document_ids" not in expected:
        return False
    document_id = str(reference.get("document_id") or metadata.get("document_id") or "")
    return not document_id or document_id not in _string_values(expected.get("allowed_document_ids"))


def _is_invalid_content(reference: dict, expected: dict) -> bool:
    metadata = reference.get("metadata") or {}
    validity = [reference.get(field, metadata.get(field)) for field in ("is_active", "is_valid")]
    if any(value is False for value in validity):
        return True
    status = str(
        reference.get("publish_status")
        or metadata.get("publish_status")
        or reference.get("status")
        or metadata.get("status")
        or ""
    ).lower()
    if status in _INACTIVE_STATUSES:
        return True
    document_id = str(reference.get("document_id") or metadata.get("document_id") or "")
    forbidden = _string_values(expected.get("retired_document_ids")) | _string_values(expected.get("forbidden_document_ids"))
    if document_id and document_id in forbidden:
        return True
    if any(value is True for value in validity):
        return False
    return False


def _string_values(value) -> set[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _primary_quality_improved(candidate: dict, baseline: dict) -> bool:
    citation_improved = any(
        float(candidate.get(key, 0) or 0) > float(baseline.get(key, 0) or 0) + _EPSILON
        for key in ("citation_precision", "citation_coverage")
    ) or float(candidate.get("no_evidence_answer_rate", 0) or 0) + _EPSILON < float(
        baseline.get("no_evidence_answer_rate", 0) or 0
    )
    if citation_improved:
        return True
    citation_at_ceiling = (
        float(baseline.get("citation_precision", 0) or 0) >= 1 - _EPSILON
        and float(baseline.get("citation_coverage", 0) or 0) >= 1 - _EPSILON
        and float(baseline.get("no_evidence_answer_rate", 0) or 0) <= _EPSILON
    )
    return citation_at_ceiling and any(
        float(candidate.get(key, 0) or 0) > float(baseline.get(key, 0) or 0) + _EPSILON
        for key in ("recall_at_5", "mrr_at_10")
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
