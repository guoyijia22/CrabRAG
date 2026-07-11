from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from services.rag_api.rag_settings import PROJECT_ROOT

DATASET_SCHEMA_VERSION = 1
DEFAULT_DATASET_PATH = PROJECT_ROOT / "config" / "evaluation-dataset.json"
_RUNTIME_FIELDS = {"fingerprint", "fixed", "gate_eligible"}


class EvaluationDatasetError(ValueError):
    """Raised when a fixed evaluation dataset cannot be trusted."""


def load_evaluation_dataset(path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDatasetError(f"invalid evaluation dataset JSON: {exc}") from exc
    return validate_evaluation_dataset(payload)


def validate_evaluation_dataset(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvaluationDatasetError("evaluation dataset must be a JSON object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != DATASET_SCHEMA_VERSION:
        raise EvaluationDatasetError(f"unsupported schema_version: {schema_version!r}")
    dataset_id = _required_text(payload, "dataset_id")
    dataset_version = _required_text(payload, "dataset_version")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationDatasetError("cases must be a non-empty list")

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise EvaluationDatasetError(f"case {index} must be an object")
        case_id = _required_text(raw_case, "id", prefix=f"case {index} ")
        if case_id in case_ids:
            raise EvaluationDatasetError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        question = _required_text(raw_case, "question", prefix=f"case {case_id} ")
        cases.append({**raw_case, "id": case_id, "question": question})

    normalized = {
        **{key: value for key, value in payload.items() if key not in _RUNTIME_FIELDS},
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "cases": cases,
    }
    normalized["fingerprint"] = _fingerprint(normalized)
    normalized["fixed"] = True
    normalized["gate_eligible"] = True
    return normalized


def dataset_to_question_set(dataset: dict[str, Any]) -> dict[str, Any]:
    validated = validate_evaluation_dataset(dataset)
    questions = [dict(case) for case in validated["cases"]]
    return {
        "question_generation": {
            "mode": "fixed",
            "fixed": True,
            "gate_eligible": True,
            "question_count": len(questions),
            "schema_version": validated["schema_version"],
            "dataset_id": validated["dataset_id"],
            "dataset_version": validated["dataset_version"],
            "dataset_fingerprint": validated["fingerprint"],
        },
        "questions": questions,
    }


def _fingerprint(dataset: dict[str, Any]) -> str:
    canonical = {key: value for key, value in dataset.items() if key not in _RUNTIME_FIELDS}
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required_text(payload: dict[str, Any], field: str, *, prefix: str = "") -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDatasetError(f"{prefix}{field} must be a non-empty string")
    return value.strip()
