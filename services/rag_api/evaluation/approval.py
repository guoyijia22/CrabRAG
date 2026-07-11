from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.rag_api import index_generation
from services.rag_api.evaluation.profiles import SWITCH_KEYS
from services.rag_api.rag_settings import PROJECT_ROOT, RagSettings

APPROVALS_PATH = PROJECT_ROOT / "data" / "evaluations" / "quality-approvals.json"


def settings_fingerprint(settings: RagSettings) -> str:
    encoded = json.dumps(settings.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_quality_approvals(run: dict[str, Any]) -> list[str]:
    generation = run.get("question_generation") or {}
    if not generation.get("fixed") or not generation.get("gate_eligible"):
        return []
    generation_id = str(run.get("generation_id") or "")
    dataset_fingerprint = str(generation.get("dataset_fingerprint") or "")
    if not generation_id or generation_id == "legacy" or not dataset_fingerprint:
        return []

    approvals = _load_approvals()
    items = list(approvals.get("items") or [])
    recorded: list[str] = []
    for profile in run.get("profiles") or []:
        gate = (profile.get("summary") or {}).get("quality_gate") or {}
        if gate.get("eligible") is not True or gate.get("passed") is not True:
            continue
        try:
            settings = RagSettings.model_validate(profile.get("settings") or {})
        except Exception:
            continue
        fingerprint = settings_fingerprint(settings)
        item = {
            "generation_id": generation_id,
            "dataset_id": str(generation.get("dataset_id") or ""),
            "dataset_version": str(generation.get("dataset_version") or ""),
            "dataset_fingerprint": dataset_fingerprint,
            "run_id": str(run.get("run_id") or ""),
            "profile_id": str(profile.get("id") or ""),
            "enabled_switches": [key for key in SWITCH_KEYS if bool(getattr(settings, key, False))],
            "settings_fingerprint": fingerprint,
            "approved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        items = [
            existing
            for existing in items
            if not (
                str(existing.get("generation_id")) == generation_id
                and str(existing.get("settings_fingerprint")) == fingerprint
            )
        ]
        items.append(item)
        recorded.append(item["profile_id"])
    if recorded:
        _save_approvals({"schema_version": 1, "items": items})
    return recorded


def is_settings_approved(settings: RagSettings, generation_id: str | None) -> bool:
    if not generation_id or generation_id == "legacy":
        return False
    fingerprint = settings_fingerprint(settings)
    return any(
        str(item.get("generation_id") or "") == generation_id
        and str(item.get("settings_fingerprint") or "") == fingerprint
        for item in (_load_approvals().get("items") or [])
    )


def require_strategy_approval(current: RagSettings, candidate: RagSettings, generation_id: str | None) -> None:
    del current
    enabled = [key for key in SWITCH_KEYS if bool(getattr(candidate, key, False))]
    if generation_id and enabled and not is_settings_approved(candidate, generation_id):
        raise ValueError(
            "启用检索增强策略前，必须在当前索引代使用固定评测集通过质量门禁；"
            f"待批准策略：{', '.join(enabled)}"
        )


def effective_runtime_settings(settings: RagSettings) -> RagSettings:
    enabled = [key for key in SWITCH_KEYS if bool(getattr(settings, key, False))]
    if not enabled:
        return settings
    try:
        generation_id = index_generation.active_generation_id()
    except Exception:  # noqa: BLE001
        generation_id = "unavailable"
    if not generation_id:
        return settings
    if is_settings_approved(settings, generation_id):
        return settings
    return settings.model_copy(update={key: False for key in SWITCH_KEYS})


def _load_approvals() -> dict[str, Any]:
    if not APPROVALS_PATH.exists():
        return {"schema_version": 1, "items": []}
    try:
        payload = json.loads(APPROVALS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "items": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return {"schema_version": 1, "items": []}
    return payload


def _save_approvals(payload: dict[str, Any]) -> None:
    APPROVALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = APPROVALS_PATH.with_suffix(APPROVALS_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, APPROVALS_PATH)
