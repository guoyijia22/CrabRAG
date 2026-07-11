from __future__ import annotations

import hashlib
import json
from typing import Any


def evaluation_configuration_fingerprint(
    *,
    generation_id: str,
    permission_fingerprint: str,
    question_generation: dict[str, Any],
    profiles: list[dict],
) -> str:
    dataset = {
        key: question_generation.get(key)
        for key in (
            "mode",
            "fixed",
            "gate_eligible",
            "schema_version",
            "dataset_id",
            "dataset_version",
            "dataset_fingerprint",
        )
        if key in question_generation
    }
    profile_configs = [
        {
            "id": profile.get("id"),
            "enabled_switches": profile.get("enabled_switches") or [],
            "collection_name": profile.get("collection_name"),
            "settings": profile.get("settings") or {},
        }
        for profile in profiles
    ]
    payload = {
        "generation_id": generation_id,
        "permission_fingerprint": permission_fingerprint,
        "dataset": dataset,
        "profiles": profile_configs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
