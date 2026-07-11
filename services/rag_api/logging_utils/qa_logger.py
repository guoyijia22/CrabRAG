from __future__ import annotations

import json
from datetime import datetime

from services.rag_api.config import get_settings


def append_qa_log(item: dict) -> None:
    settings = get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **item}
    with (settings.logs_dir / "qa.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_qa_logs(intent: str | None = None, *, permission_fingerprint: str | None = None) -> list[dict]:
    path = get_settings().logs_dir / "qa.jsonl"
    if not path.exists():
        return []
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if intent and item.get("intent") != intent:
                continue
            if permission_fingerprint is not None and item.get("permission_fingerprint") != permission_fingerprint:
                continue
            items.append(item)
    return list(reversed(items[-200:]))
