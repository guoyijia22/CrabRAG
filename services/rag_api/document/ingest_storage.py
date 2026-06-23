from __future__ import annotations

import json

from services.rag_api.rag_settings import PROJECT_ROOT

INGEST_DIR = PROJECT_ROOT / "data" / "ingest"


def save_ingest_progress(payload: dict) -> dict:
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    path = INGEST_DIR / f"{payload['run_id']}.progress.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_ingest_progress(run_id: str) -> dict | None:
    path = INGEST_DIR / f"{run_id}.progress.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_ingest_result(payload: dict) -> dict:
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    path = INGEST_DIR / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_ingest_result(run_id: str) -> dict | None:
    path = INGEST_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
