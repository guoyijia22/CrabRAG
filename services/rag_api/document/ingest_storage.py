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


def list_ingest_progresses(limit: int = 20) -> list[dict]:
    if not INGEST_DIR.exists():
        return []
    paths = sorted(INGEST_DIR.glob("*.progress.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    progresses: list[dict] = []
    for path in paths[:limit]:
        run_id = path.name.removesuffix(".progress.json")
        payload = read_ingest_progress(run_id)
        if payload:
            progresses.append(payload)
    return progresses


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
