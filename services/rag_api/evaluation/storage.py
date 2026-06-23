from __future__ import annotations

import json
from pathlib import Path

from services.rag_api.rag_settings import PROJECT_ROOT

EVALUATIONS_DIR = PROJECT_ROOT / "data" / "evaluations"


def save_evaluation_run(payload: dict) -> dict:
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATIONS_DIR / f"{payload['run_id']}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def save_evaluation_progress(payload: dict) -> dict:
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATIONS_DIR / f"{payload['run_id']}.progress.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_evaluation_run(run_id: str) -> dict | None:
    path = EVALUATIONS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return _decorate_run(payload)


def read_evaluation_progress(run_id: str) -> dict | None:
    path = EVALUATIONS_DIR / f"{run_id}.progress.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_evaluation_runs(limit: int = 20) -> list[dict]:
    if not EVALUATIONS_DIR.exists():
        return []
    items = []
    paths = [path for path in EVALUATIONS_DIR.glob("*.json") if not path.name.endswith(".progress.json")]
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        payload = read_evaluation_run(path.stem)
        if not payload:
            continue
        items.append(
            {
                "run_id": payload.get("run_id", path.stem),
                "created_at": payload.get("created_at", ""),
                "profile_count": payload.get("profile_count", 0),
                "question_count": payload.get("question_count", 0),
                "summary": payload.get("summary", {}),
            }
        )
    return items


def list_evaluation_progresses(limit: int = 20) -> list[dict]:
    if not EVALUATIONS_DIR.exists():
        return []
    items = []
    paths = sorted(EVALUATIONS_DIR.glob("*.progress.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        payload = read_evaluation_progress(path.name.removesuffix(".progress.json"))
        if payload:
            items.append(payload)
    return items


def _decorate_run(payload: dict) -> dict:
    try:
        from services.rag_api.evaluation.scoring import decorate_evaluation_run

        return decorate_evaluation_run(payload)
    except Exception:  # noqa: BLE001
        return payload
