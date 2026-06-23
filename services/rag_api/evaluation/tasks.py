from __future__ import annotations

import threading
import uuid
from datetime import datetime

from services.rag_api.evaluation import storage
from services.rag_api.evaluation.questions import generate_evaluation_question_set
from services.rag_api.evaluation.runner import get_evaluation_total_units, run_evaluation

_RUNNING_LOCK = threading.Lock()
_RUNNING_RUN_ID: str | None = None


def start_evaluation_run() -> dict:
    global _RUNNING_RUN_ID
    worker: threading.Thread | None = None
    with _RUNNING_LOCK:
        if _RUNNING_RUN_ID:
            current = storage.read_evaluation_progress(_RUNNING_RUN_ID)
            if current and current.get("status") in {"queued", "running"}:
                current["message"] = "已有评测任务正在运行，已返回当前进度"
                current["updated_at"] = _now()
                storage.save_evaluation_progress(current)
                return _with_progress_url(current)
            _RUNNING_RUN_ID = None

        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        question_set = generate_evaluation_question_set()
        progress = {
            "run_id": run_id,
            "status": "queued",
            "percent": 0,
            "completed_units": 0,
            "total_units": get_evaluation_total_units(question_set),
            "current_profile": "",
            "current_question": "",
            "message": "动态题集已生成，等待后台执行",
            "started_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        storage.save_evaluation_progress(progress)
        _RUNNING_RUN_ID = run_id
        worker = threading.Thread(target=_run_background, args=(run_id, question_set), daemon=True)
    worker.start()
    return _with_progress_url(progress)


def read_evaluation_progress(run_id: str) -> dict | None:
    progress = storage.read_evaluation_progress(run_id)
    if not progress:
        return None
    return _with_progress_url(progress)


def get_active_evaluation_progress() -> dict:
    if _RUNNING_RUN_ID:
        current = storage.read_evaluation_progress(_RUNNING_RUN_ID)
        if current and current.get("status") in {"queued", "running"}:
            return _with_progress_url(current)

    for progress in storage.list_evaluation_progresses(limit=10):
        if progress.get("status") in {"queued", "running"}:
            return _with_progress_url(progress)
    return {"status": "idle"}


def _run_background(run_id: str, question_set: dict | None = None) -> None:
    global _RUNNING_RUN_ID

    def record(update: dict) -> None:
        current = storage.read_evaluation_progress(run_id) or {"run_id": run_id, "started_at": _now()}
        payload = {**current, **update}
        payload.setdefault("started_at", current.get("started_at") or _now())
        payload["updated_at"] = _now()
        storage.save_evaluation_progress(payload)

    try:
        record({"status": "running", "message": "评测任务开始执行"})
        run_evaluation(run_id=run_id, progress_callback=record, question_set=question_set)
    except Exception as exc:  # noqa: BLE001
        record(
            {
                "status": "failed",
                "message": "评测执行失败",
                "error": str(exc),
            }
        )
    finally:
        with _RUNNING_LOCK:
            if _RUNNING_RUN_ID == run_id:
                _RUNNING_RUN_ID = None


def _with_progress_url(progress: dict) -> dict:
    return {**progress, "progress_url": f"/api/evaluations/{progress['run_id']}/progress"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
