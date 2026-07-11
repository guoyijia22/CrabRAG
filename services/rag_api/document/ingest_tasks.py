from __future__ import annotations

import threading
import uuid
from datetime import datetime

from services.rag_api import index_generation
from services.rag_api.document.ingest import ingest_knowledge_base
from services.rag_api.document import ingest_storage
from services.rag_api.exceptions import DOC_LOAD_ERROR_MESSAGE, LLM_ERROR_MESSAGE, DocumentLoadError, LLMServiceError

_RUNNING_LOCK = threading.Lock()
_RUNNING_RUN_ID: str | None = None
TOTAL_UNITS = 8


def start_ingest_run(full_rebuild: bool = False) -> dict:
    global _RUNNING_RUN_ID
    worker: threading.Thread | None = None
    with _RUNNING_LOCK:
        if _RUNNING_RUN_ID:
            current = ingest_storage.read_ingest_progress(_RUNNING_RUN_ID)
            if current and current.get("status") in {"queued", "running"}:
                current["message"] = "已有知识库重建任务正在运行，已返回当前进度"
                current["updated_at"] = _now()
                ingest_storage.save_ingest_progress(current)
                return _with_progress_url(current)
            _RUNNING_RUN_ID = None

        run_id = f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        progress = {
            "run_id": run_id,
            "status": "queued",
            "percent": 0,
            "completed_units": 0,
            "total_units": TOTAL_UNITS,
            "current_step": "等待执行",
            "message": "知识库全量重建任务已创建，等待后台执行" if full_rebuild else "知识库增量更新任务已创建，等待后台执行",
            "full_rebuild": full_rebuild,
            "started_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        ingest_storage.save_ingest_progress(progress)
        _RUNNING_RUN_ID = run_id
        worker = threading.Thread(target=_run_background, args=(run_id, full_rebuild), daemon=True)
    worker.start()
    return _with_progress_url(progress)


def read_ingest_progress(run_id: str) -> dict | None:
    progress = ingest_storage.read_ingest_progress(run_id)
    if not progress:
        return None
    return _with_progress_url(progress)


def get_active_ingest_progress() -> dict:
    active: dict | None = None
    with _RUNNING_LOCK:
        running_run_id = _RUNNING_RUN_ID
    if running_run_id:
        current = ingest_storage.read_ingest_progress(running_run_id)
        if current and current.get("status") in {"queued", "running"}:
            active = current
    if active is None:
        for progress in ingest_storage.list_ingest_progresses(limit=20):
            if progress.get("status") in {"queued", "running"}:
                active = progress
                break

    last_success = None
    for progress in ingest_storage.list_ingest_progresses(limit=20):
        if progress.get("status") == "completed":
            last_success = progress
            break

    return {
        "active": _with_progress_url(active) if active else None,
        "last_success": _with_progress_url(last_success) if last_success else None,
    }


def read_ingest_result(run_id: str) -> dict | None:
    return ingest_storage.read_ingest_result(run_id)


def _run_background(run_id: str, full_rebuild: bool = False) -> None:
    global _RUNNING_RUN_ID

    def record(update: dict) -> None:
        current = ingest_storage.read_ingest_progress(run_id) or {"run_id": run_id, "started_at": _now(), "total_units": TOTAL_UNITS}
        payload = {**current, **update}
        payload.setdefault("started_at", current.get("started_at") or _now())
        payload.setdefault("total_units", TOTAL_UNITS)
        completed = int(payload.get("completed_units") or 0)
        total = int(payload.get("total_units") or TOTAL_UNITS)
        payload["percent"] = int(payload.get("percent") if payload.get("percent") is not None else (completed / total * 100 if total else 0))
        payload["updated_at"] = _now()
        ingest_storage.save_ingest_progress(payload)

    try:
        record(
            {
                "status": "running",
                "current_step": "开始重建",
                "message": "知识库全量重建任务开始执行" if full_rebuild else "知识库增量更新任务开始执行",
                "full_rebuild": full_rebuild,
            }
        )
        with index_generation.generation_build_lock():
            result = ingest_knowledge_base(progress_callback=record, full_rebuild=full_rebuild)
        result = {"run_id": run_id, **result}
        ingest_storage.save_ingest_result(result)
        record(
            {
                "status": "completed",
                "percent": 100,
                "completed_units": TOTAL_UNITS,
                "total_units": TOTAL_UNITS,
                "current_step": "完成",
                "message": "知识库重建完成",
                "error": None,
                **_duration_payload(run_id),
            }
        )
    except DocumentLoadError:
        _record_failure(run_id, record, DOC_LOAD_ERROR_MESSAGE)
    except LLMServiceError as exc:
        _record_failure(run_id, record, str(exc) or LLM_ERROR_MESSAGE)
    except Exception as exc:  # noqa: BLE001
        _record_failure(run_id, record, str(exc))
    finally:
        with _RUNNING_LOCK:
            if _RUNNING_RUN_ID == run_id:
                _RUNNING_RUN_ID = None


def _record_failure(run_id: str, record, error: str) -> None:
    record(
        {
            "status": "failed",
            "current_step": "失败",
            "message": "知识库重建失败",
            "error": error,
            **_duration_payload(run_id),
        }
    )


def _with_progress_url(progress: dict) -> dict:
    return {**progress, "progress_url": f"/api/ingest/{progress['run_id']}/progress"}


def _duration_payload(run_id: str) -> dict:
    progress = ingest_storage.read_ingest_progress(run_id) or {}
    started_at = progress.get("started_at") or _now()
    finished_at = _now()
    duration_seconds = _elapsed_seconds(started_at, finished_at)
    return {
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "duration_label": _format_duration(duration_seconds),
    }


def _elapsed_seconds(started_at: str, finished_at: str) -> int:
    try:
        start = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        finish = datetime.strptime(finished_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0
    return max(0, int((finish - start).total_seconds()))


def _format_duration(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分")
    if seconds or not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
