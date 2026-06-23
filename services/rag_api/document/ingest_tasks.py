from __future__ import annotations

import threading
import uuid
from datetime import datetime

from services.rag_api.document.ingest import ingest_knowledge_base
from services.rag_api.document import ingest_storage
from services.rag_api.exceptions import DOC_LOAD_ERROR_MESSAGE, LLM_ERROR_MESSAGE, DocumentLoadError, LLMServiceError

_RUNNING_LOCK = threading.Lock()
_RUNNING_RUN_ID: str | None = None
TOTAL_UNITS = 8


def start_ingest_run() -> dict:
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
            "message": "知识库重建任务已创建，等待后台执行",
            "started_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        ingest_storage.save_ingest_progress(progress)
        _RUNNING_RUN_ID = run_id
        worker = threading.Thread(target=_run_background, args=(run_id,), daemon=True)
    worker.start()
    return _with_progress_url(progress)


def read_ingest_progress(run_id: str) -> dict | None:
    progress = ingest_storage.read_ingest_progress(run_id)
    if not progress:
        return None
    return _with_progress_url(progress)


def read_ingest_result(run_id: str) -> dict | None:
    return ingest_storage.read_ingest_result(run_id)


def _run_background(run_id: str) -> None:
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
        record({"status": "running", "current_step": "开始重建", "message": "知识库重建任务开始执行"})
        result = ingest_knowledge_base(progress_callback=record)
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
            }
        )
    except DocumentLoadError:
        _record_failure(record, DOC_LOAD_ERROR_MESSAGE)
    except LLMServiceError:
        _record_failure(record, LLM_ERROR_MESSAGE)
    except Exception as exc:  # noqa: BLE001
        _record_failure(record, str(exc))
    finally:
        with _RUNNING_LOCK:
            if _RUNNING_RUN_ID == run_id:
                _RUNNING_RUN_ID = None


def _record_failure(record, error: str) -> None:
    record(
        {
            "status": "failed",
            "current_step": "失败",
            "message": "知识库重建失败",
            "error": error,
        }
    )


def _with_progress_url(progress: dict) -> dict:
    return {**progress, "progress_url": f"/api/ingest/{progress['run_id']}/progress"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
