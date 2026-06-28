from __future__ import annotations

from services.rag_api.document import ingest_tasks
from services.rag_api.exceptions import LLMServiceError


def test_ingest_background_reports_specific_llm_service_error(monkeypatch):
    progress_records = []
    results = {}

    def fake_read(run_id):
        return progress_records[-1].copy() if progress_records else {"run_id": run_id, "total_units": ingest_tasks.TOTAL_UNITS}

    def fake_save(payload):
        progress_records.append(payload.copy())

    def fake_save_result(payload):
        results.update(payload)

    monkeypatch.setattr(ingest_tasks.ingest_storage, "read_ingest_progress", fake_read)
    monkeypatch.setattr(ingest_tasks.ingest_storage, "save_ingest_progress", fake_save)
    monkeypatch.setattr(ingest_tasks.ingest_storage, "save_ingest_result", fake_save_result)
    monkeypatch.setattr(
        ingest_tasks,
        "ingest_knowledge_base",
        lambda progress_callback=None: (_ for _ in ()).throw(LLMServiceError("本地向量模型文件缺失：model.onnx")),
    )
    monkeypatch.setattr(ingest_tasks, "_RUNNING_RUN_ID", "run_local_missing")

    ingest_tasks._run_background("run_local_missing")

    assert results == {}
    assert progress_records[-1]["status"] == "failed"
    assert progress_records[-1]["error"] == "本地向量模型文件缺失：model.onnx"
