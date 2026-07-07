from __future__ import annotations

from services.rag_api.document import ingest_storage, ingest_tasks
from services.rag_api.vector import chroma_store


def _write_progress(run_id: str, **payload) -> None:
    ingest_storage.save_ingest_progress({"run_id": run_id, **payload})


def test_completed_ingest_records_duration(monkeypatch):
    progress_records = []
    results = {}

    def fake_read(run_id):
        return progress_records[-1].copy() if progress_records else {"run_id": run_id, "started_at": "2026-06-25 10:00:00", "total_units": ingest_tasks.TOTAL_UNITS}

    def fake_save(payload):
        progress_records.append(payload.copy())

    monkeypatch.setattr(ingest_tasks.ingest_storage, "read_ingest_progress", fake_read)
    monkeypatch.setattr(ingest_tasks.ingest_storage, "save_ingest_progress", fake_save)
    monkeypatch.setattr(ingest_tasks.ingest_storage, "save_ingest_result", lambda payload: results.update(payload))
    monkeypatch.setattr(ingest_tasks, "ingest_knowledge_base", lambda progress_callback=None, full_rebuild=False: {"status": "success", "chunk_count": 3})
    monkeypatch.setattr(ingest_tasks, "_now", lambda: "2026-06-25 10:03:05")
    monkeypatch.setattr(ingest_tasks, "_RUNNING_RUN_ID", "run_complete")

    ingest_tasks._run_background("run_complete")

    assert results["run_id"] == "run_complete"
    assert progress_records[-1]["status"] == "completed"
    assert progress_records[-1]["finished_at"] == "2026-06-25 10:03:05"
    assert progress_records[-1]["duration_seconds"] == 185
    assert progress_records[-1]["duration_label"] == "3分5秒"


def test_full_rebuild_flag_is_passed_to_background_ingest(monkeypatch):
    calls = []
    progress_records = []
    results = {}

    monkeypatch.setattr(
        ingest_tasks.ingest_storage,
        "read_ingest_progress",
        lambda run_id: progress_records[-1].copy()
        if progress_records
        else {"run_id": run_id, "started_at": "2026-06-25 10:00:00", "total_units": ingest_tasks.TOTAL_UNITS},
    )
    monkeypatch.setattr(ingest_tasks.ingest_storage, "save_ingest_progress", lambda payload: progress_records.append(payload.copy()))
    monkeypatch.setattr(ingest_tasks.ingest_storage, "save_ingest_result", lambda payload: results.update(payload))

    def fake_ingest(progress_callback=None, full_rebuild=False):
        calls.append(full_rebuild)
        return {"status": "success", "chunk_count": 3, "incremental": not full_rebuild}

    monkeypatch.setattr(ingest_tasks, "ingest_knowledge_base", fake_ingest)
    monkeypatch.setattr(ingest_tasks, "_now", lambda: "2026-06-25 10:00:05")
    monkeypatch.setattr(ingest_tasks, "_RUNNING_RUN_ID", "run_full")

    ingest_tasks._run_background("run_full", full_rebuild=True)

    assert calls == [True]
    assert results["incremental"] is False
    assert progress_records[-1]["status"] == "completed"


def test_active_ingest_returns_running_and_last_success(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_storage, "INGEST_DIR", tmp_path)
    monkeypatch.setattr(ingest_tasks, "_RUNNING_RUN_ID", None)
    _write_progress(
        "run_success",
        status="completed",
        percent=100,
        updated_at="2026-06-25 09:00:00",
        duration_seconds=65,
        duration_label="1分5秒",
    )
    _write_progress(
        "run_running",
        status="running",
        percent=35,
        updated_at="2026-06-25 10:00:00",
        duration_seconds=None,
        duration_label="",
    )

    payload = ingest_tasks.get_active_ingest_progress()

    assert payload["active"]["run_id"] == "run_running"
    assert payload["last_success"]["run_id"] == "run_success"
    assert payload["last_success"]["duration_label"] == "1分5秒"
    assert payload["active"]["progress_url"] == "/api/ingest/run_running/progress"


def test_failed_ingest_does_not_replace_last_success(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_storage, "INGEST_DIR", tmp_path)
    monkeypatch.setattr(ingest_tasks, "_RUNNING_RUN_ID", None)
    _write_progress(
        "run_success",
        status="completed",
        percent=100,
        updated_at="2026-06-25 09:00:00",
        duration_seconds=120,
        duration_label="2分",
    )
    _write_progress(
        "run_failed",
        status="failed",
        percent=40,
        updated_at="2026-06-25 10:00:00",
        duration_seconds=30,
        duration_label="30秒",
    )

    payload = ingest_tasks.get_active_ingest_progress()

    assert payload["active"] is None
    assert payload["last_success"]["run_id"] == "run_success"


def test_add_chunks_reports_embedding_batch_progress(monkeypatch):
    chunks = [
        {"id": f"chunk-{index}", "content": f"content-{index}", "metadata": {"source_file": "doc.txt"}}
        for index in range(5)
    ]
    progress_records = []
    added_batches = []

    class FakeCollection:
        def add(self, **kwargs):
            if len(kwargs["ids"]) > 2:
                raise ValueError(f"Batch size of {len(kwargs['ids'])} is greater than max batch size of 2")
            added_batches.append(kwargs)

    monkeypatch.setattr(chroma_store, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(chroma_store, "embed_texts", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    monkeypatch.setattr(chroma_store, "reset_collection", lambda: FakeCollection())
    monkeypatch.setattr(chroma_store, "get_settings", lambda: type("Settings", (), {"embedding_provider": "local_onnx"})())
    monkeypatch.setattr(chroma_store, "_get_chroma_client", lambda: type("FakeClient", (), {"get_max_batch_size": lambda self: 2})(), raising=False)

    count = chroma_store.add_chunks(chunks, progress_callback=progress_records.append)

    assert count == 5
    assert len(progress_records) == 3
    assert progress_records[0]["current_step"] == "本地向量化"
    assert "本地向量化中：第 1 / 3 批" in progress_records[0]["message"]
    assert progress_records[-1]["detail_current"] == 3
    assert progress_records[-1]["detail_total"] == 3
    assert "已处理 5 / 5 个片段" in progress_records[-1]["message"]
    assert [batch["ids"] for batch in added_batches] == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_upsert_chunks_incremental_deletes_old_ids_and_batches_new_vectors(monkeypatch):
    chunks = [
        {
            "id": f"doc-a::chunk::{index:04d}",
            "content": f"content-{index}",
            "metadata": {"source_file": "doc-a.txt", "doc_id": "doc-a"},
        }
        for index in range(5)
    ]
    deleted = []
    upserted_batches = []

    class FakeCollection:
        def delete(self, *, ids):
            deleted.extend(ids)

        def upsert(self, **kwargs):
            if len(kwargs["ids"]) > 2:
                raise ValueError(f"Batch size of {len(kwargs['ids'])} is greater than max batch size of 2")
            upserted_batches.append(kwargs)

    monkeypatch.setattr(chroma_store, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(chroma_store, "embed_texts", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    monkeypatch.setattr(chroma_store, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(chroma_store, "get_settings", lambda: type("Settings", (), {"embedding_provider": "api"})())
    monkeypatch.setattr(chroma_store, "_get_chroma_client", lambda: type("FakeClient", (), {"get_max_batch_size": lambda self: 2})(), raising=False)

    count = chroma_store.upsert_chunks_incremental(chunks, delete_chunk_ids=["old-1", "old-2"])

    assert count == 5
    assert deleted == ["old-1", "old-2"]
    assert [batch["ids"] for batch in upserted_batches] == [
        ["doc-a::chunk::0000", "doc-a::chunk::0001"],
        ["doc-a::chunk::0002", "doc-a::chunk::0003"],
        ["doc-a::chunk::0004"],
    ]
