from __future__ import annotations

import json
from pathlib import Path

from services.rag_api.config import Settings
from services.rag_api.rag_settings import RagSettings


def _configure_incremental_paths(tmp_path: Path, monkeypatch):
    from services.rag_api.document import categories, doc_status, ingest
    from services.rag_api.graph import graph_store, kb_graph_builder

    status_path = tmp_path / "data" / "ingest" / "doc_status.json"
    snapshot_dir = tmp_path / "data" / "ingest" / "doc_snapshots"
    graph_path = tmp_path / "data" / "kb_graph.json"
    categories_path = tmp_path / "data" / "kb_categories.json"
    monkeypatch.setattr(doc_status, "DOC_STATUS_PATH", status_path)
    monkeypatch.setattr(doc_status, "DOC_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(kb_graph_builder, "KB_GRAPH_PATH", graph_path)
    monkeypatch.setattr(graph_store, "KB_GRAPH_PATH", graph_path)
    monkeypatch.setattr(categories, "KB_CATEGORIES_PATH", categories_path)
    monkeypatch.setattr(ingest, "ensure_knowledge_base_name", lambda category_payload, documents, chunk_count: ("测试知识库", "test"))
    monkeypatch.setattr(ingest, "generate_graph_schema_suggestion", lambda category_payload, documents, chunks: {})
    return status_path, snapshot_dir, graph_path


def _settings(docs_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        docs_dir=docs_dir,
        docs_dirs=[docs_dir],
        chroma_dir=tmp_path / "chroma",
        collection_name="incremental_test",
        embedding_provider="api",
        embedding_model="test-embedding",
    )


def test_incremental_ingest_skips_unchanged_documents(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("客户准入材料要求\n必须提供营业执照。", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    graph_calls = []
    upserts = []
    monkeypatch.setattr(
        ingest,
        "upsert_chunks_incremental",
        lambda chunks, delete_chunk_ids=None, progress_callback=None: upserts.append(
            {"ids": [chunk["id"] for chunk in chunks], "delete_chunk_ids": list(delete_chunk_ids or [])}
        )
        or len(chunks),
    )
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_incremental",
        lambda nodes, edges: graph_calls.append((len(nodes), len(edges)))
        or {"graph_entity_index_count": len(nodes), "graph_relationship_index_count": len(edges)},
    )

    first = ingest.ingest_knowledge_base()
    second = ingest.ingest_knowledge_base()

    assert first["incremental"] is True
    assert first["processed_document_count"] == 1
    assert first["skipped_document_count"] == 0
    assert first["reindexed_chunk_count"] == len(upserts[0]["ids"])
    assert upserts[0]["ids"]
    assert upserts[0]["delete_chunk_ids"] == []
    assert second["processed_document_count"] == 0
    assert second["skipped_document_count"] == 1
    assert second["reindexed_chunk_count"] == 0
    assert len(upserts) == 1
    assert graph_calls[-1][0] == first["graph_node_count"]


def test_incremental_ingest_removes_deleted_documents_from_vectors_and_graph(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc_path = docs_dir / "a.txt"
    doc_path.write_text("关于一渠一表的报告\n项目材料包含一渠一表。", encoding="utf-8")
    _, _, graph_path = _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    upserts = []
    monkeypatch.setattr(
        ingest,
        "upsert_chunks_incremental",
        lambda chunks, delete_chunk_ids=None, progress_callback=None: upserts.append(
            {"ids": [chunk["id"] for chunk in chunks], "delete_chunk_ids": list(delete_chunk_ids or [])}
        )
        or len(chunks),
    )
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_incremental",
        lambda nodes, edges: {"graph_entity_index_count": len(nodes), "graph_relationship_index_count": len(edges)},
    )

    first = ingest.ingest_knowledge_base()
    doc_path.unlink()
    second = ingest.ingest_knowledge_base()

    assert first["graph_node_count"] > 0
    assert second["processed_document_count"] == 0
    assert second["removed_document_count"] == 1
    assert second["document_count"] == 0
    assert second["chunk_count"] == 0
    assert second["graph_node_count"] == 0
    assert second["graph_edge_count"] == 0
    assert upserts[-1]["ids"] == []
    assert upserts[-1]["delete_chunk_ids"]
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_incremental_ingest_skips_duplicate_content(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("相同的知识库内容", encoding="utf-8")
    (docs_dir / "b.txt").write_text("相同的知识库内容", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    upserts = []
    monkeypatch.setattr(
        ingest,
        "upsert_chunks_incremental",
        lambda chunks, delete_chunk_ids=None, progress_callback=None: upserts.append([chunk["id"] for chunk in chunks]) or len(chunks),
    )
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_incremental",
        lambda nodes, edges: {"graph_entity_index_count": len(nodes), "graph_relationship_index_count": len(edges)},
    )

    result = ingest.ingest_knowledge_base()

    assert result["processed_document_count"] == 1
    assert result["duplicate_document_count"] == 1
    assert result["document_count"] == 1
    assert len(upserts) == 1


def test_incremental_ingest_reprocesses_when_pipeline_fingerprint_changes(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("客户准入材料要求\n必须提供营业执照。", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    active_settings = {"chunk_size": 200}
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(
        ingest,
        "load_rag_settings",
        lambda: RagSettings(chunk_size=active_settings["chunk_size"], chunk_overlap=20),
    )
    upserts = []
    monkeypatch.setattr(
        ingest,
        "upsert_chunks_incremental",
        lambda chunks, delete_chunk_ids=None, progress_callback=None: upserts.append(
            {"ids": [chunk["id"] for chunk in chunks], "delete_chunk_ids": list(delete_chunk_ids or [])}
        )
        or len(chunks),
    )
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_incremental",
        lambda nodes, edges: {"graph_entity_index_count": len(nodes), "graph_relationship_index_count": len(edges)},
    )

    first = ingest.ingest_knowledge_base()
    active_settings["chunk_size"] = 220
    second = ingest.ingest_knowledge_base()

    assert first["processed_document_count"] == 1
    assert second["processed_document_count"] == 1
    assert second["skipped_document_count"] == 0
    assert len(upserts) == 2
    assert upserts[1]["delete_chunk_ids"] == upserts[0]["ids"]


def test_split_documents_uses_stable_doc_ids_for_incremental_chunks():
    from services.rag_api.document.splitter import split_documents

    chunks = split_documents(
        [
            {
                "doc_id": "doc-abc",
                "source_file": "a.txt",
                "source_path": "/tmp/a.txt",
                "content_hash": "hash-abc",
                "content": "客户准入材料要求\n必须提供营业执照。",
            }
        ],
        chunk_size=200,
        chunk_overlap=20,
    )

    assert chunks[0]["id"] == "doc-abc::chunk::0001"
    assert chunks[0]["metadata"]["doc_id"] == "doc-abc"
    assert chunks[0]["metadata"]["content_hash"] == "hash-abc"
