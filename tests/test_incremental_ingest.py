from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.rag_api.config import Settings
from services.rag_api.rag_settings import RagSettings


def _configure_incremental_paths(tmp_path: Path, monkeypatch):
    from services.rag_api import index_generation
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
    index_root = tmp_path / "data" / "index"
    monkeypatch.setattr(index_generation, "INDEX_ROOT", index_root)
    monkeypatch.setattr(index_generation, "ACTIVE_INDEX_PATH", index_root / "active.json")
    monkeypatch.setattr(index_generation, "GENERATIONS_DIR", index_root / "generations")
    monkeypatch.setattr(ingest, "ensure_knowledge_base_name", lambda category_payload, documents, chunk_count: ("测试知识库", "test"))
    def save_schema_suggestion(category_payload, documents, chunks, path=None):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(ingest, "generate_graph_schema_suggestion", save_schema_suggestion)
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
    builds = _capture_generation_builds(ingest, monkeypatch)

    first = ingest.ingest_knowledge_base()
    second = ingest.ingest_knowledge_base()

    assert first["incremental"] is True
    assert first["processed_document_count"] == 1
    assert first["skipped_document_count"] == 0
    assert first["reindexed_chunk_count"] == len(builds[0]["ids"])
    assert builds[0]["ids"]
    assert second["processed_document_count"] == 0
    assert second["skipped_document_count"] == 1
    assert second["reindexed_chunk_count"] == 0
    assert len(builds) == 2
    assert builds[1]["embedded"] == 0


def test_incremental_ingest_indexes_only_active_manifest_version(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy-v1.txt").write_text("旧版政策", encoding="utf-8")
    (docs_dir / "policy-v2.txt").write_text("新版政策", encoding="utf-8")
    (docs_dir / ".crabrag-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "knowledge_base_id": "kb-test",
                "documents": [
                    {
                        "document_id": "policy",
                        "path": "policy-v1.txt",
                        "version": "1",
                        "status": "published",
                        "effective_at": "2025-01-01T00:00:00Z",
                        "updated_at": "2025-01-01T00:00:00Z",
                        "acl": {"visibility": "public", "revision": "1"},
                    },
                    {
                        "document_id": "policy",
                        "path": "policy-v2.txt",
                        "version": "2",
                        "status": "published",
                        "effective_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "acl": {"visibility": "restricted", "roles": ["sales"], "revision": "2"},
                    },
                ],
                "audit_warnings": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _capture_generation_builds(ingest, monkeypatch):
    builds = []
    known_embeddings = set()

    def build(chunks, generation_id, full_rebuild=False, progress_callback=None):
        keys = [
            (
                chunk.get("metadata", {}).get("chunk_hash"),
                chunk.get("metadata", {}).get("granularity", "chunk"),
                chunk.get("metadata", {}).get("embedding_fingerprint"),
            )
            for chunk in chunks
        ]
        embedded = len(keys) if full_rebuild else sum(key not in known_embeddings for key in keys)
        known_embeddings.update(keys)
        builds.append(
            {
                "generation_id": generation_id,
                "ids": [chunk["id"] for chunk in chunks],
                "chunks": chunks,
                "full_rebuild": full_rebuild,
                "embedded": embedded,
            }
        )
        return {
            "chunk_count": len(chunks),
            "reused_embedding_count": len(chunks) - embedded,
            "embedded_chunk_count": embedded,
        }

    monkeypatch.setattr(ingest, "build_generation_chunks", build)
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_generation",
        lambda nodes, edges, generation_id, full_rebuild=False: {
            "graph_entity_index_count": len(nodes),
            "graph_relationship_index_count": len(edges),
            "graph_reused_embedding_count": 0,
            "graph_embedded_record_count": len(nodes) + len(edges),
        },
    )
    return builds
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    builds = _capture_generation_builds(ingest, monkeypatch)

    result = ingest.ingest_knowledge_base()

    assert result["document_count"] == 1
    assert result["processed_document_count"] == 1
    assert {chunk["metadata"]["document_id"] for chunk in builds[0]["chunks"]} == {"policy"}
    assert {chunk["metadata"]["document_version"] for chunk in builds[0]["chunks"]} == {"2"}
    assert {chunk["metadata"]["acl_revision"] for chunk in builds[0]["chunks"]} == {"2"}


def test_ingest_publishes_generation_only_after_all_indexes_succeed(tmp_path, monkeypatch):
    from services.rag_api import index_generation
    from services.rag_api.document import ingest
    from services.rag_api.memory import conversation_memory
    from services.rag_api.retrieval.cache import RETRIEVAL_CACHE

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("生产知识", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    monkeypatch.setattr(
        ingest,
        "build_generation_chunks",
        lambda chunks, generation_id, full_rebuild=False, progress_callback=None: {
            "chunk_count": len(chunks),
            "reused_embedding_count": 0,
            "embedded_chunk_count": len(chunks),
        },
        raising=False,
    )
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_generation",
        lambda nodes, edges, generation_id, full_rebuild=False: {
            "graph_entity_index_count": len(nodes),
            "graph_relationship_index_count": len(edges),
            "graph_reused_embedding_count": 0,
            "graph_embedded_record_count": len(nodes) + len(edges),
        },
        raising=False,
    )
    RETRIEVAL_CACHE.clear()
    RETRIEVAL_CACHE.set("old", {"chunks": []})
    conversation_memory.SESSION_MEMORY.clear()
    conversation_memory.update_memory("session", "question", "answer", "", [])

    result = ingest.ingest_knowledge_base()

    assert result["generation_id"].startswith("gen-")
    assert index_generation.load_index_state()["active_generation"] == result["generation_id"]
    assert index_generation.generation_artifact_path(result["generation_id"], "categories.json").exists()
    assert index_generation.generation_artifact_path(result["generation_id"], "graph.json").exists()
    assert RETRIEVAL_CACHE.stats()["size"] == 0
    assert conversation_memory.SESSION_MEMORY == {}


def test_ingest_failure_keeps_previous_generation_active(tmp_path, monkeypatch):
    from services.rag_api import index_generation
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("生产知识", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    index_generation.publish_generation("gen-stable", {"permission_schema_version": 1})
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    monkeypatch.setattr(
        ingest,
        "build_generation_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vector build failed")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="vector build failed"):
        ingest.ingest_knowledge_base()

    assert index_generation.load_index_state()["active_generation"] == "gen-stable"


def test_incremental_ingest_removes_deleted_documents_from_vectors_and_graph(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc_path = docs_dir / "a.txt"
    doc_path.write_text("关于一渠一表的报告\n项目材料包含一渠一表。", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    builds = _capture_generation_builds(ingest, monkeypatch)

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
    assert builds[-1]["ids"] == []
    assert second["removed_document_count"] == 1
    from services.rag_api import index_generation

    payload = json.loads(index_generation.generation_artifact_path(second["generation_id"], "graph.json").read_text(encoding="utf-8"))
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_incremental_ingest_keeps_duplicate_content_per_document(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("相同的知识库内容", encoding="utf-8")
    (docs_dir / "b.txt").write_text("相同的知识库内容", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    builds = _capture_generation_builds(ingest, monkeypatch)

    result = ingest.ingest_knowledge_base()

    assert result["processed_document_count"] == 2
    assert result["duplicate_document_count"] == 1
    assert result["document_count"] == 2
    assert len({chunk["metadata"]["document_id"] for chunk in builds[0]["chunks"]}) == 2
    assert len(builds) == 1


def test_duplicate_copy_is_promoted_when_original_document_is_removed(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    original = docs_dir / "a.txt"
    copy = docs_dir / "b.txt"
    original.write_text("相同的知识库内容", encoding="utf-8")
    copy.write_text("相同的知识库内容", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    builds = _capture_generation_builds(ingest, monkeypatch)

    first = ingest.ingest_knowledge_base()
    original.unlink()
    second = ingest.ingest_knowledge_base()

    assert first["document_count"] == 2
    assert first["duplicate_document_count"] == 1
    assert second["document_count"] == 1
    assert second["processed_document_count"] == 0
    assert second["skipped_document_count"] == 1
    assert {chunk["metadata"]["source_file"] for chunk in builds[-1]["chunks"]} == {"b.txt"}


def test_version_replacement_records_previous_version_tombstone(tmp_path, monkeypatch):
    from services.rag_api import index_generation
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    document_path = docs_dir / "policy.txt"
    document_path.write_text("版本一", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    _capture_generation_builds(ingest, monkeypatch)

    ingest.ingest_knowledge_base()
    manifest_path = docs_dir / ".crabrag-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["version"] = "2"
    manifest["documents"][0]["updated_at"] = "2026-07-11T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    document_path.write_text("版本二", encoding="utf-8")

    result = ingest.ingest_knowledge_base()
    generation_manifest = index_generation.load_generation_manifest(result["generation_id"])

    assert any(
        item["document_version"] == "1" and item["reason"] == "version_replaced"
        for item in generation_manifest["tombstones"]
    )


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
    builds = _capture_generation_builds(ingest, monkeypatch)

    first = ingest.ingest_knowledge_base()
    active_settings["chunk_size"] = 220
    second = ingest.ingest_knowledge_base()

    assert first["processed_document_count"] == 1
    assert second["processed_document_count"] == 1
    assert second["skipped_document_count"] == 0
    assert len(builds) == 2
    assert builds[1]["embedded"] == 0


def test_full_rebuild_reprocesses_unchanged_documents(tmp_path, monkeypatch):
    from services.rag_api.document import ingest

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("客户准入材料要求\n必须提供营业执照。", encoding="utf-8")
    _configure_incremental_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(ingest, "get_settings", lambda: _settings(docs_dir, tmp_path))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings(chunk_size=200, chunk_overlap=20))
    builds = _capture_generation_builds(ingest, monkeypatch)

    first = ingest.ingest_knowledge_base()
    second = ingest.ingest_knowledge_base(full_rebuild=True)

    assert first["incremental"] is True
    assert second["incremental"] is False
    assert second["processed_document_count"] == 1
    assert second["skipped_document_count"] == 0
    assert len(builds) == 2
    assert builds[1]["full_rebuild"] is True
    assert builds[1]["embedded"] == len(builds[1]["ids"])


def test_split_documents_uses_content_addressed_chunk_ids_and_metadata():
    from services.rag_api.document.splitter import split_documents

    chunks = split_documents(
        [
            {
                "doc_id": "doc-abc",
                "document_id": "doc-abc",
                "version": "2",
                "publish_status": "published",
                "effective_at": "2026-07-01T00:00:00Z",
                "updated_at": "2026-07-02T00:00:00Z",
                "policy_ref": "policy-sales",
                "acl_revision": "7",
                "source_file": "a.txt",
                "source_path": "/tmp/a.txt",
                "content_hash": "hash-abc",
                "content": "客户准入材料要求\n必须提供营业执照。",
            }
        ],
        chunk_size=200,
        chunk_overlap=20,
    )

    assert chunks[0]["id"].startswith("doc-abc::chunk::")
    assert chunks[0]["id"].endswith("::001")
    assert chunks[0]["metadata"]["doc_id"] == "doc-abc"
    assert chunks[0]["metadata"]["document_id"] == "doc-abc"
    assert chunks[0]["metadata"]["content_hash"] == "hash-abc"
    assert chunks[0]["metadata"]["chunk_id"] == chunks[0]["id"]
    assert len(chunks[0]["metadata"]["chunk_hash"]) == 64
    assert chunks[0]["metadata"]["document_version"] == "2"
    assert chunks[0]["metadata"]["publish_status"] == "published"
    assert chunks[0]["metadata"]["effective_at"] == "2026-07-01T00:00:00Z"
    assert chunks[0]["metadata"]["policy_ref"] == "policy-sales"
    assert chunks[0]["metadata"]["acl_revision"] == "7"


def test_multi_vector_chunks_have_content_hashes_for_their_own_text():
    import hashlib
    import re

    from services.rag_api.document.multi_vector import expand_multi_vector_chunks

    base = {
        "id": "doc-a::chunk::base::001",
        "content": "第一段内容。\n第二段不同内容。",
        "metadata": {"document_id": "doc-a", "chunk_hash": "parent-hash"},
    }

    chunks = expand_multi_vector_chunks([base], RagSettings(multi_vector_enabled=True))

    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk["content"]).strip()
        expected_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        assert chunk["metadata"]["chunk_hash"] == expected_hash
        assert chunk["metadata"]["chunk_id"] == chunk["id"]


def test_tombstones_are_retained_for_thirty_days():
    from datetime import datetime, timezone

    from services.rag_api.document.ingest import _retained_tombstones

    tombstones = [
        {"document_id": "recent", "deleted_at": "2026-07-01T00:00:00Z"},
        {"document_id": "expired", "deleted_at": "2026-05-01T00:00:00Z"},
    ]

    retained = _retained_tombstones(tombstones, datetime(2026, 7, 11, tzinfo=timezone.utc))

    assert [item["document_id"] for item in retained] == ["recent"]
