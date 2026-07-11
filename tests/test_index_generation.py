from __future__ import annotations

import json
from pathlib import Path

import pytest


def _configure_generation_paths(tmp_path: Path, monkeypatch):
    from services.rag_api import index_generation

    root = tmp_path / "index"
    monkeypatch.setattr(index_generation, "INDEX_ROOT", root)
    monkeypatch.setattr(index_generation, "ACTIVE_INDEX_PATH", root / "active.json")
    monkeypatch.setattr(index_generation, "GENERATIONS_DIR", root / "generations")
    return index_generation


def test_publish_generation_atomically_moves_active_to_previous(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)

    index_generation.publish_generation("gen-1", {"permission_schema_version": 1, "stats": {"chunk_count": 2}})
    index_generation.publish_generation("gen-2", {"permission_schema_version": 1, "stats": {"chunk_count": 3}})

    state = index_generation.load_index_state()
    assert state["active_generation"] == "gen-2"
    assert state["previous_generation"] == "gen-1"
    assert json.loads((tmp_path / "index" / "active.json").read_text(encoding="utf-8")) == state
    assert index_generation.generation_artifact_path("gen-2", "categories.json").parent.is_dir()


def test_rollback_swaps_current_and_previous_compatible_generations(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    index_generation.publish_generation("gen-1", {"permission_schema_version": 1})
    index_generation.publish_generation("gen-2", {"permission_schema_version": 1})

    state = index_generation.rollback_generation()

    assert state["active_generation"] == "gen-1"
    assert state["previous_generation"] == "gen-2"


def test_rollback_rejects_incompatible_previous_generation(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    index_generation.publish_generation("legacy", {"permission_schema_version": 0})
    index_generation.publish_generation("gen-2", {"permission_schema_version": 1})

    with pytest.raises(ValueError, match="不兼容"):
        index_generation.rollback_generation()


def test_corrupt_active_pointer_fails_closed_after_governance_initialized(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    index_generation.publish_generation("gen-1", {"permission_schema_version": 1})
    index_generation.ACTIVE_INDEX_PATH.write_text("{broken", encoding="utf-8")

    with pytest.raises(index_generation.IndexStateError, match="拒绝降级"):
        index_generation.active_generation_id()


def test_build_generation_reuses_matching_embedding_and_only_embeds_changed_chunk(tmp_path, monkeypatch):
    from services.rag_api.vector import chroma_store

    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    fake_client = _FakeClient()
    settings = type("Settings", (), {"collection_name": "kb", "embedding_provider": "api", "chroma_dir": tmp_path / "chroma"})()
    monkeypatch.setattr(chroma_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(chroma_store, "get_settings", lambda: settings)
    embedded_batches = []
    monkeypatch.setattr(
        chroma_store,
        "embed_texts",
        lambda texts: embedded_batches.append(list(texts)) or [[9.0, 9.0] for _ in texts],
    )
    index_generation.publish_generation("gen-1", {"permission_schema_version": 1})
    source = fake_client.get_or_create_collection("kb__text__gen-1")
    source.upsert(
        ids=["old-a"],
        documents=["unchanged"],
        embeddings=[[1.0, 2.0]],
        metadatas=[{"chunk_hash": "hash-a", "granularity": "chunk", "embedding_fingerprint": "embed-v1"}],
    )
    chunks = [
        {
            "id": "new-a",
            "content": "unchanged",
            "metadata": {"chunk_hash": "hash-a", "granularity": "chunk", "embedding_fingerprint": "embed-v1"},
        },
        {
            "id": "new-b",
            "content": "changed",
            "metadata": {"chunk_hash": "hash-b", "granularity": "chunk", "embedding_fingerprint": "embed-v1"},
        },
    ]

    stats = chroma_store.build_generation_chunks(chunks, "gen-2")

    assert stats == {"chunk_count": 2, "reused_embedding_count": 1, "embedded_chunk_count": 1}
    assert embedded_batches == [["changed"]]
    target = fake_client.collections["kb__text__gen-2"]
    assert target.records["new-a"]["embedding"] == [1.0, 2.0]
    assert target.records["new-b"]["embedding"] == [9.0, 9.0]
    assert target.records["new-a"]["metadata"]["generation_id"] == "gen-2"


def test_build_generation_does_not_reuse_unknown_fingerprint_and_embeds_duplicate_text_once(tmp_path, monkeypatch):
    from services.rag_api.vector import chroma_store

    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    fake_client = _FakeClient()
    settings = type("Settings", (), {"collection_name": "kb", "embedding_provider": "api", "chroma_dir": tmp_path / "chroma"})()
    monkeypatch.setattr(chroma_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(chroma_store, "get_settings", lambda: settings)
    embedded_batches = []
    monkeypatch.setattr(
        chroma_store,
        "embed_texts",
        lambda texts: embedded_batches.append(list(texts)) or [[9.0, 9.0] for _ in texts],
    )
    index_generation.publish_generation("gen-1", {"permission_schema_version": 1})
    source = fake_client.get_or_create_collection("kb__text__gen-1")
    source.upsert(
        ids=["legacy"],
        documents=["same"],
        embeddings=[[1.0, 2.0]],
        metadatas=[{"chunk_hash": "hash-a", "granularity": "chunk"}],
    )
    chunks = [
        {"id": "doc-a", "content": "same", "metadata": {"chunk_hash": "hash-a", "granularity": "chunk", "embedding_fingerprint": "embed-v2"}},
        {"id": "doc-b", "content": "same", "metadata": {"chunk_hash": "hash-a", "granularity": "chunk", "embedding_fingerprint": "embed-v2"}},
    ]

    stats = chroma_store.build_generation_chunks(chunks, "gen-2")

    assert stats == {"chunk_count": 2, "reused_embedding_count": 1, "embedded_chunk_count": 1}
    assert embedded_batches == [["same"]]


def test_graph_generation_reuses_unchanged_entity_and_relationship_embeddings(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_vector_store

    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    fake_client = _FakeClient()
    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(graph_vector_store, "_base_collection_name", lambda: "kb")
    embedded_batches = []
    monkeypatch.setattr(
        graph_vector_store,
        "embed_texts",
        lambda texts: embedded_batches.append(list(texts)) or [[3.0, 4.0] for _ in texts],
    )
    nodes = [{"id": "policy", "label": "policy", "type": "topic", "source_files": ["a.txt"]}]
    edges = [{"source": "policy", "target": "rule", "label": "contains", "source_file": "a.txt"}]

    graph_vector_store.index_graph_vectors_generation(nodes, edges, "gen-1")
    index_generation.publish_generation("gen-1", {"permission_schema_version": 1})
    embedded_batches.clear()
    stats = graph_vector_store.index_graph_vectors_generation(nodes, edges, "gen-2")

    assert stats["graph_reused_embedding_count"] == 2
    assert stats["graph_embedded_record_count"] == 0
    assert embedded_batches == []

    full_stats = graph_vector_store.index_graph_vectors_generation(nodes, edges, "gen-3", full_rebuild=True)
    assert full_stats["graph_reused_embedding_count"] == 0
    assert full_stats["graph_embedded_record_count"] == 2


def test_graph_generation_reembeds_when_embedding_fingerprint_changes(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_vector_store

    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    fake_client = _FakeClient()
    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(graph_vector_store, "_base_collection_name", lambda: "kb")
    fingerprint = {"value": "embed-v1"}
    monkeypatch.setattr(
        graph_vector_store.doc_status,
        "embedding_fingerprint",
        lambda settings: fingerprint["value"],
    )
    embedded_batches = []
    monkeypatch.setattr(
        graph_vector_store,
        "embed_texts",
        lambda texts: embedded_batches.append(list(texts)) or [[3.0, 4.0] for _ in texts],
    )
    nodes = [{"id": "policy", "label": "policy", "type": "topic", "source_files": ["a.txt"]}]
    edges = [{"source": "policy", "target": "rule", "label": "contains", "source_file": "a.txt"}]

    graph_vector_store.index_graph_vectors_generation(nodes, edges, "gen-1")
    index_generation.publish_generation("gen-1", {"permission_schema_version": 1})
    embedded_batches.clear()
    fingerprint["value"] = "embed-v2"
    stats = graph_vector_store.index_graph_vectors_generation(nodes, edges, "gen-2")

    assert stats["graph_reused_embedding_count"] == 0
    assert stats["graph_embedded_record_count"] == 2
    assert sum(len(batch) for batch in embedded_batches) == 2


def test_cleanup_deletes_only_generations_older_than_current_and_previous(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    for generation_id in ("gen-0", "gen-1", "gen-2"):
        index_generation.record_generation_resources(generation_id, "kb")
        index_generation.publish_generation(generation_id, {"permission_schema_version": 1})
    deleted_collections = []

    class Client:
        def delete_collection(self, name):
            deleted_collections.append(name)

    result = index_generation.cleanup_generations("kb", Client())

    assert result["deleted_generations"] == ["gen-0"]
    assert not (index_generation.GENERATIONS_DIR / "gen-0").exists()
    assert (index_generation.GENERATIONS_DIR / "gen-1").exists()
    assert (index_generation.GENERATIONS_DIR / "gen-2").exists()
    assert deleted_collections == [
        "kb__text__gen-0",
        "kb__graph_entity__gen-0",
        "kb__graph_relationship__gen-0",
    ]


def test_cleanup_keeps_generation_pinned_by_inflight_request(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    for generation_id in ("gen-0", "gen-1", "gen-2"):
        index_generation.record_generation_resources(generation_id, "kb")
        index_generation.publish_generation(generation_id, {"permission_schema_version": 1})

    class Client:
        def delete_collection(self, name):
            pass

    with index_generation.pin_generation("gen-0"):
        result = index_generation.cleanup_generations("kb", Client())

    assert result["deleted_generations"] == []
    assert (index_generation.GENERATIONS_DIR / "gen-0").exists()


def test_generation_build_lock_rejects_second_process(tmp_path, monkeypatch):
    index_generation = _configure_generation_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(index_generation, "INDEX_LOCK_PATH", tmp_path / "index" / "build.lock")

    with index_generation.generation_build_lock():
        with pytest.raises(RuntimeError, match="正在运行"):
            with index_generation.generation_build_lock():
                pass

    assert not index_generation.INDEX_LOCK_PATH.exists()


class _FakeCollection:
    def __init__(self) -> None:
        self.records = {}

    def upsert(self, *, ids, documents, embeddings, metadatas):
        for item_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.records[item_id] = {
                "document": document,
                "embedding": list(embedding),
                "metadata": dict(metadata),
            }

    add = upsert

    def get(self, *, include):
        return {
            "ids": list(self.records),
            "documents": [item["document"] for item in self.records.values()],
            "embeddings": [item["embedding"] for item in self.records.values()],
            "metadatas": [item["metadata"] for item in self.records.values()],
        }

    def count(self):
        return len(self.records)


class _FakeClient:
    def __init__(self) -> None:
        self.collections = {}

    def get_or_create_collection(self, name, metadata=None):
        self.collections.setdefault(name, _FakeCollection())
        return self.collections[name]

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def get_max_batch_size(self):
        return 100
