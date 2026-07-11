from __future__ import annotations

from typing import Any


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.added: list[dict[str, Any]] = []
        self.upserted: list[dict[str, Any]] = []
        self.deleted: list[list[str]] = []

    def add(self, *, ids, documents, embeddings, metadatas) -> None:
        self.added.append(
            {
                "ids": list(ids),
                "documents": list(documents),
                "embeddings": list(embeddings),
                "metadatas": list(metadatas),
            }
        )

    def upsert(self, *, ids, documents, embeddings, metadatas) -> None:
        self.upserted.append(
            {
                "ids": list(ids),
                "documents": list(documents),
                "embeddings": list(embeddings),
                "metadatas": list(metadatas),
            }
        )

    def delete(self, *, ids) -> None:
        self.deleted.append(list(ids))

    def count(self) -> int:
        return sum(len(batch["ids"]) for batch in self.added + self.upserted)

    def query(self, *, query_embeddings, n_results, include):
        documents = [document for batch in self.added for document in batch["documents"]]
        metadatas = [metadata for batch in self.added for metadata in batch["metadatas"]]
        return {
            "documents": [documents[:n_results]],
            "metadatas": [metadatas[:n_results]],
            "distances": [[0.1 for _ in documents[:n_results]]],
        }


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.deleted: list[str] = []

    def get_or_create_collection(self, name: str, metadata=None):
        self.collections.setdefault(name, FakeCollection(name))
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        self.collections.pop(name, None)

    def get_max_batch_size(self) -> int:
        return 2


def test_index_graph_vectors_writes_entity_and_relationship_collections(monkeypatch):
    from services.rag_api.graph import graph_vector_store

    fake_client = FakeClient()
    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(graph_vector_store, "embed_texts", lambda texts: [[float(index)] for index, _ in enumerate(texts)])
    monkeypatch.setattr(graph_vector_store, "_base_collection_name", lambda: "crabrag_test")

    stats = graph_vector_store.index_graph_vectors(
        nodes=[
            {
                "id": "一渠一表",
                "label": "一渠一表",
                "type": "主题实体",
                "category": "项目材料",
                "source_files": ["a.docx"],
            },
            {
                "id": "合规审核",
                "label": "合规审核",
                "type": "知识分类",
                "category": "合规审核",
                "source_files": ["b.docx"],
            },
            {
                "id": "企业客户",
                "label": "企业客户",
                "type": "客户类型",
                "category": "客户准入",
                "source_files": ["c.docx"],
            },
        ],
        edges=[
            {
                "source": "一渠一表",
                "target": "合规审核",
                "label": "关联分类",
                "description": "一渠一表关联合规审核",
                "evidence": "一渠一表材料",
                "source_file": "a.docx",
                "confidence": 0.8,
            }
        ],
    )

    entity_collection = fake_client.collections["crabrag_test_graph_entities"]
    relationship_collection = fake_client.collections["crabrag_test_graph_relationships"]
    assert stats == {"graph_entity_index_count": 3, "graph_relationship_index_count": 1}
    assert [len(batch["ids"]) for batch in entity_collection.added] == [2, 1]
    assert relationship_collection.added[0]["metadatas"][0]["source"] == "一渠一表"
    assert relationship_collection.added[0]["metadatas"][0]["target"] == "合规审核"


def test_search_graph_vectors_returns_normalized_payloads(monkeypatch):
    from services.rag_api.graph import graph_vector_store

    fake_client = FakeClient()
    fake_client.collections["crabrag_test_graph_entities"] = FakeCollection("crabrag_test_graph_entities")
    fake_client.collections["crabrag_test_graph_entities"].add(
        ids=["entity::一渠一表"],
        documents=["一渠一表 主题实体"],
        embeddings=[[0.1]],
        metadatas=[{"id": "一渠一表", "label": "一渠一表", "source_files": "[\"a.docx\"]"}],
    )
    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(graph_vector_store, "embed_texts", lambda texts: [[0.1]])
    monkeypatch.setattr(graph_vector_store, "_base_collection_name", lambda: "crabrag_test")

    results = graph_vector_store.search_graph_entities("一渠一表", top_k=1)

    assert results[0]["id"] == "一渠一表"
    assert results[0]["source_files"] == ["a.docx"]
    assert results[0]["score"] == 0.9


def test_graph_generation_expands_aggregated_entity_per_source_document(monkeypatch):
    from services.rag_api.graph import graph_vector_store

    fake_client = FakeClient()
    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(graph_vector_store, "embed_texts", lambda texts: [[1.0] for _ in texts])
    monkeypatch.setattr(graph_vector_store, "_base_collection_name", lambda: "kb")
    monkeypatch.setattr(graph_vector_store.index_generation, "active_generation_id", lambda: None)

    stats = graph_vector_store.index_graph_vectors_generation(
        [{"id": "category", "label": "category", "document_ids": ["doc-a", "doc-b"], "source_files": ["a.txt", "b.txt"]}],
        [],
        "gen-1",
    )

    collection = fake_client.collections["kb__graph_entity__gen-1"]
    metadatas = [metadata for batch in collection.upserted for metadata in batch["metadatas"]]
    assert stats["graph_entity_index_count"] == 2
    assert {metadata["document_id"] for metadata in metadatas} == {"doc-a", "doc-b"}


def test_index_graph_vectors_incremental_upserts_only_changed_records(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_vector_store

    fake_client = FakeClient()
    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: fake_client)
    monkeypatch.setattr(graph_vector_store, "embed_texts", lambda texts: [[float(index)] for index, _ in enumerate(texts)])
    monkeypatch.setattr(graph_vector_store, "_base_collection_name", lambda: "crabrag_test")
    monkeypatch.setattr(graph_vector_store, "GRAPH_VECTOR_MANIFEST_PATH", tmp_path / "graph_vector_manifest.json")

    first = graph_vector_store.index_graph_vectors_incremental(
        nodes=[
            {"id": "一渠一表", "label": "一渠一表", "type": "主题实体", "source_files": ["a.docx"]},
            {"id": "项目材料", "label": "项目材料", "type": "知识分类", "source_files": ["a.docx"]},
        ],
        edges=[
            {
                "source": "一渠一表",
                "target": "项目材料",
                "label": "关联分类",
                "description": "一渠一表关联项目材料",
                "source_file": "a.docx",
            }
        ],
    )
    second = graph_vector_store.index_graph_vectors_incremental(
        nodes=[
            {"id": "一渠一表", "label": "一渠一表", "type": "主题实体", "source_files": ["a.docx"]},
            {"id": "项目材料", "label": "项目材料", "type": "知识分类", "source_files": ["a.docx"]},
        ],
        edges=[
            {
                "source": "一渠一表",
                "target": "项目材料",
                "label": "关联分类",
                "description": "一渠一表关联项目材料",
                "source_file": "a.docx",
            }
        ],
    )
    third = graph_vector_store.index_graph_vectors_incremental(
        nodes=[
            {"id": "一渠一表", "label": "一渠一表", "type": "主题实体", "source_files": ["a.docx", "b.docx"]},
        ],
        edges=[],
    )

    entity_collection = fake_client.collections["crabrag_test_graph_entities"]
    relationship_collection = fake_client.collections["crabrag_test_graph_relationships"]
    assert first == {"graph_entity_index_count": 2, "graph_relationship_index_count": 1}
    assert second == {"graph_entity_index_count": 2, "graph_relationship_index_count": 1}
    assert third == {"graph_entity_index_count": 1, "graph_relationship_index_count": 0}
    assert [batch["ids"] for batch in entity_collection.upserted] == [
        ["entity::一渠一表", "entity::项目材料"],
        ["entity::一渠一表"],
    ]
    assert relationship_collection.upserted[0]["ids"] == ["relationship::一渠一表->关联分类->项目材料"]
    assert entity_collection.deleted == [["entity::项目材料"]]
    assert relationship_collection.deleted == [["relationship::一渠一表->关联分类->项目材料"]]
