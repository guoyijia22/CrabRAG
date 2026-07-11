from __future__ import annotations

import pytest


def test_permission_provider_filters_public_and_role_documents():
    from services.rag_api.security import LocalPermissionProvider, PrincipalContext

    generation_manifest = {
        "documents": {
            "public-doc": {"acl": {"visibility": "public", "revision": "1"}},
            "sales-doc": {"acl": {"visibility": "restricted", "roles": ["sales"], "revision": "2"}},
            "finance-doc": {"acl": {"visibility": "restricted", "groups": ["finance"], "revision": "3"}},
        }
    }
    provider = LocalPermissionProvider()

    anonymous = provider.allowed_document_ids(PrincipalContext.anonymous(), generation_manifest)
    sales = provider.allowed_document_ids(
        PrincipalContext(subject="u-1", roles=("sales",), groups=(), permission_revision="7"),
        generation_manifest,
    )

    assert anonymous == frozenset({"public-doc"})
    assert sales == frozenset({"public-doc", "sales-doc"})


def test_retrieval_cache_key_changes_with_generation_and_permission_revision():
    from services.rag_api.retrieval.cache import retrieval_cache_key
    from services.rag_api.security import PrincipalContext, RetrievalContext

    first = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext(subject="u-1", roles=("sales",), groups=(), permission_revision="1"),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-1",
    )
    second = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext(subject="u-1", roles=("sales",), groups=(), permission_revision="2"),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-2",
    )

    assert retrieval_cache_key(first, {"query": "price"}) != retrieval_cache_key(second, {"query": "price"})


def test_conversation_memory_isolated_by_subject_and_generation():
    from services.rag_api.memory import conversation_memory

    conversation_memory.SESSION_MEMORY.clear()
    conversation_memory.update_memory("session", "old question", "old answer", "", [], subject="u-1", generation_id="gen-1")

    assert conversation_memory.get_history("session", subject="u-1", generation_id="gen-1")
    assert conversation_memory.get_history("session", subject="u-1", generation_id="gen-2") == []
    assert conversation_memory.get_history("session", subject="u-2", generation_id="gen-1") == []


def test_conversation_memory_isolated_by_permission_revision():
    from services.rag_api.memory import conversation_memory

    conversation_memory.SESSION_MEMORY.clear()
    conversation_memory.update_memory(
        "session",
        "secret question",
        "secret answer",
        "",
        [],
        subject="u-1",
        generation_id="gen-1",
        permission_fingerprint="permission-v1",
    )

    assert conversation_memory.get_history(
        "session", subject="u-1", generation_id="gen-1", permission_fingerprint="permission-v1"
    )
    assert conversation_memory.get_history(
        "session", subject="u-1", generation_id="gen-1", permission_fingerprint="permission-v2"
    ) == []


def test_conversation_memory_uses_sliding_ttl(monkeypatch):
    from services.rag_api.memory import conversation_memory

    now = {"value": 0.0}
    monkeypatch.setattr(conversation_memory, "_time", lambda: now["value"])
    monkeypatch.setattr(conversation_memory, "SESSION_TTL_SECONDS", 1800)
    conversation_memory.clear_memory()
    conversation_memory.update_memory("session", "question", "answer", "", [])

    now["value"] = 1700.0
    assert conversation_memory.get_history("session")
    now["value"] = 3400.0
    assert conversation_memory.get_history("session")
    now["value"] = 5201.0
    assert conversation_memory.get_history("session") == []


def test_conversation_memory_evicts_least_recently_used_session(monkeypatch):
    from services.rag_api.memory import conversation_memory

    now = {"value": 0.0}
    monkeypatch.setattr(conversation_memory, "_time", lambda: now["value"])
    monkeypatch.setattr(conversation_memory, "MAX_SESSION_ENTRIES", 2)
    conversation_memory.clear_memory()
    conversation_memory.update_memory("session-a", "a", "answer-a", "", [])
    now["value"] = 1.0
    conversation_memory.update_memory("session-b", "b", "answer-b", "", [])
    now["value"] = 2.0
    assert conversation_memory.get_history("session-a")
    now["value"] = 3.0
    conversation_memory.update_memory("session-c", "c", "answer-c", "", [])

    assert conversation_memory.get_history("session-b") == []
    assert conversation_memory.get_history("session-a")
    assert conversation_memory.get_history("session-c")
    assert len(conversation_memory.SESSION_MEMORY) == 2


def test_vector_search_applies_allowed_document_filter_before_query(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context
    from services.rag_api.vector import chroma_store

    captured = {}

    class Collection:
        def count(self):
            return 1

        def query(self, **kwargs):
            captured.update(kwargs)
            return {
                "documents": [["restricted content"]],
                "metadatas": [[{
                    "document_id": "doc-a",
                    "document_version": "2",
                    "chunk_id": "doc-a::chunk::hash::001",
                    "parent_chunk_id": "doc-a::parent",
                    "publish_status": "published",
                    "effective_at": "2026-01-01T00:00:00Z",
                }]],
                "distances": [[0.1]],
            }

    monkeypatch.setattr(chroma_store, "get_collection", lambda: Collection())
    monkeypatch.setattr(chroma_store, "embed_texts", lambda texts: [[1.0]])
    context = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext(subject="u-1", roles=("sales",), groups=(), permission_revision="2"),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-2",
    )

    with use_retrieval_context(context):
        results = chroma_store.search_chunks("query", "", top_k=1, min_score=0.0, candidate_k=1)

    assert captured["where"] == {"document_id": {"$in": ["doc-a"]}}
    assert results[0]["document_id"] == "doc-a"
    assert results[0]["document_version"] == "2"
    assert results[0]["chunk_id"] == "doc-a::chunk::hash::001"
    assert results[0]["parent_chunk_id"] == "doc-a::parent"
    assert results[0]["publish_status"] == "published"
    assert results[0]["index_generation"] == "gen-2"


def test_vector_search_returns_empty_without_query_when_no_documents_allowed(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context
    from services.rag_api.vector import chroma_store

    class Collection:
        def count(self):
            return 1

        def query(self, **kwargs):
            raise AssertionError("query must not run")

    monkeypatch.setattr(chroma_store, "get_collection", lambda: Collection())
    context = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset(),
        permission_fingerprint="anonymous",
    )

    with use_retrieval_context(context):
        assert chroma_store.search_chunks("query", "", top_k=1) == []


def test_graph_vector_search_applies_same_document_filter(monkeypatch):
    from services.rag_api.graph import graph_vector_store
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    captured = {}

    class Collection:
        def count(self):
            return 1

        def query(self, **kwargs):
            captured.update(kwargs)
            return {"documents": [["entity"]], "metadatas": [[{"id": "entity", "document_id": "doc-a"}]], "distances": [[0.1]]}

    class Client:
        def get_collection(self, name):
            return Collection()

    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: Client())
    monkeypatch.setattr(graph_vector_store, "embed_texts", lambda texts: [[1.0]])
    context = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="anonymous",
    )

    with use_retrieval_context(context):
        results = graph_vector_store.search_graph_entities("query", top_k=1)

    assert results
    assert captured["where"] == {"document_id": {"$in": ["doc-a"]}}


def test_graph_api_filters_nodes_edges_and_source_files_by_permission(monkeypatch):
    from services.rag_api.graph import graph_api
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    nodes = [
        {
            "id": "shared",
            "label": "shared",
            "document_ids": ["doc-a", "doc-b"],
            "document_sources": [
                {"document_id": "doc-a", "source_file": "a.txt"},
                {"document_id": "doc-b", "source_file": "b.txt"},
            ],
            "source_files": ["a.txt", "b.txt"],
        },
        {"id": "private", "label": "private", "document_ids": ["doc-b"], "source_files": ["b.txt"]},
    ]
    edges = [
        {"source": "shared", "target": "rule-a", "label": "contains", "document_id": "doc-a", "source_file": "a.txt"},
        {"source": "private", "target": "rule-b", "label": "contains", "document_id": "doc-b", "source_file": "b.txt"},
    ]
    monkeypatch.setattr(graph_api, "load_raw_graph", lambda path=None: (nodes, edges, "dynamic_graph"))
    context = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="anonymous",
    )

    with use_retrieval_context(context):
        payload = graph_api.build_graph_payload()

    assert [node["id"] for node in payload["nodes"]] == ["shared"]
    assert payload["nodes"][0]["properties"]["source_files"] == ["a.txt"]
    assert [edge["properties"]["source_file"] for edge in payload["edges"]] == ["a.txt"]


def test_categories_are_recomputed_from_allowed_documents(tmp_path, monkeypatch):
    from services.rag_api.document import categories
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    path = tmp_path / "categories.json"
    monkeypatch.setattr(categories, "KB_CATEGORIES_PATH", path)
    monkeypatch.setattr(categories, "active_artifact_path", lambda name, fallback: fallback, raising=False)
    documents = [
        {"document_id": "doc-a", "source_file": "客户准入.txt", "content": "营业执照"},
        {"document_id": "doc-b", "source_file": "资费咨询.txt", "content": "套餐价格"},
    ]
    chunks = [
        {"metadata": {"document_id": "doc-a", "source_file": "客户准入.txt", "category": "客户准入"}},
        {"metadata": {"document_id": "doc-b", "source_file": "资费咨询.txt", "category": "资费咨询"}},
    ]
    categories.save_kb_categories(documents, chunks)
    context = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="anonymous",
    )

    with use_retrieval_context(context):
        payload = categories.load_kb_categories()

    assert payload["categories"] == ["客户准入"]
    assert payload["items"][0]["source_files"] == ["客户准入.txt"]
    assert payload["items"][0]["document_count"] == 1


def test_dispatch_retrieval_caches_by_generation_and_permission(monkeypatch):
    from services.rag_api.agent import tools as agent_tools
    from services.rag_api.retrieval.cache import RETRIEVAL_CACHE
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    RETRIEVAL_CACHE.clear()
    calls = []

    def vector_search(*args, **kwargs):
        calls.append(args)
        return {"mode": "vector", "chunks": [{"document_id": "doc-a", "content": "answer"}], "relation_paths": [], "error": None, "trace": []}

    monkeypatch.setattr(agent_tools, "vector_rule_search", vector_search)
    context = RetrievalContext(
        generation_id="gen-2",
        principal=PrincipalContext(subject="u-1", roles=("sales",), groups=(), permission_revision="2"),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-2",
    )

    with use_retrieval_context(context):
        first = agent_tools.dispatch_retrieval("query", "intent", [], "vector_rule_search")
        second = agent_tools.dispatch_retrieval("query", "intent", [], "vector_rule_search")

    assert first == second
    assert len(calls) == 1


def test_identity_headers_are_ignored_without_matching_internal_token():
    from services.rag_api.security import principal_from_headers

    headers = {
        "x-crabrag-internal-token": "wrong",
        "x-crabrag-subject": "attacker",
        "x-crabrag-roles": "admin,sales",
        "x-crabrag-admin": "true",
    }

    principal = principal_from_headers(headers, internal_token="trusted")

    assert principal.subject == "anonymous"
    assert principal.roles == ()
    assert principal.can_manage_index is False


def test_identity_headers_create_principal_only_for_trusted_gateway():
    from services.rag_api.security import principal_from_headers

    headers = {
        "x-crabrag-internal-token": "trusted",
        "x-crabrag-subject": "user-1",
        "x-crabrag-roles": "sales, reviewer",
        "x-crabrag-groups": "north",
        "x-crabrag-permission-revision": "42",
        "x-crabrag-admin": "true",
    }

    principal = principal_from_headers(headers, internal_token="trusted")

    assert principal.subject == "user-1"
    assert principal.roles == ("reviewer", "sales")
    assert principal.groups == ("north",)
    assert principal.permission_revision == "42"
    assert principal.can_manage_index is True


def test_chat_binds_trusted_principal_generation_and_memory(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main
    from services.rag_api.memory import conversation_memory
    from services.rag_api.security import RetrievalContext, current_retrieval_context

    conversation_memory.SESSION_MEMORY.clear()
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    captured = {}

    def build_context(principal):
        captured["principal"] = principal
        return RetrievalContext(
            generation_id="gen-2",
            principal=principal,
            allowed_document_ids=frozenset({"doc-a"}),
            permission_fingerprint="permission-2",
        )

    def run_qa(state):
        captured["context"] = current_retrieval_context()
        captured["history"] = state["history"]
        return {
            **state,
            "answer": "answer",
            "references": [{"document_id": "doc-a"}, {"document_id": "doc-b"}],
            "relation_paths": [{"document_id": "doc-a", "path": "a"}, {"document_id": "doc-b", "path": "b"}],
            "trace": [],
        }

    monkeypatch.setattr(main, "build_retrieval_context", build_context, raising=False)
    monkeypatch.setattr(main, "run_qa", run_qa)
    monkeypatch.setattr(main, "append_qa_log", lambda payload: None)
    client = TestClient(main.app)

    response = client.post(
        "/api/chat",
        json={"session_id": "session", "question": "question"},
        headers={
            "x-crabrag-internal-token": "trusted",
            "x-crabrag-subject": "user-1",
            "x-crabrag-roles": "sales",
            "x-crabrag-permission-revision": "9",
        },
    )

    assert response.status_code == 200
    assert response.json()["index_generation"] == "gen-2"
    assert captured["principal"].subject == "user-1"
    assert captured["context"].generation_id == "gen-2"
    assert conversation_memory.get_history(
        "session",
        subject="user-1",
        generation_id="gen-2",
        permission_fingerprint="permission-2",
    )
    assert response.json()["references"] == [{"document_id": "doc-a"}]
    assert response.json()["relation_paths"] == [{"document_id": "doc-a", "path": "a"}]


def test_dynamic_graph_search_uses_active_generation_artifact_by_default(monkeypatch):
    from services.rag_api.graph import graph_search

    captured = {}

    def load(path=None):
        captured["path"] = path
        return [], [], "empty_graph"

    monkeypatch.setattr(graph_search, "load_raw_graph", load)

    assert graph_search._dynamic_graph_relation_search("query", "", 1) is None
    assert captured["path"] is None


def test_dynamic_relation_path_preserves_document_identity():
    from services.rag_api.graph.graph_search import _format_dynamic_relation

    result = _format_dynamic_relation(
        {
            "source": "a",
            "target": "b",
            "label": "contains",
            "document_id": "doc-a",
            "source_file": "a.txt",
        },
        0.9,
        match_source="test",
    )

    assert result["document_id"] == "doc-a"


def test_request_context_pins_vector_collection_when_active_pointer_changes(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context
    from services.rag_api.vector import chroma_store

    monkeypatch.setattr(chroma_store.index_generation, "active_generation_id", lambda: "gen-2")
    monkeypatch.setattr(chroma_store, "get_settings", lambda: type("Settings", (), {"collection_name": "kb"})())
    context = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="anonymous",
    )

    with use_retrieval_context(context):
        collection_name = chroma_store._collection_name()

    assert collection_name == "kb__text__gen-1"


def test_request_context_pins_category_artifact_when_active_pointer_changes(tmp_path, monkeypatch):
    import json

    from services.rag_api import index_generation
    from services.rag_api.document import categories
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    root = tmp_path / "index"
    monkeypatch.setattr(index_generation, "INDEX_ROOT", root)
    monkeypatch.setattr(index_generation, "ACTIVE_INDEX_PATH", root / "active.json")
    monkeypatch.setattr(index_generation, "GENERATIONS_DIR", root / "generations")
    for generation_id, name in (("gen-1", "old"), ("gen-2", "new")):
        index_generation.generation_artifact_path(generation_id, "categories.json").write_text(
            json.dumps({"items": [{"name": name}], "categories": [name]}),
            encoding="utf-8",
        )
        index_generation.publish_generation(generation_id, {"permission_schema_version": 1})
    context = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=None,
        permission_fingerprint="anonymous",
    )

    with use_retrieval_context(context):
        payload = categories.load_kb_categories()

    assert payload["categories"] == ["old"]


def test_categories_endpoint_runs_inside_trusted_retrieval_context(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main
    from services.rag_api.security import RetrievalContext, current_retrieval_context

    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")

    def build_context(principal):
        return RetrievalContext(
            generation_id="gen-2",
            principal=principal,
            allowed_document_ids=frozenset({"doc-a"}),
            permission_fingerprint="permission-2",
        )

    monkeypatch.setattr(main, "build_retrieval_context", build_context)
    monkeypatch.setattr(
        main,
        "load_kb_categories",
        lambda: {
            "generation": current_retrieval_context().generation_id if current_retrieval_context() else None,
            "allowed": sorted(current_retrieval_context().allowed_document_ids) if current_retrieval_context() else [],
        },
    )

    response = TestClient(main.app).get(
        "/api/categories",
        headers={"x-crabrag-internal-token": "trusted", "x-crabrag-subject": "user-1"},
    )

    assert response.json() == {"generation": "gen-2", "allowed": ["doc-a"]}


def test_graph_schema_endpoints_require_trusted_admin(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main

    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    monkeypatch.setattr(main, "load_graph_schema", lambda: {"kind": "schema"})
    monkeypatch.setattr(main, "load_graph_schema_suggestion", lambda: {"kind": "suggestion"})
    monkeypatch.setattr(main, "save_graph_schema_config", lambda payload: {"kind": "saved", **payload})
    client = TestClient(main.app)

    requests = [
        ("get", "/api/graph/schema", None),
        ("get", "/api/graph/schema/suggestion", None),
        ("put", "/api/graph/schema", {"version": 1}),
    ]
    for method, path, payload in requests:
        response = getattr(client, method)(path, json=payload) if payload is not None else getattr(client, method)(path)
        assert response.status_code == 403

    admin_headers = {
        "x-crabrag-internal-token": "trusted",
        "x-crabrag-subject": "admin",
        "x-crabrag-admin": "true",
    }
    for method, path, payload in requests:
        response = (
            getattr(client, method)(path, headers=admin_headers, json=payload)
            if payload is not None
            else getattr(client, method)(path, headers=admin_headers)
        )
        assert response.status_code == 200


def test_settings_endpoints_require_trusted_admin(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main, model_api_settings

    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    monkeypatch.setattr(model_api_settings, "MODEL_API_SETTINGS_PATH", tmp_path / "model-settings.json")
    client = TestClient(main.app)

    requests = [
        ("get", "/api/app-settings", None),
        ("get", "/api/model-settings", None),
        ("get", "/api/settings", None),
        ("put", "/api/config", {"system_name": "Blocked"}),
        ("put", "/api/app-settings", {}),
        ("put", "/api/model-settings", {"use_local_models": False}),
        ("put", "/api/settings", {}),
    ]
    for method, path, payload in requests:
        response = (
            getattr(client, method)(path, json=payload)
            if payload is not None
            else getattr(client, method)(path)
        )
        assert response.status_code == 403

    admin_headers = {
        "x-crabrag-internal-token": "trusted",
        "x-crabrag-subject": "admin",
        "x-crabrag-admin": "true",
    }
    for path in ("/api/app-settings", "/api/model-settings", "/api/settings"):
        assert client.get(path, headers=admin_headers).status_code == 200


def test_model_settings_returns_controlled_error_when_secure_storage_write_fails(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main
    from services.rag_api.secret_store import SecretStorageError

    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "trusted")
    monkeypatch.setattr(
        main,
        "update_model_api_settings",
        lambda settings: (_ for _ in ()).throw(SecretStorageError("secure store unavailable")),
    )
    response = TestClient(main.app).put(
        "/api/model-settings",
        headers={
            "x-crabrag-internal-token": "trusted",
            "x-crabrag-subject": "admin",
            "x-crabrag-admin": "true",
        },
        json={"use_local_models": False},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "secure store unavailable"


def test_governed_text_collection_read_never_creates_missing_collection(monkeypatch):
    from services.rag_api import exceptions
    from services.rag_api.vector import chroma_store

    calls = {"get": 0, "create": 0}

    class Client:
        def get_collection(self, name):
            calls["get"] += 1
            raise KeyError(name)

        def get_or_create_collection(self, name, metadata=None):
            calls["create"] += 1
            return object()

    monkeypatch.setattr(chroma_store, "_get_chroma_client", lambda: Client())
    monkeypatch.setattr(chroma_store, "get_settings", lambda: type("Settings", (), {"collection_name": "kb"})())
    monkeypatch.setattr(chroma_store.index_generation, "active_generation_id", lambda: "gen-1")
    error_type = getattr(exceptions, "IndexCollectionUnavailable", RuntimeError)

    with pytest.raises(error_type, match="kb__text__gen-1"):
        chroma_store.get_collection()

    assert calls == {"get": 1, "create": 0}


def test_legacy_text_collection_read_can_initialize_collection(monkeypatch):
    from services.rag_api.vector import chroma_store

    expected = object()
    calls = {"create": 0}

    class Client:
        def get_or_create_collection(self, name, metadata=None):
            calls["create"] += 1
            return expected

    monkeypatch.setattr(chroma_store, "_get_chroma_client", lambda: Client())
    monkeypatch.setattr(chroma_store, "get_settings", lambda: type("Settings", (), {"collection_name": "kb"})())
    monkeypatch.setattr(chroma_store.index_generation, "active_generation_id", lambda: None)

    assert chroma_store.get_collection() is expected
    assert calls["create"] == 1


def test_governed_graph_collection_read_never_creates_missing_collection(monkeypatch):
    from services.rag_api import exceptions
    from services.rag_api.graph import graph_vector_store

    calls = {"get": 0, "create": 0}

    class Client:
        def get_collection(self, name):
            calls["get"] += 1
            raise KeyError(name)

        def get_or_create_collection(self, name, metadata=None):
            calls["create"] += 1
            return type("Collection", (), {"count": lambda self: 0})()

    monkeypatch.setattr(graph_vector_store, "_get_chroma_client", lambda: Client())
    monkeypatch.setattr(graph_vector_store, "get_settings", lambda: type("Settings", (), {"collection_name": "kb"})())
    monkeypatch.setattr(graph_vector_store.index_generation, "active_generation_id", lambda: "gen-1")
    error_type = getattr(exceptions, "IndexCollectionUnavailable", RuntimeError)

    with pytest.raises(error_type, match="kb__graph_entity__gen-1"):
        graph_vector_store.search_graph_entities("query", top_k=1)

    assert calls == {"get": 1, "create": 0}


def test_graph_collection_failure_is_not_hidden_by_literal_fallback(monkeypatch):
    from services.rag_api import exceptions
    from services.rag_api.graph import graph_search

    error_type = getattr(exceptions, "IndexCollectionUnavailable", RuntimeError)

    def fail(*args, **kwargs):
        raise error_type("missing graph collection")

    monkeypatch.setattr(graph_search, "search_graph_entities", fail)

    with pytest.raises(error_type, match="missing graph collection"):
        graph_search._dynamic_graph_vector_search(
            "query",
            "intent",
            [{"id": "a"}, {"id": "b"}],
            [{"source": "a", "target": "b", "label": "rel"}],
            {"entity_keywords": ["a"], "relationship_keywords": ["rel"]},
            1,
        )


def test_vector_tool_propagates_missing_governed_collection(monkeypatch):
    from services.rag_api.agent import tools
    from services.rag_api.exceptions import IndexCollectionUnavailable

    def fail(*args, **kwargs):
        raise IndexCollectionUnavailable("missing text collection")

    monkeypatch.setattr(tools, "search_chunks", fail)

    with pytest.raises(IndexCollectionUnavailable, match="missing text collection"):
        tools.vector_rule_search(
            "query",
            "intent",
            [],
            allow_query_expansion=False,
            allow_rerank=False,
        )


def test_governed_text_fallback_propagates_missing_collection(monkeypatch):
    from services.rag_api.exceptions import IndexCollectionUnavailable
    from services.rag_api.graph import graph_search
    from services.rag_api.vector import chroma_store

    def fail(*args, **kwargs):
        raise IndexCollectionUnavailable("missing text collection")

    monkeypatch.setattr(chroma_store, "search_chunks_by_keywords", fail)

    with pytest.raises(IndexCollectionUnavailable, match="missing text collection"):
        graph_search._governed_text_fallback("query", "intent", 1)


def test_hybrid_retrieval_keeps_successful_channel_and_reports_collection_error(monkeypatch):
    from services.rag_api.agent import tools
    from services.rag_api.exceptions import IndexCollectionUnavailable

    def fail_graph(*args, **kwargs):
        raise IndexCollectionUnavailable("missing graph collection")

    monkeypatch.setattr(tools, "graph_relation_search_tool", fail_graph)
    monkeypatch.setattr(
        tools,
        "vector_rule_search",
        lambda *args, **kwargs: {
            "mode": "vector",
            "chunks": [{"document_id": "doc-a", "content": "allowed", "source_file": "a.txt"}],
            "relation_paths": [],
            "error": None,
            "trace": [],
        },
    )
    monkeypatch.setattr(tools, "search_all_chunks", lambda: [])

    result = tools.dispatch_retrieval("query", "intent", [], "hybrid_search", allow_rerank=False)

    assert result["chunks"][0]["document_id"] == "doc-a"
    assert "missing graph collection" in result["error"]


def test_chat_returns_503_when_governed_collection_is_unavailable(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import main
    from services.rag_api.exceptions import IndexCollectionUnavailable
    from services.rag_api.security import PrincipalContext, RetrievalContext

    principal = PrincipalContext.anonymous()
    context = RetrievalContext(
        generation_id="gen-1",
        principal=principal,
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="anonymous",
    )
    monkeypatch.setattr(main, "_request_retrieval_context", lambda request: (principal, context))
    monkeypatch.setattr(
        main,
        "run_qa",
        lambda state: (_ for _ in ()).throw(IndexCollectionUnavailable("missing text collection")),
    )

    response = TestClient(main.app, raise_server_exceptions=False).post(
        "/api/chat",
        json={"question": "query"},
    )

    assert response.status_code == 503
    assert "missing text collection" in response.json()["detail"]
