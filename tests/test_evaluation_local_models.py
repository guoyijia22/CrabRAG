from __future__ import annotations

from pathlib import Path

from services.rag_api import config
from services.rag_api.evaluation import questions, runner, tasks
from services.rag_api.llm import local_qwen_llm, siliconflow_client
from services.rag_api.rag_settings import RagSettings


def test_evaluation_question_generation_local_llm_fallback_does_not_use_remote_client(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: config.Settings(use_local_models=True, api_key=None, local_llm_model_dir=tmp_path),
    )
    monkeypatch.setattr(
        siliconflow_client,
        "get_chat_client",
        lambda: (_ for _ in ()).throw(AssertionError("evaluation question generation must not use remote chat client")),
    )
    monkeypatch.setattr(local_qwen_llm, "chat_completion_local", lambda *args, **kwargs: "不是 JSON")
    monkeypatch.setattr(
        questions,
        "load_kb_categories",
        lambda: {"items": [{"name": "客户准入", "source_files": ["a.txt"], "chunk_count": 1}]},
    )
    monkeypatch.setattr(
        questions,
        "search_all_chunks",
        lambda: [{"category": "客户准入", "source_file": "a.txt", "section_title": "", "content": "企业客户材料要求"}],
    )
    monkeypatch.setattr(questions, "RELATIONS", [])

    payload = questions.generate_evaluation_question_set()

    assert payload["question_generation"]["mode"] == "fallback"
    assert payload["questions"]


def test_run_evaluation_preserves_local_rerank_profile_without_remote_client(monkeypatch):
    local_settings = RagSettings(rerank_enabled=True, rerank_provider="local_onnx")
    profile = {
        "id": "local_profile",
        "name": "本地重排配置",
        "description": "本地模型评测",
        "profile_type": "single",
        "enabled_switches": ["rerank_enabled"],
        "settings": local_settings,
        "collection_name": None,
    }
    question_set = {
        "question_generation": {"mode": "fallback", "question_count": 1},
        "questions": [
            {
                "id": "q1",
                "question": "政企专线需要哪些材料？",
                "history": [],
                "category": "客户准入",
                "question_type": "单一规则",
                "expected_intent": "客户准入",
                "expected_retrieval_modes": ["vector"],
                "expect_references": True,
                "expect_relation_paths": False,
                "expected_source_files": ["a.txt"],
                "source_category": "客户准入",
            }
        ],
    }

    monkeypatch.setattr(runner, "load_rag_settings", lambda: local_settings)
    monkeypatch.setattr(runner, "get_settings", lambda: config.Settings(use_local_models=False), raising=False)
    monkeypatch.setattr(runner, "build_evaluation_profiles", lambda current_settings: [profile])
    monkeypatch.setattr(
        runner,
        "run_qa",
        lambda state: {
            **state,
            "answer": "本地回答",
            "intent": "客户准入",
            "retrieval_mode": "vector",
            "references": [{"source_file": "a.txt", "content": "材料要求", "score": 0.88}],
            "relation_paths": [],
            "trace": [{"node": "retrieve", "output": {"provider": "local_onnx"}}],
            "error": None,
        },
    )
    monkeypatch.setattr(runner.storage, "save_evaluation_run", lambda payload: payload)
    monkeypatch.setattr(
        siliconflow_client,
        "get_chat_client",
        lambda: (_ for _ in ()).throw(AssertionError("evaluation run must not create remote chat client")),
    )

    result = runner.run_evaluation(run_id="eval_local_test", question_set=question_set)

    assert result["question_generation"]["mode"] == "fallback"
    assert result["profiles"][0]["settings"]["rerank_provider"] == "local_onnx"
    assert result["profiles"][0]["cases"][0]["answer"] == "本地回答"
    assert result["profiles"][0]["cases"][0]["error"] is None


def test_start_evaluation_run_queues_before_generating_questions(monkeypatch):
    saved_progress: dict[str, dict] = {}
    started_threads = []

    class CapturingThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started_threads.append(self)

    monkeypatch.setattr(tasks.storage, "read_evaluation_progress", lambda run_id: saved_progress.get(run_id))
    monkeypatch.setattr(tasks.storage, "save_evaluation_progress", lambda payload: saved_progress.setdefault(payload["run_id"], payload.copy()) or payload)
    monkeypatch.setattr(tasks.threading, "Thread", CapturingThread)
    monkeypatch.setattr(
        tasks,
        "generate_evaluation_question_set",
        lambda: (_ for _ in ()).throw(AssertionError("question generation must run in background")),
    )
    monkeypatch.setattr(tasks, "get_evaluation_total_units", lambda question_set=None: 0)
    monkeypatch.setattr(tasks, "_RUNNING_RUN_ID", None)

    progress = tasks.start_evaluation_run()

    assert progress["status"] == "queued"
    assert progress["total_units"] == 0
    assert progress["progress_url"].startswith("/api/evaluations/")
    assert len(started_threads) == 1


def test_active_evaluation_ignores_disk_running_progress_without_live_worker(monkeypatch):
    monkeypatch.setattr(tasks, "_RUNNING_RUN_ID", None)
    monkeypatch.setattr(
        tasks.storage,
        "list_evaluation_progresses",
        lambda limit=10: [
            {
                "run_id": "eval_old",
                "status": "running",
                "updated_at": "2026-06-26 01:00:00",
            }
        ],
    )

    assert tasks.get_active_evaluation_progress() == {"status": "idle"}


def test_local_model_evaluation_case_uses_retrieval_without_full_qa(monkeypatch):
    profile = {
        "id": "local_fast_profile",
        "settings": RagSettings(rerank_enabled=False, rerank_provider="local_onnx"),
        "collection_name": None,
    }
    question = {
        "id": "q1",
        "question": "政企专线需要哪些材料？",
        "history": [],
        "category": "客户准入",
        "question_type": "单一规则",
        "expected_intent": "客户准入",
    }
    chunk = {
        "source_file": "a.txt",
        "source_path": "docs/a.txt",
        "section_title": "材料要求",
        "content": "客户需要提供营业执照和授权材料。",
        "score": 0.91,
    }

    monkeypatch.setattr(runner, "get_settings", lambda: config.Settings(use_local_models=True), raising=False)
    monkeypatch.setattr(
        runner,
        "run_qa",
        lambda state: (_ for _ in ()).throw(AssertionError("local evaluation should not call full QA graph")),
    )
    monkeypatch.setattr(runner, "get_category_names", lambda: ["客户准入"], raising=False)
    monkeypatch.setattr(
        runner,
        "check_business_scope",
        lambda question_text, categories: {"in_scope": True, "matched_entities": []},
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "heuristic_classify",
        lambda question_text, history, categories: {
            "intent": "客户准入",
            "question_type": "单一规则",
            "retrieval_mode": "vector",
            "entities": [],
        },
        raising=False,
    )
    monkeypatch.setattr(runner, "heuristic_tool_choice", lambda state: ("vector_rule_search", "test"), raising=False)
    monkeypatch.setattr(
        runner,
        "dispatch_retrieval",
        lambda query, intent, entities, selected_tool, **kwargs: {
            "mode": "vector",
            "chunks": [chunk],
            "relation_paths": [],
            "error": None,
            "trace": [{"node": "retrieve", "output": {"mode": "vector"}}],
        },
        raising=False,
    )

    case = runner._run_case("eval_local_fast", profile, question)

    assert case["error"] is None
    assert case["intent"] == "客户准入"
    assert case["retrieval_mode"] == "vector"
    assert case["references"] == [chunk]
    assert "营业执照" in case["answer"]


def test_evaluation_collection_is_built_without_publishing_production_generation(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    profile = {
        "id": "multi",
        "settings": RagSettings(multi_vector_enabled=True),
        "collection_name": "crabrag_eval_multi_vector",
    }
    principal = PrincipalContext.anonymous()
    context = RetrievalContext(
        generation_id="gen-1",
        principal=principal,
        allowed_document_ids=frozenset({"doc-a"}),
        permission_fingerprint="permission-1",
    )
    added = []
    registered = []
    monkeypatch.setattr(runner, "collection_status", lambda: {"count": 0, "metadata": {}})
    monkeypatch.setattr(runner, "_evaluation_fingerprint", lambda profile, generation_id: "eval-fp-1")
    monkeypatch.setattr(runner, "_evaluation_chunks", lambda settings: [{"id": "chunk-a"}])
    monkeypatch.setattr(
        runner.index_generation,
        "register_generation_resource",
        lambda generation_id, kind, name: registered.append((generation_id, kind, name)),
    )
    monkeypatch.setattr(
        runner,
        "add_chunks",
        lambda chunks, collection_metadata=None: added.append((chunks, collection_metadata)) or len(chunks),
    )

    with use_retrieval_context(context):
        runner.ensure_evaluation_collection(profile)

    assert registered == [("gen-1", "evaluation", "crabrag_eval_multi_vector__gen-1")]
    assert added == [
        (
            [{"id": "chunk-a"}],
            {"evaluation_fingerprint": "eval-fp-1", "embedding_fingerprint": runner.doc_status.embedding_fingerprint(runner.get_settings())},
        )
    ]


def test_evaluation_collection_rebuilds_when_fingerprint_changes_but_count_matches(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    profile = {
        "id": "multi",
        "settings": RagSettings(multi_vector_enabled=True),
        "collection_name": "crabrag_eval_multi_vector",
    }
    context = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset(),
        permission_fingerprint="permission-1",
    )
    rebuilt = []
    monkeypatch.setattr(runner, "collection_status", lambda: {"count": 1, "metadata": {"evaluation_fingerprint": "old"}})
    monkeypatch.setattr(runner, "_evaluation_fingerprint", lambda profile, generation_id: "new")
    monkeypatch.setattr(runner, "_evaluation_chunks", lambda settings: [{"id": "chunk-a"}])
    monkeypatch.setattr(runner.index_generation, "register_generation_resource", lambda *args: None)
    monkeypatch.setattr(runner, "add_chunks", lambda chunks, collection_metadata=None: rebuilt.append(list(chunks)) or len(chunks))

    with use_retrieval_context(context):
        runner.ensure_evaluation_collection(profile)

    assert rebuilt == [[{"id": "chunk-a"}]]


def test_evaluation_collection_keeps_matching_fingerprint_and_clears_stale_empty_collection(monkeypatch):
    from services.rag_api.security import PrincipalContext, RetrievalContext, use_retrieval_context

    profile = {
        "id": "multi",
        "settings": RagSettings(multi_vector_enabled=True),
        "collection_name": "crabrag_eval_multi_vector",
    }
    context = RetrievalContext(
        generation_id="gen-1",
        principal=PrincipalContext.anonymous(),
        allowed_document_ids=frozenset(),
        permission_fingerprint="permission-1",
    )
    rebuilt = []
    status = {"count": 0, "metadata": {"evaluation_fingerprint": "same"}}
    chunks = []
    monkeypatch.setattr(runner, "collection_status", lambda: status)
    monkeypatch.setattr(runner, "_evaluation_fingerprint", lambda profile, generation_id: "same")
    monkeypatch.setattr(runner, "_evaluation_chunks", lambda settings: list(chunks))
    monkeypatch.setattr(runner.index_generation, "register_generation_resource", lambda *args: None)
    monkeypatch.setattr(runner, "add_chunks", lambda items, collection_metadata=None: rebuilt.append(list(items)) or len(items))

    with use_retrieval_context(context):
        runner.ensure_evaluation_collection(profile)
        assert rebuilt == []
        status["count"] = 1
        runner.ensure_evaluation_collection(profile)

    assert rebuilt == [[]]
