from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Callable

from services.rag_api.agent.business_scope import check_business_scope
from services.rag_api.agent.graph import run_qa
from services.rag_api.agent.heuristics import heuristic_classify, heuristic_tool_choice
from services.rag_api.agent.tools import dispatch_retrieval
from services.rag_api.config import get_settings
from services.rag_api.document.categories import get_category_names
from services.rag_api.document import doc_status
from services.rag_api.document.multi_vector import expand_multi_vector_chunks
from services.rag_api.document.splitter import split_documents
from services.rag_api.evaluation.approval import record_quality_approvals
from services.rag_api.evaluation.profiles import build_evaluation_profiles, evaluation_collection_names, serialize_profile
from services.rag_api.evaluation.quality import attach_quality_gates
from services.rag_api.evaluation.questions import generate_evaluation_question_set
from services.rag_api.evaluation.scoring import attach_baseline_deltas, build_overall_summary, score_case, score_profile
from services.rag_api.evaluation import storage
from services.rag_api.llm.call_metrics import capture_model_calls
from services.rag_api.rag_settings import get_retrieval_top_k, load_rag_settings, override_rag_settings
from services.rag_api.vector.chroma_store import add_chunks, collection_status, override_collection_name
from services.rag_api import index_generation
from services.rag_api.security import (
    PrincipalContext,
    RetrievalContext,
    build_retrieval_context,
    current_retrieval_context,
    use_retrieval_context,
)

ProgressCallback = Callable[[dict], None]


def get_evaluation_total_units(question_set: dict | None = None) -> int:
    profiles = build_evaluation_profiles(load_rag_settings())
    question_set = question_set or generate_evaluation_question_set()
    return len(profiles) * len(question_set.get("questions", [])) + len(evaluation_collection_names(profiles))


def run_evaluation(
    run_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    question_set: dict | None = None,
    retrieval_context: RetrievalContext | None = None,
) -> dict:
    context = retrieval_context or current_retrieval_context() or build_retrieval_context(PrincipalContext.anonymous())
    with use_retrieval_context(context):
        return _run_evaluation(run_id=run_id, progress_callback=progress_callback, question_set=question_set)


def _run_evaluation(
    run_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    question_set: dict | None = None,
) -> dict:
    current_settings = load_rag_settings()
    profiles = build_evaluation_profiles(current_settings)
    run_id = run_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    question_set = question_set or generate_evaluation_question_set()
    questions = question_set["questions"]
    question_generation = question_set["question_generation"]
    total_units = len(profiles) * len(questions) + len(evaluation_collection_names(profiles))
    completed_units = 0
    prepared_collections: set[str] = set()

    def emit(
        *,
        status: str = "running",
        current_profile: str = "",
        current_question: str = "",
        message: str = "",
        error: str | None = None,
    ) -> None:
        if not progress_callback:
            return
        progress_callback(
            {
                "run_id": run_id,
                "status": status,
                "percent": int(round((completed_units / total_units) * 100)) if total_units else 100,
                "completed_units": completed_units,
                "total_units": total_units,
                "current_profile": current_profile,
                "current_question": current_question,
                "message": message,
                "error": error,
            }
        )

    emit(message="正在准备评测配置")
    serialized_profiles: list[dict] = []
    for profile_index, profile in enumerate(profiles, start=1):
        profile_name = profile.get("name", profile["id"])
        collection_name = profile.get("collection_name")
        if collection_name and collection_name not in prepared_collections:
            emit(current_profile=profile["id"], message=f"正在检查评测知识库：{profile_name}")
            ensure_evaluation_collection(profile)
            prepared_collections.add(collection_name)
            completed_units += 1
            emit(current_profile=profile["id"], message=f"评测知识库准备完成：{profile_name}")
        else:
            ensure_evaluation_collection(profile)
        cases = []
        for question_index, question in enumerate(questions, start=1):
            emit(
                current_profile=profile["id"],
                current_question=question["question"],
                message=f"正在执行 {profile_index}/{len(profiles)} 组配置，问题 {question_index}/{len(questions)}",
            )
            cases.append(_run_case(run_id, profile, question))
            completed_units += 1
            emit(
                current_profile=profile["id"],
                current_question=question["question"],
                message=f"已完成 {profile_name} 的第 {question_index} 个问题",
            )
        payload = {
            **serialize_profile(profile, include_settings=True),
            "cases": cases,
            "summary": score_profile(profile["id"], cases),
        }
        serialized_profiles.append(payload)
    attach_baseline_deltas(serialized_profiles)
    attach_quality_gates(
        serialized_profiles,
        gate_eligible=bool(question_generation.get("gate_eligible", False)),
    )
    security_context = current_retrieval_context()
    result = {
        "run_id": run_id,
        "generation_id": security_context.generation_id if security_context else "legacy",
        "subject": security_context.principal.subject if security_context else "anonymous",
        "permission_fingerprint": security_context.permission_fingerprint if security_context else "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile_count": len(serialized_profiles),
        "question_count": len(questions),
        "question_generation": question_generation,
        "questions": questions,
        "profiles": serialized_profiles,
        "summary": build_overall_summary(serialized_profiles, question_generation),
    }
    result["approved_profiles"] = record_quality_approvals(result)
    saved = storage.save_evaluation_run(result)
    completed_units = total_units
    emit(status="completed", message="评测完成")
    return saved


def ensure_evaluation_collection(profile: dict) -> None:
    collection_name = profile.get("collection_name")
    if not collection_name:
        return
    generation_id = _evaluation_generation_id()
    target_name = _evaluation_collection_name(profile)
    evaluation_fingerprint = _evaluation_fingerprint(profile, generation_id)
    embedding_fingerprint = doc_status.embedding_fingerprint(get_settings())
    index_generation.register_generation_resource(generation_id, "evaluation", target_name)
    with override_rag_settings(profile["settings"]), override_collection_name(target_name):
        chunks = _evaluation_chunks(profile["settings"])
        status = collection_status()
        metadata = status.get("metadata") or {}
        if int(status.get("count", 0)) != len(chunks) or metadata.get("evaluation_fingerprint") != evaluation_fingerprint:
            add_chunks(
                chunks,
                collection_metadata={
                    "evaluation_fingerprint": evaluation_fingerprint,
                    "embedding_fingerprint": embedding_fingerprint,
                },
            )


def _run_case(run_id: str, profile: dict, question: dict) -> dict:
    session_id = f"{run_id}:{profile['id']}:{question['id']}"
    state = {"session_id": session_id, "question": question["question"], "history": question.get("history", []), "trace": []}
    started = time.perf_counter()
    with capture_model_calls() as model_calls:
        try:
            with override_rag_settings(profile["settings"]), override_collection_name(_evaluation_collection_name(profile)):
                if get_settings().use_local_models:
                    result = _run_local_retrieval_case(state, profile)
                else:
                    result = run_qa(state)
            error = result.get("error")
        except Exception as exc:  # noqa: BLE001
            result = {"answer": "", "references": [], "trace": [], "relation_paths": [], "error": str(exc)}
            error = str(exc)
    model_call_snapshot = model_calls.snapshot()
    latency_ms = int((time.perf_counter() - started) * 1000)
    top_k = get_retrieval_top_k(profile["settings"])
    references = result.get("references", [])[:top_k]
    case = {
        "question_id": question["id"],
        "question": question["question"],
        "history": question.get("history", []),
        "category": question.get("category", ""),
        "question_type": question.get("question_type", ""),
        "answer": result.get("answer", ""),
        "intent": result.get("intent", ""),
        "retrieval_mode": result.get("retrieval_mode", ""),
        "references": references,
        "relation_paths": result.get("relation_paths", [])[:top_k],
        "trace": result.get("trace", []),
        "latency_ms": latency_ms,
        "model_call_count": model_call_snapshot["total_calls"],
        "model_calls": model_call_snapshot,
        "error": error,
        "expected": {
            "expected_intent": question.get("expected_intent", ""),
            "expected_retrieval_modes": question.get("expected_retrieval_modes", []),
            "expect_references": question.get("expect_references", True),
            "expect_relation_paths": question.get("expect_relation_paths", False),
            "expected_source_files": question.get("expected_source_files", []),
            "expected_document_ids": question.get("expected_document_ids", []),
            "expected_chunk_ids": question.get("expected_chunk_ids", []),
            "allowed_document_ids": question.get("allowed_document_ids", []),
            "retired_document_ids": question.get("retired_document_ids", []),
            "forbidden_document_ids": question.get("forbidden_document_ids", []),
            "source_category": question.get("source_category", ""),
        },
    }
    case["metrics"] = score_case(case)
    return case


def _run_local_retrieval_case(state: dict, profile: dict) -> dict:
    question = state["question"]
    history = state.get("history", [])
    categories = get_category_names()
    effective_question = question
    trace = state.get("trace", []) + [
        {"node": "evaluation_mode", "output": {"use_local_models": True, "llm_answer_generation": False}},
    ]
    scope = check_business_scope(effective_question, categories)
    trace.append({"node": "business_scope_check", "output": scope})
    if not scope.get("in_scope", True):
        return {
            **state,
            "answer": "",
            "intent": "业务外",
            "retrieval_mode": "none",
            "references": [],
            "relation_paths": [],
            "trace": trace,
            "error": None,
        }

    parsed = heuristic_classify(effective_question, history, categories)
    tool_state = {**state, **parsed, "effective_question": effective_question}
    selected_tool, reason = heuristic_tool_choice(tool_state)
    retrieval = dispatch_retrieval(
        effective_question,
        parsed["intent"],
        parsed.get("entities", []),
        selected_tool,
        allow_query_expansion=False,
        allow_rerank=bool(getattr(profile["settings"], "rerank_enabled", False)),
    )
    top_k = get_retrieval_top_k(profile["settings"])
    references = retrieval.get("chunks", [])[:top_k]
    trace.extend(retrieval.get("trace", []))
    trace.append(
        {
            "node": "agent_tool_choice",
            "output": {"selected_tool": selected_tool, "retrieval_mode": retrieval.get("mode", ""), "reason": reason},
        }
    )
    trace.append(
        {
            "node": "retrieve",
            "output": {"top_k": top_k, "mode": retrieval.get("mode", ""), "sources": [chunk.get("source_file", "") for chunk in references]},
        }
    )
    trace.append({"node": "generate_answer", "output": {"skipped": True, "reason": "local_evaluation_retrieval_only"}})
    return {
        **state,
        **parsed,
        "effective_question": effective_question,
        "retrieval_mode": retrieval.get("mode", parsed.get("retrieval_mode", "")),
        "references": references,
        "retrieved_chunks": references,
        "relation_paths": retrieval.get("relation_paths", [])[:top_k],
        "answer": _format_retrieval_only_answer(parsed["intent"], references),
        "trace": trace,
        "error": retrieval.get("error"),
    }


def _format_retrieval_only_answer(intent: str, references: list[dict]) -> str:
    if not references:
        return ""
    refs = "\n\n".join(
        f"{index}. 来源：《{chunk.get('source_file', '')}》\n原文片段：{chunk.get('content', '')}"
        for index, chunk in enumerate(references, start=1)
    )
    return f"【业务类别】\n{intent}\n\n【答复】\n评测模式仅汇总检索证据，不调用本地大模型生成最终答案。\n\n【参考知识库原文片段】\n{refs}"


def _evaluation_collection_name(profile: dict) -> str | None:
    collection_name = profile.get("collection_name")
    if not collection_name:
        return None
    generation_id = _evaluation_generation_id()
    return f"{collection_name}__{generation_id}"


def _evaluation_generation_id() -> str:
    context = current_retrieval_context()
    generation_id = context.generation_id if context else index_generation.active_generation_id()
    if not generation_id or generation_id == "legacy":
        raise RuntimeError("评测专用索引要求已发布的治理索引代")
    return str(generation_id)


def _evaluation_fingerprint(profile: dict, generation_id: str) -> str:
    payload = {
        "generation_id": generation_id,
        "pipeline_fingerprint": doc_status.pipeline_fingerprint(get_settings(), profile["settings"]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _evaluation_chunks(rag_settings) -> list[dict]:
    generation_id = (current_retrieval_context() or build_retrieval_context(PrincipalContext.anonymous())).generation_id
    manifest_path = index_generation.generation_artifact_path(generation_id, "doc_status.json")
    snapshot_dir = index_generation.generation_artifact_path(generation_id, "snapshots")
    manifest = doc_status.load_manifest(manifest_path)
    documents: list[dict] = []
    for document_id, record in sorted((manifest.get("documents") or {}).items()):
        if record.get("status") != doc_status.PROCESSED:
            continue
        snapshot = doc_status.load_snapshot(str(document_id), snapshot_dir)
        document = (snapshot or {}).get("document")
        if isinstance(document, dict):
            documents.append(document)
    chunks = expand_multi_vector_chunks(
        split_documents(documents, chunk_size=rag_settings.chunk_size, chunk_overlap=rag_settings.chunk_overlap),
        rag_settings,
    )
    embedding_fingerprint = doc_status.embedding_fingerprint(get_settings())
    for chunk in chunks:
        chunk.setdefault("metadata", {})["embedding_fingerprint"] = embedding_fingerprint
        chunk["metadata"]["generation_id"] = generation_id
    return chunks
