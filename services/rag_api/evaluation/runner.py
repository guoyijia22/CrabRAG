from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Callable

from services.rag_api.agent.graph import run_qa
from services.rag_api.document.ingest import ingest_knowledge_base
from services.rag_api.evaluation.profiles import build_evaluation_profiles, evaluation_collection_names, serialize_profile
from services.rag_api.evaluation.questions import generate_evaluation_question_set
from services.rag_api.evaluation.scoring import attach_baseline_deltas, build_overall_summary, score_case, score_profile
from services.rag_api.evaluation import storage
from services.rag_api.rag_settings import get_retrieval_top_k, load_rag_settings, override_rag_settings
from services.rag_api.vector.chroma_store import collection_status, override_collection_name

ProgressCallback = Callable[[dict], None]


def get_evaluation_total_units(question_set: dict | None = None) -> int:
    profiles = build_evaluation_profiles(load_rag_settings())
    question_set = question_set or generate_evaluation_question_set()
    return len(profiles) * len(question_set.get("questions", [])) + len(evaluation_collection_names(profiles))


def run_evaluation(
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
    result = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile_count": len(serialized_profiles),
        "question_count": len(questions),
        "question_generation": question_generation,
        "questions": questions,
        "profiles": serialized_profiles,
        "summary": build_overall_summary(serialized_profiles, question_generation),
    }
    saved = storage.save_evaluation_run(result)
    completed_units = total_units
    emit(status="completed", message="评测完成")
    return saved


def ensure_evaluation_collection(profile: dict) -> None:
    collection_name = profile.get("collection_name")
    if not collection_name:
        return
    with override_rag_settings(profile["settings"]), override_collection_name(collection_name):
        status = collection_status()
        if status.get("count", 0) <= 0:
            ingest_knowledge_base()


def _run_case(run_id: str, profile: dict, question: dict) -> dict:
    session_id = f"{run_id}:{profile['id']}:{question['id']}"
    state = {"session_id": session_id, "question": question["question"], "history": question.get("history", []), "trace": []}
    started = time.perf_counter()
    try:
        with override_rag_settings(profile["settings"]), override_collection_name(profile.get("collection_name")):
            result = run_qa(state)
        error = result.get("error")
    except Exception as exc:  # noqa: BLE001
        result = {"answer": "", "references": [], "trace": [], "relation_paths": [], "error": str(exc)}
        error = str(exc)
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
        "error": error,
        "expected": {
            "expected_intent": question.get("expected_intent", ""),
            "expected_retrieval_modes": question.get("expected_retrieval_modes", []),
            "expect_references": question.get("expect_references", True),
            "expect_relation_paths": question.get("expect_relation_paths", False),
            "expected_source_files": question.get("expected_source_files", []),
            "source_category": question.get("source_category", ""),
        },
    }
    case["metrics"] = score_case(case)
    return case
