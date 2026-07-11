from __future__ import annotations

from contextlib import asynccontextmanager
import os
import uuid

from fastapi import FastAPI, Query, Request, Response
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.rag_api.agent.graph import run_qa
from services.rag_api.app_settings import AppSettings, SidebarImageUpload, load_app_settings, read_sidebar_image_asset, save_app_settings, save_sidebar_image
from services.rag_api.config import get_settings, read_app_config, write_system_name
from services.rag_api.document.categories import load_kb_categories
from services.rag_api.document.ingest import ingest_knowledge_base
from services.rag_api.document.ingest_tasks import get_active_ingest_progress, read_ingest_progress, read_ingest_result, start_ingest_run
from services.rag_api.document.loader import has_supported_documents
from services.rag_api.exceptions import DOC_LOAD_ERROR_MESSAGE, LLM_ERROR_MESSAGE, DocumentLoadError, LLMServiceError
from services.rag_api.evaluation.storage import list_evaluation_runs, read_evaluation_run
from services.rag_api.evaluation.tasks import get_active_evaluation_progress, read_evaluation_progress, start_evaluation_run
from services.rag_api.graph.graph_api import build_graph_payload, build_subgraph_payload
from services.rag_api.graph.schema_config import load_graph_schema, load_graph_schema_suggestion, save_graph_schema_config
from services.rag_api.logging_utils.qa_logger import append_qa_log, read_qa_logs
from services.rag_api.memory.conversation_memory import get_history, update_memory
from services.rag_api.model_api_settings import public_model_api_settings, update_model_api_settings
from services.rag_api.rag_settings import RagSettings, load_rag_settings, save_rag_settings
from services.rag_api.schemas import ChatRequest, ChatResponse, ConfigUpdateRequest, ModelApiSettingsRequest, ModelApiSettingsResponse, SettingsResponse
from services.rag_api.vector.chroma_store import collection_status, validate_generation_collections
from services.rag_api import index_generation
from services.rag_api.index_scheduler import INDEX_SCHEDULER
from services.rag_api.retrieval.cache import RETRIEVAL_CACHE
from services.rag_api.security import (
    PermissionServiceError,
    build_retrieval_context,
    principal_from_headers,
    use_retrieval_context,
)


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    INDEX_SCHEDULER.start()
    try:
        yield
    finally:
        INDEX_SCHEDULER.stop()


app = FastAPI(title="crabrag-api", lifespan=app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3003", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "crabrag-api",
        "status": "ok",
        "endpoints": ["/api/health", "/api/chat", "/api/ingest", "/api/index/status", "/api/categories", "/api/graph", "/api/logs", "/api/settings", "/api/evaluations", "/docs"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    principal, retrieval_context = _request_retrieval_context(http_request)
    session_id = request.session_id or str(uuid.uuid4())
    with use_retrieval_context(retrieval_context):
        state = run_qa(
            {
                "session_id": session_id,
                "question": request.question,
                "history": get_history(
                    session_id,
                    subject=principal.subject,
                    generation_id=retrieval_context.generation_id,
                    permission_fingerprint=retrieval_context.permission_fingerprint,
                ),
                "trace": [],
            }
        )
    references = _authorized_references(state.get("references", []), retrieval_context.allowed_document_ids)
    relation_paths = _authorized_references(state.get("relation_paths", []), retrieval_context.allowed_document_ids)
    update_memory(
        session_id,
        request.question,
        state.get("answer", ""),
        state.get("intent", ""),
        state.get("entities", []),
        subject=principal.subject,
        generation_id=retrieval_context.generation_id,
        permission_fingerprint=retrieval_context.permission_fingerprint,
    )
    append_qa_log(
        {
            "session_id": session_id,
            "subject": principal.subject,
            "generation_id": retrieval_context.generation_id,
            "permission_fingerprint": retrieval_context.permission_fingerprint,
            "document_ids": sorted(
                {str(item.get("document_id") or "") for item in references if item.get("document_id")}
            ),
            "question": request.question,
            "intent": state.get("intent", ""),
            "question_type": state.get("question_type", ""),
            "retrieval_mode": state.get("retrieval_mode", ""),
            "entities": state.get("entities", []),
            "sources": [item.get("source_file", "") for item in references],
            "answer": state.get("answer", ""),
            "error": state.get("error"),
        }
    )
    return ChatResponse(
        session_id=session_id,
        index_generation=retrieval_context.generation_id,
        intent=state.get("intent", ""),
        question_type=state.get("question_type", ""),
        retrieval_mode=state.get("retrieval_mode", ""),
        entities=state.get("entities", []),
        answer=state.get("answer", ""),
        references=references,
        relation_paths=relation_paths,
        trace=state.get("trace", []),
        error=state.get("error"),
    )


def _authorized_references(references: list[dict], allowed_document_ids: frozenset[str] | None) -> list[dict]:
    if allowed_document_ids is None:
        return references
    return [
        item
        for item in references
        if str(item.get("document_id") or "") in allowed_document_ids
    ]


@app.post("/api/ingest")
def ingest(http_request: Request) -> dict:
    _require_index_admin(http_request)
    return start_ingest_run()


@app.post("/api/ingest/run")
def run_ingest(http_request: Request) -> dict:
    _require_index_admin(http_request)
    return start_ingest_run()


@app.post("/api/ingest/full")
def full_ingest(http_request: Request) -> dict:
    _require_index_admin(http_request)
    return start_ingest_run(full_rebuild=True)


@app.get("/api/ingest/active")
def active_ingest(http_request: Request) -> dict:
    _require_index_admin(http_request)
    return get_active_ingest_progress()


@app.get("/api/ingest/{run_id}/progress")
def ingest_progress(run_id: str, http_request: Request) -> dict:
    _require_index_admin(http_request)
    payload = read_ingest_progress(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="ingest progress not found")
    return payload


@app.get("/api/ingest/{run_id}")
def ingest_result(run_id: str, http_request: Request) -> dict:
    _require_index_admin(http_request)
    payload = read_ingest_result(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="ingest result not found")
    return payload


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    docs_dir_exists = any(path.exists() and path.is_dir() for path in settings.docs_dirs)
    chroma_state = "error"
    try:
        status = collection_status()
        chroma_state = f"ok:{status['count']}"
    except Exception:
        pass
    try:
        active_generation = index_generation.active_generation_id()
    except index_generation.IndexStateError:
        active_generation = "error"
    return {
        "web": "ok",
        "rag_service": "ok",
        "docs_dir_exists": docs_dir_exists,
        "docs_dir_has_files": has_supported_documents(settings.docs_dirs) if docs_dir_exists else False,
        "docs_dirs": [str(path) for path in settings.docs_dirs],
        "chroma": chroma_state,
        "llm_api": "local_qwen_onnx" if settings.use_local_models else ("configured" if settings.api_key else "missing_api_key"),
        "active_generation": active_generation,
        "index_scheduler": INDEX_SCHEDULER.status(),
    }


@app.get("/api/index/status")
def index_status(http_request: Request) -> dict:
    principal = principal_from_headers(http_request.headers, internal_token=os.getenv("CRABRAG_INTERNAL_TOKEN"))
    if not principal.can_manage_index:
        raise HTTPException(status_code=403, detail="index management permission required")
    try:
        state = index_generation.load_index_state()
    except index_generation.IndexStateError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    active = _generation_manifest_or_none(state.get("active_generation"))
    previous = _generation_manifest_or_none(state.get("previous_generation"))
    return {
        **state,
        "active": active,
        "previous": previous,
        "can_rollback": bool(previous and previous.get("permission_schema_version") == 1),
        "cache": RETRIEVAL_CACHE.stats(),
        "scheduler": INDEX_SCHEDULER.status(),
    }


@app.post("/api/index/rollback")
def rollback_index(http_request: Request) -> dict:
    principal = principal_from_headers(http_request.headers, internal_token=os.getenv("CRABRAG_INTERNAL_TOKEN"))
    if not principal.can_manage_index:
        raise HTTPException(status_code=403, detail="index management permission required")
    try:
        current_state = index_generation.load_index_state()
        previous_generation = str(current_state.get("previous_generation") or "")
        if previous_generation:
            previous_manifest = index_generation.load_generation_manifest(previous_generation)
            index_generation.validate_generation_artifacts(previous_generation, previous_manifest)
            validate_generation_collections(previous_generation, previous_manifest)
        state = index_generation.rollback_generation()
    except index_generation.IndexStateError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    RETRIEVAL_CACHE.clear()
    from services.rag_api.memory import conversation_memory

    conversation_memory.SESSION_MEMORY.clear()
    return {**state, "status": "rolled_back"}


def _generation_manifest_or_none(generation_id) -> dict | None:
    if not generation_id:
        return None
    try:
        return index_generation.load_generation_manifest(str(generation_id))
    except ValueError:
        return None


@app.get("/api/config")
def app_config() -> dict:
    return read_app_config()


@app.put("/api/config")
def update_app_config(request: ConfigUpdateRequest) -> dict:
    write_system_name(request.system_name)
    return read_app_config()


@app.get("/api/app-settings", response_model=AppSettings)
def get_app_settings() -> AppSettings:
    return load_app_settings()


@app.put("/api/app-settings", response_model=AppSettings)
def update_app_settings(settings: AppSettings) -> AppSettings:
    return save_app_settings(settings)


@app.put("/api/app-settings/sidebar-image", response_model=AppSettings)
def update_sidebar_image(upload: SidebarImageUpload) -> AppSettings:
    try:
        return save_sidebar_image(upload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/app-assets/sidebar-image")
def get_sidebar_image() -> Response:
    data, content_type = read_sidebar_image_asset()
    return Response(content=data, media_type=content_type)


@app.get("/api/model-settings", response_model=ModelApiSettingsResponse)
def get_model_settings() -> ModelApiSettingsResponse:
    return public_model_api_settings()


@app.put("/api/model-settings", response_model=ModelApiSettingsResponse)
def update_model_settings(settings: ModelApiSettingsRequest) -> ModelApiSettingsResponse:
    payload = update_model_api_settings(settings)
    get_settings.cache_clear()
    try:
        from services.rag_api.llm.local_onnx_embedding import get_local_embedding_model
        from services.rag_api.llm.local_onnx_rerank import get_local_rerank_model
        from services.rag_api.llm.local_qwen_llm import shutdown_local_qwen_worker
        from services.rag_api.llm.siliconflow_client import get_chat_client, get_embedding_client

        get_chat_client.cache_clear()
        get_embedding_client.cache_clear()
        get_local_embedding_model.cache_clear()
        get_local_rerank_model.cache_clear()
        shutdown_local_qwen_worker()
    except Exception:
        pass
    return payload


@app.get("/api/categories")
def categories(http_request: Request) -> dict:
    _, retrieval_context = _request_retrieval_context(http_request)
    with use_retrieval_context(retrieval_context):
        return load_kb_categories()


@app.get("/api/graph")
def graph_payload(http_request: Request) -> dict:
    _, retrieval_context = _request_retrieval_context(http_request)
    with use_retrieval_context(retrieval_context):
        return build_graph_payload()


@app.post("/api/graph/subgraph")
def graph_subgraph(payload: dict, http_request: Request) -> dict:
    _, retrieval_context = _request_retrieval_context(http_request)
    allowed = retrieval_context.allowed_document_ids
    if allowed is not None:
        payload = {
            **payload,
            "relation_paths": _authorized_references(payload.get("relation_paths", []), allowed),
        }
    with use_retrieval_context(retrieval_context):
        return build_subgraph_payload(payload)


def _request_retrieval_context(http_request: Request):
    principal = principal_from_headers(http_request.headers, internal_token=os.getenv("CRABRAG_INTERNAL_TOKEN"))
    try:
        return principal, build_retrieval_context(principal)
    except PermissionServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _require_index_admin(http_request: Request):
    principal = principal_from_headers(http_request.headers, internal_token=os.getenv("CRABRAG_INTERNAL_TOKEN"))
    if not principal.can_manage_index:
        raise HTTPException(status_code=403, detail="需要索引管理权限")
    return principal


def _admin_retrieval_context(http_request: Request):
    _require_index_admin(http_request)
    return _request_retrieval_context(http_request)[1]


def _require_matching_evaluation(payload: dict, permission_fingerprint: str) -> dict:
    if payload.get("permission_fingerprint") != permission_fingerprint:
        raise HTTPException(status_code=403, detail="评测结果属于不同权限上下文")
    return payload


@app.get("/api/graph/schema")
def graph_schema() -> dict:
    return load_graph_schema()


@app.get("/api/graph/schema/suggestion")
def graph_schema_suggestion() -> dict:
    return load_graph_schema_suggestion()


@app.put("/api/graph/schema")
def update_graph_schema(payload: dict) -> dict:
    return save_graph_schema_config(payload)


@app.get("/api/logs")
def logs(http_request: Request, intent: str | None = Query(default=None)) -> dict:
    context = _admin_retrieval_context(http_request)
    return {"items": read_qa_logs(intent, permission_fingerprint=context.permission_fingerprint)}


@app.get("/api/settings", response_model=SettingsResponse)
def get_rag_settings() -> RagSettings:
    return load_rag_settings()


@app.put("/api/settings", response_model=SettingsResponse)
def update_rag_settings(settings: RagSettings) -> RagSettings:
    if not get_settings().use_local_models and settings.rerank_provider != "api":
        settings = settings.model_copy(update={"rerank_provider": "api"})
    payload = save_rag_settings(settings)
    try:
        from services.rag_api.llm.local_onnx_rerank import get_local_rerank_model

        get_local_rerank_model.cache_clear()
    except Exception:
        pass
    return payload


@app.post("/api/evaluations/run")
def run_evaluations(http_request: Request) -> dict:
    _, retrieval_context = _request_retrieval_context(http_request)
    _require_index_admin(http_request)
    return start_evaluation_run(retrieval_context)


@app.get("/api/evaluations")
def evaluations(http_request: Request) -> dict:
    context = _admin_retrieval_context(http_request)
    return {
        "items": [
            item for item in list_evaluation_runs()
            if item.get("permission_fingerprint") == context.permission_fingerprint
        ]
    }


@app.get("/api/evaluations/active")
def active_evaluation(http_request: Request) -> dict:
    context = _admin_retrieval_context(http_request)
    payload = get_active_evaluation_progress()
    if payload.get("status") == "idle":
        return payload
    return _require_matching_evaluation(payload, context.permission_fingerprint)


@app.get("/api/evaluations/{run_id}")
def evaluation_detail(run_id: str, http_request: Request) -> dict:
    context = _admin_retrieval_context(http_request)
    payload = read_evaluation_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    return _require_matching_evaluation(payload, context.permission_fingerprint)


@app.get("/api/evaluations/{run_id}/progress")
def evaluation_progress(run_id: str, http_request: Request) -> dict:
    context = _admin_retrieval_context(http_request)
    payload = read_evaluation_progress(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="evaluation progress not found")
    return _require_matching_evaluation(payload, context.permission_fingerprint)
