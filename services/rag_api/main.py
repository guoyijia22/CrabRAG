from __future__ import annotations

import uuid

from fastapi import FastAPI, Query
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.rag_api.agent.graph import run_qa
from services.rag_api.app_settings import AppSettings, load_app_settings, save_app_settings
from services.rag_api.config import get_settings, read_app_config, write_system_name
from services.rag_api.document.categories import load_kb_categories
from services.rag_api.document.ingest import ingest_knowledge_base
from services.rag_api.document.ingest_tasks import read_ingest_progress, read_ingest_result, start_ingest_run
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
from services.rag_api.vector.chroma_store import collection_status

app = FastAPI(title="enterprise-line-compliance-rag-api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "enterprise-line-compliance-rag-api",
        "status": "ok",
        "endpoints": ["/api/health", "/api/chat", "/api/ingest", "/api/categories", "/api/graph", "/api/logs", "/api/settings", "/api/evaluations", "/docs"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    state = run_qa({"session_id": session_id, "question": request.question, "history": get_history(session_id), "trace": []})
    update_memory(session_id, request.question, state.get("answer", ""), state.get("intent", ""), state.get("entities", []))
    append_qa_log(
        {
            "session_id": session_id,
            "question": request.question,
            "intent": state.get("intent", ""),
            "question_type": state.get("question_type", ""),
            "retrieval_mode": state.get("retrieval_mode", ""),
            "entities": state.get("entities", []),
            "sources": [item.get("source_file", "") for item in state.get("references", [])],
            "answer": state.get("answer", ""),
            "error": state.get("error"),
        }
    )
    return ChatResponse(
        session_id=session_id,
        intent=state.get("intent", ""),
        question_type=state.get("question_type", ""),
        retrieval_mode=state.get("retrieval_mode", ""),
        entities=state.get("entities", []),
        answer=state.get("answer", ""),
        references=state.get("references", []),
        relation_paths=state.get("relation_paths", []),
        trace=state.get("trace", []),
        error=state.get("error"),
    )


@app.post("/api/ingest")
def ingest() -> dict:
    return start_ingest_run()


@app.post("/api/ingest/run")
def run_ingest() -> dict:
    return start_ingest_run()


@app.get("/api/ingest/{run_id}/progress")
def ingest_progress(run_id: str) -> dict:
    payload = read_ingest_progress(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="ingest progress not found")
    return payload


@app.get("/api/ingest/{run_id}")
def ingest_result(run_id: str) -> dict:
    payload = read_ingest_result(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="ingest result not found")
    return payload


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    chroma_state = "error"
    try:
        status = collection_status()
        chroma_state = f"ok:{status['count']}"
    except Exception:
        pass
    return {
        "web": "ok",
        "rag_service": "ok",
        "docs_dir_exists": settings.docs_dir.exists(),
        "chroma": chroma_state,
        "llm_api": "configured" if settings.api_key else "missing_api_key",
    }


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


@app.get("/api/model-settings", response_model=ModelApiSettingsResponse)
def get_model_settings() -> ModelApiSettingsResponse:
    return public_model_api_settings()


@app.put("/api/model-settings", response_model=ModelApiSettingsResponse)
def update_model_settings(settings: ModelApiSettingsRequest) -> ModelApiSettingsResponse:
    payload = update_model_api_settings(settings)
    get_settings.cache_clear()
    try:
        from services.rag_api.llm.siliconflow_client import get_chat_client, get_embedding_client

        get_chat_client.cache_clear()
        get_embedding_client.cache_clear()
    except Exception:
        pass
    return payload


@app.get("/api/categories")
def categories() -> dict:
    return load_kb_categories()


@app.get("/api/graph")
def graph_payload() -> dict:
    return build_graph_payload()


@app.post("/api/graph/subgraph")
def graph_subgraph(payload: dict) -> dict:
    return build_subgraph_payload(payload)


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
def logs(intent: str | None = Query(default=None)) -> dict:
    return {"items": read_qa_logs(intent)}


@app.get("/api/settings", response_model=SettingsResponse)
def get_rag_settings() -> RagSettings:
    return load_rag_settings()


@app.put("/api/settings", response_model=SettingsResponse)
def update_rag_settings(settings: RagSettings) -> RagSettings:
    return save_rag_settings(settings)


@app.post("/api/evaluations/run")
def run_evaluations() -> dict:
    return start_evaluation_run()


@app.get("/api/evaluations")
def evaluations() -> dict:
    return {"items": list_evaluation_runs()}


@app.get("/api/evaluations/active")
def active_evaluation() -> dict:
    return get_active_evaluation_progress()


@app.get("/api/evaluations/{run_id}")
def evaluation_detail(run_id: str) -> dict:
    payload = read_evaluation_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    return payload


@app.get("/api/evaluations/{run_id}/progress")
def evaluation_progress(run_id: str) -> dict:
    payload = read_evaluation_progress(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="evaluation progress not found")
    return payload
