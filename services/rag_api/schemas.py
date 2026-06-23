from typing import Any

from pydantic import BaseModel, Field

from services.rag_api.model_api_settings import ModelApiSettingsUpdate, PublicModelApiSettings
from services.rag_api.rag_settings import RagSettings


class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    intent: str = ""
    question_type: str = ""
    retrieval_mode: str = ""
    entities: list[str] = []
    answer: str = ""
    references: list[dict[str, Any]] = []
    relation_paths: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    error: str | None = None


class SettingsResponse(RagSettings):
    pass


class ConfigUpdateRequest(BaseModel):
    system_name: str = Field(min_length=4, max_length=40)


class ModelApiSettingsResponse(PublicModelApiSettings):
    pass


class ModelApiSettingsRequest(ModelApiSettingsUpdate):
    pass
