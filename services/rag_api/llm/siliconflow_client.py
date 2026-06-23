from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from services.rag_api.config import get_settings
from services.rag_api.exceptions import LLMServiceError


@lru_cache(maxsize=1)
def get_chat_client() -> OpenAI:
    settings = get_settings()
    if not settings.api_key:
        raise LLMServiceError("missing api key")
    if not settings.openai_compatible:
        raise LLMServiceError("configured api is not openai compatible")
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.request_timeout)


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    settings = get_settings()
    if not settings.embedding_openai_compatible:
        raise LLMServiceError("configured embedding api is not openai compatible")
    return OpenAI(api_key=settings.embedding_api_key or "local", base_url=settings.embedding_base_url, timeout=settings.request_timeout)


def get_llm_client() -> OpenAI:
    return get_chat_client()


def _translate_error(exc: Exception) -> LLMServiceError:
    if isinstance(exc, APIStatusError) and exc.status_code in {401, 403, 429, 500, 503, 504}:
        return LLMServiceError(f"siliconflow status {exc.status_code}")
    if isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError)):
        return LLMServiceError("siliconflow timeout or connection error")
    if isinstance(exc, LLMServiceError):
        return exc
    return LLMServiceError("siliconflow request failed")


def chat_completion(messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 1200) -> str:
    settings = get_settings()
    try:
        response = get_chat_client().chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        raise _translate_error(exc) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if not texts:
        return []
    try:
        kwargs: dict[str, Any] = {"model": settings.embedding_model, "input": texts}
        response = get_embedding_client().embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    except Exception as exc:  # noqa: BLE001
        raise _translate_error(exc) from exc
