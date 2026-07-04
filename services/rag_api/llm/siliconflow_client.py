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
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.request_timeout)


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    settings = get_settings()
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
    if settings.use_local_models:
        try:
            local_qwen_llm = _local_qwen_llm_module()
            return local_qwen_llm.chat_completion_local(
                messages,
                settings.local_llm_model_dir,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc
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
    if settings.embedding_provider == "local_onnx":
        try:
            local_onnx_embedding = _local_onnx_embedding_module()
            return local_onnx_embedding.embed_texts_local(
                texts,
                settings.local_embedding_model_dir,
                settings.embedding_onnx_model_file,
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate_error(exc) from exc
    try:
        kwargs: dict[str, Any] = {"model": settings.embedding_model, "input": texts}
        response = get_embedding_client().embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    except Exception as exc:  # noqa: BLE001
        raise _translate_error(exc) from exc


def _local_qwen_llm_module():
    from services.rag_api.llm import local_qwen_llm

    return local_qwen_llm


def _local_onnx_embedding_module():
    from services.rag_api.llm import local_onnx_embedding

    return local_onnx_embedding
