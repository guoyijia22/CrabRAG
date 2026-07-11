from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal

ModelCallKind = Literal["chat", "embedding", "rerank"]


@dataclass
class ModelCallMetrics:
    chat_calls: int = 0
    embedding_calls: int = 0
    embedding_inputs: int = 0
    rerank_calls: int = 0
    rerank_inputs: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "total_calls": self.chat_calls + self.embedding_calls + self.rerank_calls,
            "chat_calls": self.chat_calls,
            "embedding_calls": self.embedding_calls,
            "embedding_inputs": self.embedding_inputs,
            "rerank_calls": self.rerank_calls,
            "rerank_inputs": self.rerank_inputs,
        }


_CURRENT_METRICS: ContextVar[ModelCallMetrics | None] = ContextVar("model_call_metrics", default=None)


def current_model_calls() -> ModelCallMetrics | None:
    return _CURRENT_METRICS.get()


def record_model_call(kind: ModelCallKind, *, input_count: int = 0) -> None:
    metrics = current_model_calls()
    if metrics is None:
        return
    if kind == "chat":
        metrics.chat_calls += 1
    elif kind == "embedding":
        metrics.embedding_calls += 1
        metrics.embedding_inputs += max(0, input_count)
    elif kind == "rerank":
        metrics.rerank_calls += 1
        metrics.rerank_inputs += max(0, input_count)


@contextmanager
def capture_model_calls() -> Iterator[ModelCallMetrics]:
    metrics = ModelCallMetrics()
    token = _CURRENT_METRICS.set(metrics)
    try:
        yield metrics
    finally:
        _CURRENT_METRICS.reset(token)
