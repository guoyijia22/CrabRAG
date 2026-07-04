from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from services.rag_api.exceptions import LLMServiceError

ALLOWED_ONNX_MODEL_FILES = ("model.onnx", "model_fp16.onnx", "model_int8.onnx", "model_q4.onnx")
REQUIRED_RERANK_MODEL_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json")
QWEN_RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
QWEN_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_RERANK_BATCH_SIZE = 1


class LocalOnnxRerankModel:
    def __init__(self, tokenizer: Any, session: Any) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self.input_specs = {item.name: item for item in session.get_inputs()}
        self.input_names = set(self.input_specs)
        self.output_names = {item.name for item in session.get_outputs()} if hasattr(session, "get_outputs") else set()
        self.pad_token_id = _pad_token_id(tokenizer)
        self.uses_kv_cache = any(name.startswith("past_key_values.") for name in self.input_names)
        self.is_causal_lm_reranker = self.uses_kv_cache and "logits" in self.output_names
        self.yes_token_id = _token_id(tokenizer, "yes") if self.is_causal_lm_reranker else None
        self.no_token_id = _token_id(tokenizer, "no") if self.is_causal_lm_reranker else None

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, float | int]]:
        if not documents:
            return []
        if self.is_causal_lm_reranker:
            return self._rerank_causal_lm(query, documents, top_n)
        inputs = self._build_inputs(query, documents)
        outputs = self.session.run(None, inputs)
        if not outputs:
            raise LLMServiceError("本地重排模型输出为空")
        logits = _extract_logits(np.asarray(outputs[0], dtype=np.float32))
        if len(logits) != len(documents):
            raise LLMServiceError("本地重排模型输出数量异常")
        scores = _sigmoid(logits)
        ranked = [{"index": index, "relevance_score": float(score)} for index, score in enumerate(scores)]
        ranked.sort(key=lambda item: item["relevance_score"], reverse=True)
        return ranked[: min(top_n, len(ranked))]

    def _rerank_causal_lm(self, query: str, documents: list[str], top_n: int) -> list[dict[str, float | int]]:
        ranked: list[dict[str, float | int]] = []
        for start in range(0, len(documents), QWEN_RERANK_BATCH_SIZE):
            batch = documents[start : start + QWEN_RERANK_BATCH_SIZE]
            prompts = [_qwen_rerank_prompt(query, document) for document in batch]
            inputs = self._build_inputs_from_encodings(self.tokenizer.encode_batch(prompts))
            outputs = self.session.run(None, inputs)
            if not outputs:
                raise LLMServiceError("本地重排模型输出为空")
            scores = _causal_lm_scores(
                np.asarray(outputs[0], dtype=np.float32),
                inputs["attention_mask"],
                self.yes_token_id,
                self.no_token_id,
            )
            for offset, score in enumerate(scores):
                ranked.append({"index": start + offset, "relevance_score": float(score)})
        ranked.sort(key=lambda item: item["relevance_score"], reverse=True)
        return ranked[: min(top_n, len(ranked))]

    def _build_inputs(self, query: str, documents: list[str]) -> dict[str, np.ndarray]:
        return self._build_inputs_from_encodings(self.tokenizer.encode_batch([(query, document) for document in documents]))

    def _build_inputs_from_encodings(self, encodings: list[Any]) -> dict[str, np.ndarray]:
        max_len = max((len(item.ids) for item in encodings), default=0)
        if max_len <= 0:
            raise LLMServiceError("本地重排模型分词结果为空")

        input_ids = []
        attention_mask = []
        token_type_ids = []
        for item in encodings:
            ids = list(item.ids)
            mask = list(item.attention_mask)
            types = list(getattr(item, "type_ids", []) or [0] * len(ids))
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            token_type_ids.append(types + [0] * pad_len)

        payload = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
        }
        if "position_ids" in self.input_names:
            payload["position_ids"] = np.tile(np.arange(max_len, dtype=np.int64), (len(encodings), 1))
        if "token_type_ids" in self.input_names:
            payload["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int64)
        for name, spec in self.input_specs.items():
            if name.startswith("past_key_values."):
                payload[name] = _empty_past_key_value(spec, batch_size=len(encodings))
        return {key: value for key, value in payload.items() if key in self.input_names}


def validate_local_rerank_model_dir(model_dir: Path, onnx_model_file: str = "model.onnx") -> Path:
    missing = [name for name in REQUIRED_RERANK_MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        joined = "、".join(missing)
        raise LLMServiceError(f"本地重排模型文件缺失：{joined}。请确认模型已放入 {model_dir}")
    return _resolve_onnx_model_path(model_dir, onnx_model_file)


def rerank_documents_local(query: str, documents: list[str], model_dir: Path, top_n: int, onnx_model_file: str = "model.onnx") -> list[dict[str, float | int]]:
    if not documents:
        return []
    return get_local_rerank_model(str(model_dir), onnx_model_file).rerank(query, documents, top_n)


@lru_cache(maxsize=4)
def get_local_rerank_model(model_dir: str, onnx_model_file: str = "model.onnx") -> LocalOnnxRerankModel:
    path = Path(model_dir)
    model_path = validate_local_rerank_model_dir(path, onnx_model_file)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    session = _load_onnxruntime().InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return LocalOnnxRerankModel(tokenizer, session)


def _load_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except Exception as exc:  # noqa: BLE001
        raise LLMServiceError(f"本地 ONNX runtime 不可用：{exc}") from exc


def _resolve_onnx_model_path(model_dir: Path, onnx_model_file: str) -> Path:
    if onnx_model_file not in ALLOWED_ONNX_MODEL_FILES:
        raise LLMServiceError(f"不支持的本地重排模型文件：{onnx_model_file}")
    candidates = [model_dir / onnx_model_file, model_dir / "onnx" / onnx_model_file]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise LLMServiceError(f"本地重排模型文件缺失：{onnx_model_file}。请确认模型已放入 {model_dir} 或 {model_dir / 'onnx'}")


def _extract_logits(output: np.ndarray) -> np.ndarray:
    if output.ndim == 1:
        return output
    if output.ndim == 2:
        if output.shape[1] == 1:
            return output[:, 0]
        return output[:, -1]
    raise LLMServiceError("本地重排模型输出维度异常")


def _causal_lm_scores(output: np.ndarray, attention_mask: np.ndarray, yes_token_id: int | None, no_token_id: int | None) -> np.ndarray:
    if yes_token_id is None or no_token_id is None:
        raise LLMServiceError("本地重排模型缺少 yes/no token")
    if output.ndim != 3:
        raise LLMServiceError("本地重排模型输出维度异常")
    token_counts = np.sum(attention_mask, axis=1).astype(np.int64)
    indices = np.maximum(token_counts - 1, 0)
    scores = []
    for batch_index, token_index in enumerate(indices):
        yes_logit = float(output[batch_index, token_index, yes_token_id])
        no_logit = float(output[batch_index, token_index, no_token_id])
        max_logit = max(yes_logit, no_logit)
        yes_score = np.exp(yes_logit - max_logit)
        no_score = np.exp(no_logit - max_logit)
        scores.append(float(yes_score / max(yes_score + no_score, 1e-12)))
    return np.asarray(scores, dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _qwen_rerank_prompt(query: str, document: str) -> str:
    return (
        f"<|im_start|>system\n{QWEN_RERANK_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {QWEN_RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query}\n\n<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )


def _pad_token_id(tokenizer: Any) -> int:
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id):
        for token in ("<pad>", "[PAD]", "<|endoftext|>"):
            token_id = token_to_id(token)
            if token_id is not None:
                return int(token_id)
    return 0


def _token_id(tokenizer: Any, token: str) -> int | None:
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id):
        token_id = token_to_id(token)
        if token_id is not None:
            return int(token_id)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        token_id = convert_tokens_to_ids(token)
        if token_id is not None:
            return int(token_id)
    return None


def _empty_past_key_value(input_spec: Any, batch_size: int) -> np.ndarray:
    shape = list(getattr(input_spec, "shape", []) or [])
    if not shape:
        raise LLMServiceError(f"本地重排模型输入维度异常：{getattr(input_spec, 'name', '')}")
    dims: list[int] = []
    for index, dim in enumerate(shape):
        if index == 0:
            dims.append(batch_size)
        elif isinstance(dim, int):
            dims.append(dim)
        elif isinstance(dim, str) and "past_sequence_length" in dim:
            dims.append(0)
        else:
            raise LLMServiceError(f"本地重排模型输入维度异常：{getattr(input_spec, 'name', '')}")
    return np.zeros(tuple(dims), dtype=np.float32)
