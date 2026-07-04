from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from services.rag_api.exceptions import LLMServiceError

ALLOWED_ONNX_MODEL_FILES = ("model.onnx", "model_fp16.onnx", "model_int8.onnx", "model_q4.onnx")
REQUIRED_MODEL_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json")


class LocalOnnxEmbeddingModel:
    def __init__(self, tokenizer: Any, session: Any) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self.input_specs = {item.name: item for item in session.get_inputs()}
        self.input_names = set(self.input_specs)
        self.uses_kv_cache = any(name.startswith("past_key_values.") for name in self.input_names)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = self._build_inputs(texts)
        outputs = self.session.run(None, inputs)
        if not outputs:
            raise LLMServiceError("本地向量模型输出为空")
        hidden = np.asarray(outputs[0], dtype=np.float32)
        if self.uses_kv_cache or "position_ids" in self.input_names:
            vectors = _last_token_pool(hidden, inputs["attention_mask"])
        else:
            vectors = _mean_pool(hidden, inputs["attention_mask"])
        vectors = _l2_normalize(vectors)
        return vectors.astype(np.float32).tolist()

    def _build_inputs(self, texts: list[str]) -> dict[str, np.ndarray]:
        encodings = self.tokenizer.encode_batch(texts)
        max_len = max((len(item.ids) for item in encodings), default=0)
        if max_len <= 0:
            raise LLMServiceError("本地向量模型分词结果为空")

        input_ids = []
        attention_mask = []
        token_type_ids = []
        for item in encodings:
            ids = list(item.ids)
            mask = list(item.attention_mask)
            types = list(getattr(item, "type_ids", []) or [0] * len(ids))
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            token_type_ids.append(types + [0] * pad_len)

        payload = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
        }
        if "position_ids" in self.input_names:
            payload["position_ids"] = np.tile(np.arange(max_len, dtype=np.int64), (len(texts), 1))
        if "token_type_ids" in self.input_names:
            payload["token_type_ids"] = np.asarray(token_type_ids, dtype=np.int64)
        for name, spec in self.input_specs.items():
            if name.startswith("past_key_values."):
                payload[name] = _empty_past_key_value(spec, batch_size=len(texts))
        return {key: value for key, value in payload.items() if key in self.input_names}


def validate_local_model_dir(model_dir: Path, onnx_model_file: str = "model.onnx") -> Path:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        joined = "、".join(missing)
        raise LLMServiceError(f"本地向量模型文件缺失：{joined}。请确认模型已放入 {model_dir}")
    return _resolve_onnx_model_path(model_dir, onnx_model_file)


def embed_texts_local(texts: list[str], model_dir: Path, onnx_model_file: str = "model.onnx") -> list[list[float]]:
    if not texts:
        return []
    return get_local_embedding_model(str(model_dir), onnx_model_file).embed(texts)


@lru_cache(maxsize=4)
def get_local_embedding_model(model_dir: str, onnx_model_file: str = "model.onnx") -> LocalOnnxEmbeddingModel:
    path = Path(model_dir)
    model_path = validate_local_model_dir(path, onnx_model_file)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    session = _load_onnxruntime().InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return LocalOnnxEmbeddingModel(tokenizer, session)


def _load_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except Exception as exc:  # noqa: BLE001
        raise LLMServiceError(f"本地 ONNX runtime 不可用：{exc}") from exc


def _resolve_onnx_model_path(model_dir: Path, onnx_model_file: str) -> Path:
    if onnx_model_file not in ALLOWED_ONNX_MODEL_FILES:
        raise LLMServiceError(f"不支持的本地向量模型文件：{onnx_model_file}")
    candidates = [model_dir / onnx_model_file, model_dir / "onnx" / onnx_model_file]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise LLMServiceError(f"本地向量模型文件缺失：{onnx_model_file}。请确认模型已放入 {model_dir} 或 {model_dir / 'onnx'}")


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    if last_hidden_state.ndim != 3:
        raise LLMServiceError("本地向量模型输出维度异常")
    mask = attention_mask.astype(np.float32)
    expanded = np.expand_dims(mask, axis=-1)
    summed = np.sum(last_hidden_state * expanded, axis=1)
    counts = np.maximum(np.sum(expanded, axis=1), 1e-12)
    return summed / counts


def _last_token_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    if last_hidden_state.ndim != 3:
        raise LLMServiceError("本地向量模型输出维度异常")
    token_counts = np.sum(attention_mask, axis=1).astype(np.int64)
    indices = np.maximum(token_counts - 1, 0)
    return last_hidden_state[np.arange(last_hidden_state.shape[0]), indices]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def _empty_past_key_value(input_spec: Any, batch_size: int) -> np.ndarray:
    shape = list(getattr(input_spec, "shape", []) or [])
    if not shape:
        raise LLMServiceError(f"本地向量模型输入维度异常：{getattr(input_spec, 'name', '')}")
    dims: list[int] = []
    for index, dim in enumerate(shape):
        if index == 0:
            dims.append(batch_size)
        elif isinstance(dim, int):
            dims.append(dim)
        elif isinstance(dim, str) and "past_sequence_length" in dim:
            dims.append(0)
        else:
            raise LLMServiceError(f"本地向量模型输入维度异常：{getattr(input_spec, 'name', '')}")
    return np.zeros(tuple(dims), dtype=np.float32)
