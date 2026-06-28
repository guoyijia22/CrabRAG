from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from services.rag_api import config
from services.rag_api.exceptions import LLMServiceError
from services.rag_api.llm import siliconflow_client


def test_local_onnx_embedding_returns_normalized_768d_vectors():
    from services.rag_api.llm.local_onnx_embedding import LocalOnnxEmbeddingModel

    class FakeEncoding:
        def __init__(self, ids, attention_mask, type_ids):
            self.ids = ids
            self.attention_mask = attention_mask
            self.type_ids = type_ids

    class FakeTokenizer:
        def encode_batch(self, texts):
            assert texts == ["甲", "乙"]
            return [
                FakeEncoding([101, 1, 102], [1, 1, 0], [0, 0, 0]),
                FakeEncoding([101, 2, 102], [1, 1, 1], [0, 0, 0]),
            ]

    class FakeInput:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

        def run(self, output_names, inputs):
            assert output_names is None
            assert sorted(inputs) == ["attention_mask", "input_ids", "token_type_ids"]
            hidden = np.zeros((2, 3, 768), dtype=np.float32)
            hidden[0, 0, 0] = 2.0
            hidden[0, 1, 0] = 2.0
            hidden[1, 0, 1] = 3.0
            hidden[1, 1, 1] = 3.0
            hidden[1, 2, 1] = 3.0
            return [hidden]

    model = LocalOnnxEmbeddingModel(FakeTokenizer(), FakeSession())

    vectors = model.embed(["甲", "乙"])

    assert len(vectors) == 2
    assert all(len(vector) == 768 for vector in vectors)
    assert vectors[0][0] == pytest.approx(1.0)
    assert vectors[1][1] == pytest.approx(1.0)
    assert all(math.isclose(np.linalg.norm(vector), 1.0, rel_tol=1e-6) for vector in vectors)


def test_qwen_embedding_supplies_position_ids_and_empty_kv_cache():
    from services.rag_api.llm.local_onnx_embedding import LocalOnnxEmbeddingModel

    class FakeEncoding:
        def __init__(self, ids, attention_mask):
            self.ids = ids
            self.attention_mask = attention_mask
            self.type_ids = [0] * len(ids)

    class FakeTokenizer:
        def encode_batch(self, texts):
            assert texts == ["政企专线", "办理材料"]
            return [
                FakeEncoding([11, 12, 13], [1, 1, 0]),
                FakeEncoding([21, 22], [1, 1]),
            ]

    class FakeInput:
        def __init__(self, name, shape=None):
            self.name = name
            self.shape = shape or []

    class FakeSession:
        def get_inputs(self):
            return [
                FakeInput("input_ids"),
                FakeInput("attention_mask"),
                FakeInput("position_ids"),
                FakeInput("past_key_values.0.key", ["batch_size", 8, "past_sequence_length", 128]),
                FakeInput("past_key_values.0.value", ["batch_size", 8, "past_sequence_length", 128]),
            ]

        def run(self, output_names, inputs):
            assert output_names is None
            assert sorted(inputs) == [
                "attention_mask",
                "input_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "position_ids",
            ]
            np.testing.assert_array_equal(inputs["position_ids"], np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64))
            assert inputs["past_key_values.0.key"].shape == (2, 8, 0, 128)
            assert inputs["past_key_values.0.key"].dtype == np.float32
            assert inputs["past_key_values.0.value"].shape == (2, 8, 0, 128)
            hidden = np.zeros((2, 3, 1024), dtype=np.float32)
            hidden[0, 1, 5] = 4.0
            hidden[1, 1, 9] = 7.0
            return [hidden]

    model = LocalOnnxEmbeddingModel(FakeTokenizer(), FakeSession())

    vectors = model.embed(["政企专线", "办理材料"])

    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)
    assert vectors[0][5] == pytest.approx(1.0)
    assert vectors[1][9] == pytest.approx(1.0)
    assert all(math.isclose(np.linalg.norm(vector), 1.0, rel_tol=1e-6) for vector in vectors)


def test_local_model_dir_validation_reports_missing_files(tmp_path):
    from services.rag_api.llm.local_onnx_embedding import validate_local_model_dir

    (tmp_path / "model.onnx").write_bytes(b"placeholder")

    with pytest.raises(LLMServiceError, match="本地向量模型文件缺失"):
        validate_local_model_dir(tmp_path)


def test_local_embedding_model_loads_selected_onnx_file_from_onnx_subdir(monkeypatch, tmp_path: Path):
    from services.rag_api.llm import local_onnx_embedding

    for name in ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_int8.onnx").write_bytes(b"int8")

    captured = {}

    class FakeTokenizer:
        def enable_truncation(self, max_length):
            captured["max_length"] = max_length

    class FakeInput:
        name = "input_ids"

    class FakeSession:
        def __init__(self, path, providers):
            captured["path"] = Path(path)
            captured["providers"] = providers

        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(local_onnx_embedding.Tokenizer, "from_file", lambda path: FakeTokenizer())
    monkeypatch.setattr(local_onnx_embedding.ort, "InferenceSession", FakeSession)
    local_onnx_embedding.get_local_embedding_model.cache_clear()

    local_onnx_embedding.get_local_embedding_model(str(tmp_path), "model_int8.onnx")

    assert captured["path"] == tmp_path / "onnx" / "model_int8.onnx"
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["max_length"] == 512


def test_local_embedding_accepts_qwen_tokenizer_without_vocab_txt(tmp_path: Path):
    from services.rag_api.llm.local_onnx_embedding import validate_local_model_dir

    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_int8.onnx").write_bytes(b"int8")

    assert validate_local_model_dir(tmp_path, "model_int8.onnx") == tmp_path / "onnx" / "model_int8.onnx"


def test_local_embedding_missing_selected_onnx_file_names_file(tmp_path: Path):
    from services.rag_api.llm.local_onnx_embedding import validate_local_model_dir

    for name in ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(LLMServiceError, match="model_int8.onnx"):
        validate_local_model_dir(tmp_path, "model_int8.onnx")


def test_embed_texts_uses_local_provider_without_embedding_client(monkeypatch):
    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: config.Settings(
            api_key="chat-secret",
            embedding_provider="local_onnx",
            embedding_api_key="",
            embedding_base_url="https://should-not-be-used.invalid/v1",
            embedding_model="Qwen3-Embedding-0.6B-ONNX",
            embedding_onnx_model_file="model_int8.onnx",
        ),
    )
    monkeypatch.setattr(
        siliconflow_client.local_onnx_embedding,
        "embed_texts_local",
        lambda texts, model_dir, onnx_model_file: [[1.0] + [0.0] * 767 for _ in texts],
    )
    monkeypatch.setattr(
        siliconflow_client,
        "get_embedding_client",
        lambda: (_ for _ in ()).throw(AssertionError("API embedding client should not be used")),
    )

    vectors = siliconflow_client.embed_texts(["离线文本"])

    assert vectors == [[1.0] + [0.0] * 767]
