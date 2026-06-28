from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from services.rag_api import config
from services.rag_api.retrieval import optimizations
from services.rag_api.rag_settings import RagSettings


def test_local_onnx_rerank_scores_pairs_and_sorts_documents():
    from services.rag_api.llm.local_onnx_rerank import LocalOnnxRerankModel

    class FakeEncoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids
            self.attention_mask = [1] * len(ids)
            self.type_ids = [0] * len(ids)

    class FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pairs = []

        def encode_batch(self, pairs):
            self.pairs = list(pairs)
            return [FakeEncoding([101, index + 1, 102]) for index, _ in enumerate(pairs)]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

        def run(self, output_names, inputs):
            assert output_names is None
            assert inputs["input_ids"].shape == (3, 3)
            return [np.asarray([[0.0], [2.0], [-2.0]], dtype=np.float32)]

    tokenizer = FakeTokenizer()
    model = LocalOnnxRerankModel(tokenizer, FakeSession())

    scores = model.rerank("资费标准", ["普通材料", "资费办理材料", "无关材料"], top_n=2)

    assert tokenizer.pairs == [
        ("资费标准", "普通材料"),
        ("资费标准", "资费办理材料"),
        ("资费标准", "无关材料"),
    ]
    assert [item["index"] for item in scores] == [1, 0]
    assert scores[0]["relevance_score"] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert scores[1]["relevance_score"] == pytest.approx(0.5)


def test_qwen_causal_rerank_scores_yes_no_logits_and_sorts_documents():
    from services.rag_api.llm.local_onnx_rerank import LocalOnnxRerankModel

    class FakeEncoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids
            self.attention_mask = [1] * len(ids)
            self.type_ids = [0] * len(ids)

    class FakeInput:
        def __init__(self, name: str, shape=None) -> None:
            self.name = name
            self.shape = shape or []

    class FakeOutput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeTokenizer:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self._next_doc_id = 1

        def encode_batch(self, texts):
            self.prompts.extend(texts)
            encodings = []
            for _ in texts:
                encodings.append(FakeEncoding([101, self._next_doc_id, 102]))
                self._next_doc_id += 1
            return encodings

        def token_to_id(self, token):
            return {"yes": 7, "no": 3, "<pad>": 0}.get(token)

    class FakeSession:
        def get_inputs(self):
            return [
                FakeInput("input_ids"),
                FakeInput("attention_mask"),
                FakeInput("position_ids"),
                FakeInput("past_key_values.0.key", ["batch_size", 8, "past_sequence_length", 128]),
                FakeInput("past_key_values.0.value", ["batch_size", 8, "past_sequence_length", 128]),
            ]

        def get_outputs(self):
            return [FakeOutput("logits")]

        def run(self, output_names, inputs):
            assert output_names is None
            assert sorted(inputs) == [
                "attention_mask",
                "input_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "position_ids",
            ]
            assert inputs["past_key_values.0.key"].shape == (1, 8, 0, 128)
            logits = np.zeros((1, 3, 10), dtype=np.float32)
            doc_token = int(inputs["input_ids"][0, 1])
            if doc_token == 1:
                logits[0, 2, 7] = 0.0
                logits[0, 2, 3] = 0.0
            elif doc_token == 2:
                logits[0, 2, 7] = 3.0
                logits[0, 2, 3] = 0.0
            else:
                logits[0, 2, 7] = -2.0
                logits[0, 2, 3] = 2.0
            return [logits]

    tokenizer = FakeTokenizer()
    model = LocalOnnxRerankModel(tokenizer, FakeSession())

    scores = model.rerank("资费标准", ["普通材料", "资费办理材料", "无关材料"], top_n=2)

    assert len(tokenizer.prompts) == 3
    assert all("<Query>: 资费标准" in prompt for prompt in tokenizer.prompts)
    assert "<Document>: 资费办理材料" in tokenizer.prompts[1]
    assert all('answer can only be "yes" or "no"' in prompt for prompt in tokenizer.prompts)
    assert [item["index"] for item in scores] == [1, 0]
    assert scores[0]["relevance_score"] == pytest.approx(math.exp(3) / (math.exp(3) + 1))
    assert scores[1]["relevance_score"] == pytest.approx(0.5)


def test_local_rerank_model_dir_validation_reports_missing_files(tmp_path: Path):
    from services.rag_api.llm.local_onnx_rerank import validate_local_rerank_model_dir
    from services.rag_api.exceptions import LLMServiceError

    with pytest.raises(LLMServiceError, match="本地重排模型文件缺失"):
        validate_local_rerank_model_dir(tmp_path)


def test_local_rerank_model_loads_selected_onnx_file_from_onnx_subdir(monkeypatch, tmp_path: Path):
    from services.rag_api.llm import local_onnx_rerank

    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_int8.onnx").write_bytes(b"int8")

    captured = {}

    class FakeTokenizer:
        def enable_truncation(self, max_length):
            captured["max_length"] = max_length

        def token_to_id(self, token):
            return 0 if token == "<pad>" else None

    class FakeInput:
        name = "input_ids"

    class FakeSession:
        def __init__(self, path, providers):
            captured["path"] = Path(path)
            captured["providers"] = providers

        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(local_onnx_rerank.Tokenizer, "from_file", lambda path: FakeTokenizer())
    monkeypatch.setattr(local_onnx_rerank.ort, "InferenceSession", FakeSession)
    local_onnx_rerank.get_local_rerank_model.cache_clear()

    local_onnx_rerank.get_local_rerank_model(str(tmp_path), "model_int8.onnx")

    assert captured["path"] == tmp_path / "onnx" / "model_int8.onnx"
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["max_length"] == 512


def test_local_rerank_accepts_qwen_q4_model_file(tmp_path: Path):
    from services.rag_api.llm.local_onnx_rerank import validate_local_rerank_model_dir

    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_q4.onnx").write_bytes(b"q4")

    assert validate_local_rerank_model_dir(tmp_path, "model_q4.onnx") == tmp_path / "onnx" / "model_q4.onnx"


def test_local_rerank_missing_selected_onnx_file_names_file(tmp_path: Path):
    from services.rag_api.llm.local_onnx_rerank import validate_local_rerank_model_dir
    from services.rag_api.exceptions import LLMServiceError

    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(LLMServiceError, match="model_int8.onnx"):
        validate_local_rerank_model_dir(tmp_path, "model_int8.onnx")


def test_apply_rerank_uses_local_provider_without_http(monkeypatch, tmp_path: Path):
    chunks = [
        {"content": "普通材料", "source_file": "a.txt", "score": 0.4},
        {"content": "资费办理材料", "source_file": "b.txt", "score": 0.3},
    ]

    def fail_post(*args, **kwargs):
        raise AssertionError("local rerank must not call HTTP")

    def fake_rerank(query, documents, model_dir, top_n, onnx_model_file):
        assert query == "资费标准"
        assert documents == ["普通材料", "资费办理材料"]
        assert model_dir == tmp_path
        assert top_n == 1
        assert onnx_model_file == "model_int8.onnx"
        return [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.52}]

    monkeypatch.setattr(optimizations.requests, "post", fail_post)
    monkeypatch.setattr(optimizations.local_onnx_rerank, "rerank_documents_local", fake_rerank)
    monkeypatch.setattr(
        optimizations,
        "get_settings",
        lambda: config.Settings(local_rerank_model_dir=tmp_path, rerank_onnx_model_file="model_int8.onnx"),
    )

    reranked, trace = optimizations.apply_rerank(
        "资费标准",
        chunks,
        RagSettings(rerank_enabled=True, rerank_provider="local_onnx"),
        top_k=1,
    )

    assert [item["source_file"] for item in reranked] == ["b.txt"]
    assert reranked[0]["score"] == 0.91
    assert reranked[0]["rerank_score"] == 0.91
    assert trace["provider"] == "local_onnx"
    assert trace["returned_count"] == 1
    assert trace["fallback"] is False


def test_apply_rerank_local_missing_model_falls_back_without_http(monkeypatch, tmp_path: Path):
    chunks = [{"content": "候选片段", "source_file": "doc.txt", "score": 0.5}]

    def fail_post(*args, **kwargs):
        raise AssertionError("missing local rerank model must not call HTTP")

    monkeypatch.setattr(optimizations.requests, "post", fail_post)
    monkeypatch.setattr(optimizations, "get_settings", lambda: config.Settings(local_rerank_model_dir=tmp_path))

    reranked, trace = optimizations.apply_rerank(
        "资费标准",
        chunks,
        RagSettings(rerank_enabled=True, rerank_provider="local_onnx"),
        top_k=1,
    )

    assert reranked == chunks
    assert trace["provider"] == "local_onnx"
    assert trace["fallback"] is True
    assert "本地重排模型文件缺失" in trace["error"]
