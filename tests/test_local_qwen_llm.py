from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.rag_api import config
from services.rag_api.exceptions import LLMServiceError
from services.rag_api.llm import siliconflow_client


def _write_qwen_model_dir(path: Path) -> None:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (path / name).write_text("{}", encoding="utf-8")
    onnx_dir = path / "onnx"
    onnx_dir.mkdir()
    for name in (
        "decoder_model_merged_q4.onnx",
        "decoder_model_merged_q4.onnx_data",
        "embed_tokens_q4.onnx",
        "embed_tokens_q4.onnx_data",
    ):
        (onnx_dir / name).write_bytes(b"qwen")


def test_validate_local_qwen_model_dir_reports_missing_files(tmp_path: Path):
    from services.rag_api.llm.local_qwen_llm import validate_local_qwen_model_dir

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LLMServiceError) as exc_info:
        validate_local_qwen_model_dir(tmp_path)

    message = str(exc_info.value)
    assert "本地大语言模型文件缺失" in message
    assert "tokenizer.json" in message
    assert "decoder_model_merged_q4.onnx" in message


def test_chat_completion_local_uses_qwen_worker_jsonl(monkeypatch, tmp_path: Path):
    from services.rag_api.llm import local_qwen_llm

    _write_qwen_model_dir(tmp_path)
    captured = {}

    class FakeStdin:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def write(self, data: str) -> None:
            self.payloads.append(json.loads(data))
            captured["request"] = self.payloads[-1]

        def flush(self) -> None:
            captured["flushed"] = True

    class FakeStdout:
        def readline(self) -> str:
            request_id = captured["request"]["id"]
            return json.dumps({"id": request_id, "ok": True, "text": "<think>\n\n</think>\n\n本地回答"}) + "\n"

    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            captured["args"] = args[0]
            captured["kwargs"] = kwargs
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = None

        def poll(self):
            return None

    monkeypatch.setattr(local_qwen_llm.subprocess, "Popen", FakeProcess)
    local_qwen_llm.shutdown_local_qwen_worker()

    answer = local_qwen_llm.chat_completion_local(
        [{"role": "user", "content": "资费标准是什么？"}],
        model_dir=tmp_path,
        temperature=0.2,
        max_tokens=1200,
    )

    assert answer == "本地回答"
    assert captured["request"]["messages"] == [{"role": "user", "content": "资费标准是什么？"}]
    assert captured["request"]["temperature"] == 0.2
    assert captured["request"]["max_tokens"] == 768
    assert "qwen35_worker.mjs" in " ".join(map(str, captured["args"]))
    assert captured["flushed"] is True


def test_chat_completion_uses_local_qwen_without_openai_client(monkeypatch, tmp_path: Path):
    from services.rag_api.llm import local_qwen_llm

    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: config.Settings(use_local_models=True, api_key=None, local_llm_model_dir=tmp_path),
    )
    monkeypatch.setattr(
        siliconflow_client,
        "get_chat_client",
        lambda: (_ for _ in ()).throw(AssertionError("remote chat client must not be used")),
    )
    monkeypatch.setattr(
        local_qwen_llm,
        "chat_completion_local",
        lambda messages, model_dir, temperature, max_tokens: f"{model_dir.name}:{messages[-1]['content']}",
    )

    assert siliconflow_client.chat_completion([{"role": "user", "content": "本地问题"}]) == f"{tmp_path.name}:本地问题"


def test_chat_completion_remote_mode_still_uses_openai_client(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "远程回答"})()})()]},
            )()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: config.Settings(use_local_models=False, api_key="chat-secret", chat_model="remote-model"),
    )
    monkeypatch.setattr(siliconflow_client, "get_chat_client", lambda: FakeClient())

    assert siliconflow_client.chat_completion([{"role": "user", "content": "远程问题"}]) == "远程回答"
    assert captured["model"] == "remote-model"
    assert captured["messages"] == [{"role": "user", "content": "远程问题"}]


def test_qwen_worker_uses_node_supported_cpu_device():
    worker = Path("services/rag_api/llm/qwen35_worker.mjs").read_text(encoding="utf-8")

    assert 'device: "cpu"' in worker
    assert 'device: "wasm"' not in worker
