from __future__ import annotations

import atexit
import json
import os
import queue
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import TextIO

from services.rag_api.config import PROJECT_DIR
from services.rag_api.exceptions import LLMServiceError

REQUIRED_QWEN_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "onnx/decoder_model_merged_q4.onnx",
    "onnx/decoder_model_merged_q4.onnx_data",
    "onnx/embed_tokens_q4.onnx",
    "onnx/embed_tokens_q4.onnx_data",
)
WORKER_PATH = Path(__file__).with_name("qwen35_worker.mjs")
DEFAULT_LOCAL_MAX_TOKENS = 768
DEFAULT_LOCAL_TIMEOUT_SECONDS = 300.0

_WORKER: _QwenWorker | None = None
_WORKER_LOCK = threading.Lock()


def validate_local_qwen_model_dir(model_dir: Path) -> Path:
    model_dir = Path(model_dir)
    missing = [name for name in REQUIRED_QWEN_FILES if not (model_dir / name).exists()]
    if missing:
        joined = "、".join(missing)
        raise LLMServiceError(f"本地大语言模型文件缺失：{joined}。请确认模型已放入 {model_dir}")
    return model_dir


def chat_completion_local(
    messages: list[dict[str, str]],
    model_dir: Path,
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> str:
    model_dir = validate_local_qwen_model_dir(model_dir)
    worker = _get_worker(model_dir)
    return worker.generate(messages, temperature=temperature, max_tokens=max_tokens)


def shutdown_local_qwen_worker() -> None:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is not None:
            _WORKER.shutdown()
            _WORKER = None


def _get_worker(model_dir: Path) -> "_QwenWorker":
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None or _WORKER.model_dir != model_dir:
            if _WORKER is not None:
                _WORKER.shutdown()
            _WORKER = _QwenWorker(model_dir)
        return _WORKER


class _QwenWorker:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: queue.Queue[str] = queue.Queue(maxsize=20)

    def generate(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        with self._lock:
            process = self._ensure_process()
            request_id = uuid.uuid4().hex
            payload = {
                "id": request_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max(1, min(int(max_tokens or DEFAULT_LOCAL_MAX_TOKENS), DEFAULT_LOCAL_MAX_TOKENS)),
            }
            try:
                if process.stdin is None or process.stdout is None:
                    raise LLMServiceError("本地大语言模型 worker 管道不可用")
                process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                process.stdin.flush()
                line = self._readline_with_timeout(process.stdout, DEFAULT_LOCAL_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, LLMServiceError):
                    raise
                self.shutdown()
                raise LLMServiceError(f"本地大语言模型生成失败：{exc}") from exc
            if not line:
                self.shutdown()
                raise LLMServiceError(f"本地大语言模型生成失败：worker 无响应。{self._stderr_tail()}")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self.shutdown()
                raise LLMServiceError(f"本地大语言模型生成失败：worker 返回非 JSON 内容。{line[:200]}") from exc
            if response.get("id") != request_id:
                raise LLMServiceError("本地大语言模型生成失败：worker 响应 ID 不匹配")
            if not response.get("ok"):
                raise LLMServiceError(f"本地大语言模型生成失败：{response.get('error') or self._stderr_tail()}")
            return _clean_generated_text(str(response.get("text") or ""))

    def shutdown(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        except Exception:
            pass

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        bun = PROJECT_DIR / "runtime" / "bun" / "bun.exe"
        bun_command = str(bun) if bun.exists() else "bun"
        if not WORKER_PATH.exists():
            raise LLMServiceError(f"本地大语言模型 worker 文件缺失：{WORKER_PATH}")
        env = os.environ.copy()
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        self._process = subprocess.Popen(
            [bun_command, str(WORKER_PATH), "--model-dir", str(self.model_dir)],
            cwd=str(PROJECT_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if self._process.stderr is not None:
            threading.Thread(target=self._drain_stderr, args=(self._process.stderr,), daemon=True).start()
        return self._process

    def _drain_stderr(self, stream: TextIO) -> None:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if self._stderr_lines.full():
                try:
                    self._stderr_lines.get_nowait()
                except queue.Empty:
                    pass
            self._stderr_lines.put_nowait(line)

    def _stderr_tail(self) -> str:
        lines = list(self._stderr_lines.queue)
        return "；".join(lines[-3:])

    @staticmethod
    def _readline_with_timeout(stream: TextIO, timeout_seconds: float) -> str:
        result: queue.Queue[str] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                result.put(stream.readline())
            except Exception as exc:  # noqa: BLE001
                result.put(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        try:
            return result.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise LLMServiceError("本地大语言模型生成超时") from exc


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, flags=re.S)
    return text.strip()


atexit.register(shutdown_local_qwen_worker)
