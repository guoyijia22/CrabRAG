from services.rag_api.llm import call_metrics, siliconflow_client
from services.rag_api.rag_settings import RagSettings
from services.rag_api.retrieval import optimizations


def test_model_call_metrics_count_chat_and_embedding_batches(monkeypatch):
    class FakeChatCompletions:
        @staticmethod
        def create(**kwargs):
            del kwargs
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeEmbeddings:
        @staticmethod
        def create(**kwargs):
            return type(
                "Response",
                (),
                {"data": [type("Embedding", (), {"embedding": [1.0]})() for _ in kwargs["input"]]},
            )()

    monkeypatch.setattr(
        siliconflow_client,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "use_local_models": False,
                "chat_model": "chat",
                "embedding_provider": "api",
                "embedding_model": "embedding",
            },
        )(),
    )
    monkeypatch.setattr(
        siliconflow_client,
        "get_chat_client",
        lambda: type("Client", (), {"chat": type("Chat", (), {"completions": FakeChatCompletions()})()})(),
    )
    monkeypatch.setattr(
        siliconflow_client,
        "get_embedding_client",
        lambda: type("Client", (), {"embeddings": FakeEmbeddings()})(),
    )

    with call_metrics.capture_model_calls() as captured:
        siliconflow_client.chat_completion([{"role": "user", "content": "hello"}])
        siliconflow_client.embed_texts(["a", "b"])
        siliconflow_client.embed_texts([])

    assert captured.snapshot() == {
        "total_calls": 2,
        "chat_calls": 1,
        "embedding_calls": 1,
        "embedding_inputs": 2,
        "rerank_calls": 0,
        "rerank_inputs": 0,
    }


def test_model_call_metrics_are_isolated_and_restore_outer_capture():
    with call_metrics.capture_model_calls() as outer:
        call_metrics.record_model_call("chat")
        with call_metrics.capture_model_calls() as inner:
            call_metrics.record_model_call("rerank", input_count=3)
        call_metrics.record_model_call("embedding", input_count=2)

    assert inner.snapshot()["total_calls"] == 1
    assert inner.snapshot()["rerank_inputs"] == 3
    assert outer.snapshot()["total_calls"] == 2
    assert outer.snapshot()["rerank_calls"] == 0
    assert call_metrics.current_model_calls() is None


def test_model_call_metrics_count_enabled_rerank_candidates(monkeypatch):
    class FakeLocalRerank:
        @staticmethod
        def rerank_documents_local(*args, **kwargs):
            del args, kwargs
            return [{"index": 0, "relevance_score": 0.9}]

    monkeypatch.setattr(
        optimizations,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {"local_rerank_model_dir": ".", "rerank_onnx_model_file": "model.onnx"},
        )(),
    )
    monkeypatch.setattr(optimizations, "_local_onnx_rerank_module", lambda: FakeLocalRerank)
    settings = RagSettings(rerank_enabled=True, rerank_provider="local_onnx")

    with call_metrics.capture_model_calls() as captured:
        optimizations.apply_rerank("query", [{"content": "a"}, {"content": "b"}], settings, top_k=1)

    assert captured.snapshot()["rerank_calls"] == 1
    assert captured.snapshot()["rerank_inputs"] == 2
