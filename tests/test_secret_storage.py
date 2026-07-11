from __future__ import annotations

import json

import pytest

from services.rag_api import model_api_settings
from services.rag_api import secret_store
from services.rag_api.secret_store import SecretStorageError


class FakeSecretStore:
    def __init__(self, values=None, *, fail_on_set: str | None = None) -> None:
        self.values = dict(values or {})
        self.fail_on_set = fail_on_set

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        if name == self.fail_on_set:
            raise SecretStorageError("write failed")
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


def _isolate(tmp_path, monkeypatch, store: FakeSecretStore) -> None:
    monkeypatch.setattr(model_api_settings, "MODEL_API_SETTINGS_PATH", tmp_path / "model_api_settings.json")
    monkeypatch.setattr(model_api_settings.secret_store, "get_secret_store", lambda: store)


def test_model_update_saves_keys_to_secret_store_and_never_json(tmp_path, monkeypatch):
    store = FakeSecretStore()
    _isolate(tmp_path, monkeypatch, store)

    public = model_api_settings.update_model_api_settings(model_api_settings.ModelApiSettingsUpdate(
        api_key="chat-secret",
        embedding_api_key="embedding-secret",
        rerank_api_key="rerank-secret",
    ))

    payload = json.loads((tmp_path / "model_api_settings.json").read_text(encoding="utf-8"))
    assert payload["api_key"] == ""
    assert payload["embedding_api_key"] == ""
    assert payload["rerank_api_key"] == ""
    assert store.values == {
        "chat_api_key": "chat-secret",
        "embedding_api_key": "embedding-secret",
        "rerank_api_key": "rerank-secret",
    }
    assert public.api_key_source == "keyring"
    assert public.embedding_api_key_source == "keyring"
    assert public.rerank_api_key_source == "keyring"


def test_environment_key_has_priority_over_keyring(tmp_path, monkeypatch):
    store = FakeSecretStore({"chat_api_key": "stored-secret"})
    _isolate(tmp_path, monkeypatch, store)
    monkeypatch.setenv("CRABRAG_API_KEY", "environment-secret")

    public = model_api_settings.public_model_api_settings()

    assert model_api_settings.effective_api_key() == "environment-secret"
    assert public.api_key_source == "env"
    assert public.api_key_hint == "已通过环境变量配置"


def test_legacy_plaintext_migration_verifies_store_then_clears_json(tmp_path, monkeypatch):
    store = FakeSecretStore()
    _isolate(tmp_path, monkeypatch, store)
    path = tmp_path / "model_api_settings.json"
    path.write_text(
        json.dumps({"api_key": "legacy-chat", "embedding_api_key": "legacy-embedding"}),
        encoding="utf-8",
    )

    settings = model_api_settings.load_model_api_settings()

    assert settings.api_key == ""
    assert settings.embedding_api_key == ""
    assert store.values["chat_api_key"] == "legacy-chat"
    assert store.values["embedding_api_key"] == "legacy-embedding"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["api_key"] == ""
    assert payload["embedding_api_key"] == ""


def test_failed_legacy_migration_preserves_json_and_rolls_back_store(tmp_path, monkeypatch):
    store = FakeSecretStore({"chat_api_key": "previous-chat"}, fail_on_set="embedding_api_key")
    _isolate(tmp_path, monkeypatch, store)
    path = tmp_path / "model_api_settings.json"
    original = json.dumps({"api_key": "legacy-chat", "embedding_api_key": "legacy-embedding"})
    path.write_text(original, encoding="utf-8")

    with pytest.raises(SecretStorageError):
        model_api_settings.load_model_api_settings()

    assert path.read_text(encoding="utf-8") == original
    assert store.values == {"chat_api_key": "previous-chat"}


def test_clear_removes_keyring_value_without_writing_plaintext(tmp_path, monkeypatch):
    store = FakeSecretStore({"chat_api_key": "stored-secret"})
    _isolate(tmp_path, monkeypatch, store)

    public = model_api_settings.update_model_api_settings(
        model_api_settings.ModelApiSettingsUpdate(clear_api_key=True)
    )

    assert "chat_api_key" not in store.values
    assert public.api_key_source == "missing"


def test_keyring_adapter_converts_backend_errors_to_controlled_error(monkeypatch):
    monkeypatch.setattr(
        secret_store.keyring,
        "get_password",
        lambda *args: (_ for _ in ()).throw(secret_store.KeyringError("unavailable")),
    )

    with pytest.raises(SecretStorageError, match="安全存储不可用"):
        secret_store.KeyringSecretStore().get("chat_api_key")


def test_json_write_failure_restores_previous_secure_value(tmp_path, monkeypatch):
    store = FakeSecretStore({"chat_api_key": "previous-secret"})
    _isolate(tmp_path, monkeypatch, store)
    monkeypatch.setattr(
        model_api_settings,
        "_write_settings_json",
        lambda settings: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        model_api_settings.update_model_api_settings(
            model_api_settings.ModelApiSettingsUpdate(api_key="new-secret")
        )

    assert store.values["chat_api_key"] == "previous-secret"
    assert not (tmp_path / "model_api_settings.json").exists()
