from __future__ import annotations

import pytest


class MemorySecretStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


@pytest.fixture(autouse=True)
def isolate_security_audit(tmp_path, monkeypatch):
    from services.rag_api import audit, config, model_api_settings, secret_store

    monkeypatch.setattr(audit, "SECURITY_AUDIT", audit.AuditLog(tmp_path / "audit" / "security-audit.jsonl"))
    monkeypatch.setattr(model_api_settings, "MODEL_API_SETTINGS_PATH", tmp_path / "model_api_settings.json")
    monkeypatch.setattr(model_api_settings, "ENV_PATH", tmp_path / ".env")
    store = MemorySecretStore()
    monkeypatch.setattr(secret_store, "get_secret_store", lambda: store)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
