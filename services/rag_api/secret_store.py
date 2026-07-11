from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import keyring
from keyring.errors import KeyringError, PasswordDeleteError


KEYRING_SERVICE = "CrabRAG"


class SecretStorageError(RuntimeError):
    pass


class SecretStore(Protocol):
    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


class KeyringSecretStore:
    def __init__(self, service_name: str = KEYRING_SERVICE) -> None:
        self.service_name = service_name

    def get(self, name: str) -> str | None:
        try:
            return keyring.get_password(self.service_name, name)
        except KeyringError as exc:
            raise SecretStorageError("操作系统安全存储不可用") from exc

    def set(self, name: str, value: str) -> None:
        if not value:
            raise SecretStorageError("不能保存空密钥")
        try:
            keyring.set_password(self.service_name, name, value)
        except KeyringError as exc:
            raise SecretStorageError("无法写入操作系统安全存储") from exc

    def delete(self, name: str) -> None:
        try:
            keyring.delete_password(self.service_name, name)
        except PasswordDeleteError:
            return
        except KeyringError as exc:
            raise SecretStorageError("无法清除操作系统安全存储") from exc


@lru_cache(maxsize=1)
def get_secret_store() -> SecretStore:
    return KeyringSecretStore()


def secret_storage_status() -> dict[str, object]:
    try:
        backend = type(keyring.get_keyring()).__name__
        keyring.get_password(KEYRING_SERVICE, "__crabrag_probe__")
    except KeyringError as exc:
        return {"available": False, "backend": locals().get("backend"), "error_type": type(exc).__name__}
    return {"available": True, "backend": backend, "error_type": None}
