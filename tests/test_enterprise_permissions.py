from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests

from services.rag_api.security import (
    HttpPermissionProvider,
    PermissionServiceError,
    PrincipalContext,
)


class FakeResponse:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        return self.response


def _principal() -> PrincipalContext:
    return PrincipalContext(
        subject="alice",
        roles=("reviewer",),
        groups=("north",),
        permission_revision="42",
    )


def _manifest() -> dict:
    return {
        "generation_id": "gen-3",
        "knowledge_base_id": "kb-1",
        "documents": {
            "doc-a": {"content": "private body"},
            "doc-b": {"content": "another body"},
        },
    }


def test_http_permission_provider_sends_only_subject_and_generation_summary():
    session = FakeSession(FakeResponse({
        "allowed_document_ids": ["doc-b", "doc-a", "doc-a"],
        "permission_revision": "42",
    }))
    provider = HttpPermissionProvider(
        "https://permissions.example.test/check",
        timeout_seconds=2.5,
        session=session,
    )

    allowed = provider.allowed_document_ids(_principal(), _manifest())

    assert allowed == frozenset({"doc-a", "doc-b"})
    assert session.calls == [{
        "url": "https://permissions.example.test/check",
        "json": {
            "subject": "alice",
            "roles": ["reviewer"],
            "groups": ["north"],
            "permission_revision": "42",
            "generation": {
                "generation_id": "gen-3",
                "knowledge_base_id": "kb-1",
                "document_count": 2,
            },
        },
        "timeout": 2.5,
    }]
    assert "private body" not in str(session.calls)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, status_code=503),
        FakeResponse([]),
        FakeResponse({"allowed_document_ids": "doc-a", "permission_revision": "42"}),
        FakeResponse({"allowed_document_ids": ["doc-a"], "permission_revision": "41"}),
        FakeResponse({"allowed_document_ids": ["doc-a"]}),
        FakeResponse({"allowed_document_ids": ["doc-outside"], "permission_revision": "42"}),
    ],
)
def test_http_permission_provider_fails_closed_on_invalid_response(response):
    provider = HttpPermissionProvider(
        "https://permissions.example.test/check",
        session=FakeSession(response),
    )

    with pytest.raises(PermissionServiceError):
        provider.allowed_document_ids(_principal(), _manifest())


def test_http_permission_provider_fails_closed_on_timeout():
    provider = HttpPermissionProvider(
        "https://permissions.example.test/check",
        session=FakeSession(error=requests.Timeout("slow")),
    )

    with pytest.raises(PermissionServiceError):
        provider.allowed_document_ids(_principal(), _manifest())
