from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.rag_api.audit import AuditIntegrityError, AuditLog
from services.rag_api.security import PrincipalContext


def test_verifying_empty_audit_does_not_create_runtime_files(tmp_path):
    audit = AuditLog(tmp_path / "security-audit.jsonl")

    assert audit.verify()["record_count"] == 0
    assert not audit.lock_path.exists()


def test_audit_log_appends_hash_chain_and_verifies(tmp_path):
    audit = AuditLog(tmp_path / "security-audit.jsonl")
    principal = PrincipalContext("alice", ("admin",), (), "42", True)

    first = audit.append("authentication.accepted", principal=principal, details={"provider": "oidc"})
    second = audit.append("permission.resolved", principal=principal, details={"allowed_count": 3})

    assert first["sequence"] == 1
    assert first["previous_hash"] == "0" * 64
    assert second["sequence"] == 2
    assert second["previous_hash"] == first["record_hash"]
    result = audit.verify()
    assert result == {"valid": True, "record_count": 2, "last_hash": second["record_hash"]}


def test_audit_log_serializes_concurrent_appends(tmp_path):
    audit = AuditLog(tmp_path / "security-audit.jsonl")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: audit.append("concurrent", details={"index": index}), range(40)))

    records = [json.loads(line) for line in audit.path.read_text(encoding="utf-8").splitlines()]
    assert [record["sequence"] for record in records] == list(range(1, 41))
    assert audit.verify()["record_count"] == 40


@pytest.mark.parametrize("mutation", ["modify", "delete", "insert", "reorder", "truncate"])
def test_audit_verify_detects_tampering_and_truncation(tmp_path, mutation):
    audit = AuditLog(tmp_path / "security-audit.jsonl")
    for index in range(4):
        audit.append("event", details={"index": index})
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    if mutation == "modify":
        payload = json.loads(lines[1])
        payload["details"]["index"] = 999
        lines[1] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    elif mutation == "delete":
        del lines[1]
    elif mutation == "insert":
        lines.insert(1, lines[0])
    elif mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines.pop()
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError):
        audit.verify()


def test_audit_log_redacts_sensitive_details(tmp_path):
    audit = AuditLog(tmp_path / "security-audit.jsonl")

    audit.append("settings.updated", details={
        "api_key": "api-secret",
        "question": "private question",
        "document_content": "private body",
        "nested": {"access_token": "token-secret", "safe": "kept"},
    })

    text = audit.path.read_text(encoding="utf-8")
    for secret in ("api-secret", "private question", "private body", "token-secret"):
        assert secret not in text
    record = json.loads(text)
    assert record["details"]["nested"]["safe"] == "kept"


def test_audit_verify_rejects_missing_log_when_anchor_exists(tmp_path):
    audit = AuditLog(tmp_path / "security-audit.jsonl")
    audit.append("event")
    audit.path.unlink()

    with pytest.raises(AuditIntegrityError):
        audit.verify()


def test_api_authentication_and_permission_events_are_audited_without_query_text(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import audit, main
    from services.rag_api.security import PrincipalContext, RetrievalContext

    principal = PrincipalContext.anonymous()
    monkeypatch.setattr(main, "get_identity_provider", lambda: type(
        "Provider", (), {"authenticate": lambda self, headers: principal}
    )())
    monkeypatch.setattr(
        main,
        "build_retrieval_context",
        lambda value: RetrievalContext("gen-3", value, frozenset({"doc-a"}), "permission-1"),
    )
    monkeypatch.setattr(main, "load_kb_categories", lambda: {"categories": []})

    response = TestClient(main.app).get("/api/categories")

    assert response.status_code == 200
    text = audit.SECURITY_AUDIT.path.read_text(encoding="utf-8")
    assert "authentication.accepted" in text
    assert "permission.resolved" in text
    assert "doc-a" not in text


def test_api_fails_closed_when_security_audit_is_unavailable(monkeypatch):
    from fastapi.testclient import TestClient

    from services.rag_api import audit, main

    monkeypatch.setattr(
        audit.SECURITY_AUDIT,
        "append",
        lambda *args, **kwargs: (_ for _ in ()).throw(audit.AuditWriteError("disk full")),
    )

    response = TestClient(main.app, raise_server_exceptions=False).get("/api/categories")

    assert response.status_code == 503
    assert "安全审计不可用" in response.json()["detail"]
