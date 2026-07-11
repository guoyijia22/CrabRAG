from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_security_audit(tmp_path, monkeypatch):
    from services.rag_api import audit

    monkeypatch.setattr(audit, "SECURITY_AUDIT", audit.AuditLog(tmp_path / "audit" / "security-audit.jsonl"))
