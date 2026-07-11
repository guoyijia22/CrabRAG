from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def test_load_manifest_auto_registers_public_document_with_persistent_warning(tmp_path: Path):
    from services.rag_api.document.manifest import load_or_create_manifest

    document_path = tmp_path / "policy.txt"
    document_path.write_text("policy", encoding="utf-8")
    now = datetime(2026, 7, 11, 1, 2, 3, tzinfo=timezone.utc)

    manifest = load_or_create_manifest(tmp_path, [document_path], now=now)

    assert manifest["schema_version"] == 1
    assert manifest["knowledge_base_id"].startswith("kb-")
    assert manifest["documents"] == [
        {
            "document_id": manifest["documents"][0]["document_id"],
            "path": "policy.txt",
            "version": "1",
            "status": "published",
            "effective_at": "2026-07-11T01:02:03Z",
            "updated_at": "2026-07-11T01:02:03Z",
            "acl": {
                "visibility": "public",
                "users": [],
                "roles": [],
                "groups": [],
                "policy_ref": "",
                "revision": "1",
            },
        }
    ]
    assert manifest["audit_warnings"][0]["code"] == "AUTO_PUBLIC_DOCUMENT"
    persisted = json.loads((tmp_path / ".crabrag-manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_select_active_versions_uses_latest_published_effective_version(tmp_path: Path):
    from services.rag_api.document.manifest import select_active_versions

    for name in ("policy-v1.txt", "policy-v2.txt", "policy-draft.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "knowledge_base_id": "kb-test",
        "documents": [
            _entry("policy", "policy-v1.txt", "1", "2026-01-01T00:00:00Z"),
            _entry("policy", "policy-v2.txt", "2", "2026-07-01T00:00:00Z"),
            _entry("policy", "policy-draft.txt", "3", "2026-06-01T00:00:00Z", status="draft"),
        ],
    }

    active = select_active_versions(
        manifest,
        tmp_path,
        cutoff=datetime(2026, 7, 11, tzinfo=timezone.utc),
    )

    assert [(item["document_id"], item["version"], item["path"].name) for item in active] == [
        ("policy", "2", "policy-v2.txt")
    ]


def test_select_active_versions_rejects_equal_effective_time_conflict(tmp_path: Path):
    from services.rag_api.document.manifest import ManifestError, select_active_versions

    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "knowledge_base_id": "kb-test",
        "documents": [
            _entry("policy", "a.txt", "1", "2026-01-01T00:00:00Z"),
            _entry("policy", "b.txt", "2", "2026-01-01T00:00:00Z"),
        ],
    }

    with pytest.raises(ManifestError, match="相同生效时间"):
        select_active_versions(manifest, tmp_path, cutoff=datetime(2026, 7, 11, tzinfo=timezone.utc))


def test_load_active_catalog_returns_only_effective_version_and_next_activation(tmp_path: Path):
    from services.rag_api.document.manifest import load_active_catalog

    version_one = tmp_path / "policy-v1.txt"
    version_two = tmp_path / "policy-v2.txt"
    version_one.write_text("old", encoding="utf-8")
    version_two.write_text("new", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "knowledge_base_id": "kb-test",
        "documents": [
            _entry("policy", "policy-v1.txt", "1", "2026-01-01T00:00:00Z"),
            _entry("policy", "policy-v2.txt", "2", "2026-08-01T00:00:00Z"),
        ],
        "audit_warnings": [],
    }
    (tmp_path / ".crabrag-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    catalog = load_active_catalog(
        [tmp_path],
        [version_one, version_two],
        cutoff=datetime(2026, 7, 11, tzinfo=timezone.utc),
    )

    assert [(item["document_id"], item["version"], item["path"]) for item in catalog["documents"]] == [
        ("policy", "1", version_one.resolve())
    ]
    assert catalog["next_activation_at"] == "2026-08-01T00:00:00Z"
    assert catalog["warnings"] == []


def test_retired_version_deactivates_older_published_version(tmp_path: Path):
    from services.rag_api.document.manifest import select_active_versions

    path = tmp_path / "policy.txt"
    path.write_text("policy", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "knowledge_base_id": "kb-test",
        "documents": [
            _entry("policy", "policy.txt", "1", "2026-01-01T00:00:00Z"),
            _entry("policy", "policy.txt", "2", "2026-07-01T00:00:00Z", status="retired"),
        ],
    }

    active = select_active_versions(manifest, tmp_path, cutoff=datetime(2026, 7, 11, tzinfo=timezone.utc))

    assert active == []


def test_future_retirement_is_reported_as_next_activation(tmp_path: Path):
    from services.rag_api.document.manifest import load_active_catalog

    path = tmp_path / "policy.txt"
    path.write_text("policy", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "knowledge_base_id": "kb-test",
        "documents": [
            _entry("policy", "policy.txt", "1", "2026-01-01T00:00:00Z"),
            _entry("policy", "policy.txt", "2", "2026-08-01T00:00:00Z", status="retired"),
        ],
        "audit_warnings": [],
    }
    (tmp_path / ".crabrag-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    catalog = load_active_catalog([tmp_path], [path], cutoff=datetime(2026, 7, 11, tzinfo=timezone.utc))

    assert catalog["next_activation_at"] == "2026-08-01T00:00:00Z"


def _entry(document_id: str, path: str, version: str, effective_at: str, *, status: str = "published") -> dict:
    return {
        "document_id": document_id,
        "path": path,
        "version": version,
        "status": status,
        "effective_at": effective_at,
        "updated_at": effective_at,
        "acl": {"visibility": "public", "users": [], "roles": [], "groups": [], "policy_ref": "", "revision": "1"},
    }
