from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest


def _seed_project(root: Path, external_docs: Path) -> None:
    (root / "config").mkdir(parents=True)
    (root / "config" / ".env").write_text("CRABRAG_API_KEY=backup-secret\n", encoding="utf-8")
    (root / "data" / "chroma").mkdir(parents=True)
    (root / "data" / "chroma" / "chroma.sqlite3").write_bytes(b"chroma-state")
    (root / "data" / "index" / "generations" / "gen-1").mkdir(parents=True)
    (root / "data" / "index" / "active.json").write_text(
        '{"schema_version": 1, "active_generation": "gen-1"}', encoding="utf-8"
    )
    (root / "data" / "index" / "generations" / "gen-1" / "manifest.json").write_text(
        '{"generation_id": "gen-1"}', encoding="utf-8"
    )
    (root / "data" / "ui").mkdir(parents=True)
    (root / "data" / "ui" / "sidebar-image.json").write_text("{}", encoding="utf-8")
    (root / "data" / "app_settings.json").write_text(
        json.dumps({"knowledge_base_dirs": [str(external_docs)]}), encoding="utf-8"
    )
    (root / "data" / "rag_settings.json").write_text("{}", encoding="utf-8")
    (root / "data" / "model_api_settings.json").write_text(
        '{"api_key": "stored-secret"}', encoding="utf-8"
    )


def test_doctor_clean_install_reports_warnings_without_exposing_secrets(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    root.mkdir()
    (root / "VERSION").write_text("1.1.0\n", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / ".env").write_text("CRABRAG_API_KEY=do-not-print\n", encoding="utf-8")
    monkeypatch.setattr(crabrag_admin, "_port_is_open", lambda _port: False)
    monkeypatch.setattr(crabrag_admin, "_bun_version", lambda _root: None)

    report, exit_code = crabrag_admin.doctor(root)

    assert report["software_version"] == "1.1.0"
    assert report["summary"]["error"] == 0
    assert report["summary"]["warning"] > 0
    assert exit_code == crabrag_admin.EXIT_WARNING
    assert "do-not-print" not in json.dumps(report)
    assert {item["name"] for item in report["checks"]} >= {
        "python", "platform", "configuration", "knowledge_base", "chroma", "generation",
        "service", "bun", "generated_assets", "remote_models", "local_models",
    }


def test_backup_contains_governed_state_and_only_records_external_document_paths(tmp_path: Path):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    external_docs = tmp_path / "private-docs"
    external_docs.mkdir()
    (external_docs / "customer.txt").write_text("private customer content", encoding="utf-8")
    _seed_project(root, external_docs)
    archive = tmp_path / "backup.zip"

    manifest = crabrag_admin.create_backup(root, archive)

    assert archive.is_file()
    assert manifest["format_version"] == 1
    assert manifest["software_version"] == crabrag_admin.SOFTWARE_VERSION
    assert manifest["external_knowledge_base_paths"] == [str(external_docs.resolve())]
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert "manifest.json" in names
        assert "payload/config/.env" in names
        assert "payload/data/chroma/chroma.sqlite3" in names
        assert "payload/data/index/active.json" in names
        assert "payload/data/ui/sidebar-image.json" in names
        assert "payload/data/app_settings.json" in names
        assert not any("customer.txt" in name for name in names)
        stored_manifest = json.loads(bundle.read("manifest.json"))
        for item in stored_manifest["files"]:
            assert hashlib.sha256(bundle.read(f"payload/{item['path']}")).hexdigest() == item["sha256"]


def test_restore_rejects_checksum_tamper_before_mutating_target(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    external = tmp_path / "external"
    external.mkdir()
    _seed_project(source, external)
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(tampered, "w") as output:
        for info in original.infolist():
            content = original.read(info.filename)
            if info.filename == "payload/data/rag_settings.json":
                content = b'{"tampered": true}'
            output.writestr(info, content)

    target = tmp_path / "target"
    (target / "data").mkdir(parents=True)
    sentinel = target / "data" / "rag_settings.json"
    sentinel.write_text('{"before": true}', encoding="utf-8")
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    with pytest.raises(crabrag_admin.BackupError, match="checksum"):
        crabrag_admin.restore_backup(target, tampered, assume_yes=True)

    assert sentinel.read_text(encoding="utf-8") == '{"before": true}'


def test_restore_rejects_zip_slip_before_mutating_target(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    archive = tmp_path / "malicious.zip"
    payload = b"owned"
    manifest = {
        "format_version": 1,
        "software_version": crabrag_admin.SOFTWARE_VERSION,
        "files": [{"path": "../escaped.txt", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}],
        "external_knowledge_base_paths": [],
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("payload/../escaped.txt", payload)
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    with pytest.raises(crabrag_admin.BackupError, match="unsafe|path"):
        crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert not (tmp_path / "escaped.txt").exists()
    assert list(target.iterdir()) == []


def test_restore_requires_stopped_service_and_confirmation(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    external = tmp_path / "external"
    external.mkdir()
    _seed_project(source, external)
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: True)
    with pytest.raises(crabrag_admin.BackupError, match="running"):
        crabrag_admin.restore_backup(target, archive, assume_yes=True)

    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)
    with pytest.raises(crabrag_admin.BackupError, match="cancelled"):
        crabrag_admin.restore_backup(target, archive, assume_yes=False, confirm=lambda _prompt: "no")


def test_restore_successfully_replaces_state_after_complete_validation(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    external = tmp_path / "external"
    external.mkdir()
    _seed_project(source, external)
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    target = tmp_path / "target"
    (target / "data" / "chroma").mkdir(parents=True)
    (target / "data" / "chroma" / "old.db").write_bytes(b"old")
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    result = crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert result["restored_files"] == len(result["manifest"]["files"])
    assert (target / "data" / "chroma" / "chroma.sqlite3").read_bytes() == b"chroma-state"
    assert not (target / "data" / "chroma" / "old.db").exists()
    assert (target / "data" / "index" / "active.json").is_file()


def test_restore_recovers_current_unit_if_atomic_replace_is_interrupted(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    external = tmp_path / "external"
    external.mkdir()
    _seed_project(source, external)
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    target = tmp_path / "target"
    (target / "config").mkdir(parents=True)
    old_env = target / "config" / ".env"
    old_env.write_text("ORIGINAL=1\n", encoding="utf-8")
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)
    real_replace = crabrag_admin.os.replace

    def interrupted_replace(source_path, target_path):
        if Path(source_path).name == "stage-0":
            raise OSError("simulated interruption")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(crabrag_admin.os, "replace", interrupted_replace)

    with pytest.raises(crabrag_admin.BackupError, match="recovered"):
        crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert old_env.read_text(encoding="utf-8") == "ORIGINAL=1\n"
