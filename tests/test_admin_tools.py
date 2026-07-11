from __future__ import annotations

import hashlib
import json
import os
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


def test_restore_clears_whitelisted_units_absent_from_backup(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    (source / "config").mkdir(parents=True)
    (source / "config" / ".env").write_text("FROM_BACKUP=1\n", encoding="utf-8")
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    target = tmp_path / "target"
    (target / "data" / "chroma").mkdir(parents=True)
    (target / "data" / "chroma" / "old.db").write_bytes(b"stale")
    (target / "data" / "index").mkdir(parents=True)
    (target / "data" / "index" / "active.json").write_text("{}", encoding="utf-8")
    (target / "data" / "model_api_settings.json").write_text('{"api_key":"stale"}', encoding="utf-8")
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert (target / "config" / ".env").read_text(encoding="utf-8") == "FROM_BACKUP=1\n"
    assert not (target / "data" / "chroma").exists()
    assert not (target / "data" / "index").exists()
    assert not (target / "data" / "model_api_settings.json").exists()


def test_restore_invalid_archive_does_not_create_target_root(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not a zip")
    target = tmp_path / "missing-target"
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    with pytest.raises(crabrag_admin.BackupError, match="invalid"):
        crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert not target.exists()


def test_restore_rejects_undeclared_top_level_member(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(malicious, "w") as output:
        for info in original.infolist():
            output.writestr(info, original.read(info.filename))
        output.writestr("surprise.txt", b"undeclared")
    target = tmp_path / "missing-target"
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    with pytest.raises(crabrag_admin.BackupError, match="undeclared|member"):
        crabrag_admin.restore_backup(target, malicious, assume_yes=True)

    assert not target.exists()


def test_service_running_detects_custom_ports_from_project_run_state(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    (root / "data").mkdir(parents=True)
    (root / "data" / "run.json").write_text(
        json.dumps({"project_root": str(root.resolve()), "web_port": 3103, "api_port": 8101, "processes": [{"pid": 123, "role": "api", "start_identity": "start-1"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(crabrag_admin, "_port_is_open", lambda port: port == 8101)
    monkeypatch.setattr(crabrag_admin, "_run_state_process_matches", lambda _item, _root, _ports: True)
    monkeypatch.setattr(crabrag_admin, "_project_process_detected", lambda _root: False)

    assert crabrag_admin.service_is_running(root) is True


def test_run_scripts_publish_project_scoped_runtime_state():
    powershell = Path("run.ps1").read_text(encoding="utf-8")
    shell = Path("run.sh").read_text(encoding="utf-8")

    assert "data\\run.json" in powershell and "project_root" in powershell
    assert "data/run.json" in shell and "project_root" in shell
    assert 'RUN_STATE_WRITTEN="0"' in shell
    assert 'if [[ "$RUN_STATE_WRITTEN" == "1" ]]' in shell
    assert "start_identity" in powershell and "start_identity" in shell


def test_restore_rejects_symlink_or_reparse_point_in_target_ancestors(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    source = tmp_path / "source"
    (source / "data" / "chroma").mkdir(parents=True)
    (source / "data" / "chroma" / "db").write_bytes(b"db")
    archive = tmp_path / "backup.zip"
    crabrag_admin.create_backup(source, archive)
    target = tmp_path / "target"
    (target / "data").mkdir(parents=True)
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)
    monkeypatch.setattr(crabrag_admin, "_path_is_link_or_reparse", lambda path: path == target / "data")

    with pytest.raises(crabrag_admin.BackupError, match="symlink|reparse|boundary"):
        crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert not (target / "data" / "chroma").exists()


def test_backup_excludes_configured_knowledge_base_nested_inside_state_whitelist(tmp_path: Path):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    internal_docs = root / "data" / "chroma" / "knowledge-docs"
    internal_docs.mkdir(parents=True)
    (internal_docs / "private.txt").write_text("private body", encoding="utf-8")
    (root / "data" / "chroma" / "chroma.sqlite3").write_bytes(b"index")
    (root / "data" / "app_settings.json").write_text(
        json.dumps({"knowledge_base_dirs": [str(internal_docs)]}), encoding="utf-8"
    )
    archive = tmp_path / "backup.zip"

    manifest = crabrag_admin.create_backup(root, archive)

    assert manifest["external_knowledge_base_paths"] == [str(internal_docs.resolve())]
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "payload/data/chroma/chroma.sqlite3" in names
    assert "payload/data/chroma/knowledge-docs/private.txt" not in names


@pytest.mark.parametrize("first,second", [
    ("data/chroma/Foo.db", "data/chroma/foo.db"),
    ("data/chroma/name", "data/chroma/name."),
    ("data/chroma/file", "data/chroma/file "),
    ("data/chroma/con.txt", "data/chroma/other.txt"),
    ("data/chroma/file:stream", "data/chroma/other.txt"),
])
def test_restore_rejects_windows_ambiguous_or_reserved_paths(tmp_path: Path, monkeypatch, first: str, second: str):
    from scripts import crabrag_admin

    archive = tmp_path / "ambiguous.zip"
    files = [(first, b"one"), (second, b"two")]
    manifest = {
        "format_version": 1,
        "software_version": crabrag_admin.SOFTWARE_VERSION,
        "files": [
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
            for path, content in files
        ],
        "external_knowledge_base_paths": [],
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        for path, content in files:
            bundle.writestr(f"payload/{path}", content)
    target = tmp_path / "target"
    monkeypatch.setattr(crabrag_admin, "service_is_running", lambda _root: False)

    with pytest.raises(crabrag_admin.BackupError, match="ambiguous|reserved|unsafe|portable"):
        crabrag_admin.restore_backup(target, archive, assume_yes=True)

    assert not target.exists()


def test_backup_marks_plaintext_secrets_warns_and_applies_owner_only_permissions(tmp_path: Path, monkeypatch, capsys):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    external = tmp_path / "docs"
    external.mkdir()
    _seed_project(root, external)
    archive = tmp_path / "backup.zip"
    def permissions_not_enforced(path):
        path.touch()
        return False

    monkeypatch.setattr(crabrag_admin, "_create_private_archive_file", permissions_not_enforced)
    manifest = crabrag_admin.create_backup(root, archive)

    assert manifest["contains_secrets"] is True
    assert manifest["permissions_enforced"] is False
    monkeypatch.setattr(crabrag_admin, "PROJECT_ROOT", root)
    assert crabrag_admin.main(["backup", "--output", str(tmp_path / "cli-backup.zip")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert "plaintext" in output["warning"].lower()
    assert "not enforced" in output["warning"].lower()


def test_stale_run_state_with_reused_unrelated_pid_does_not_block_restore(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    (root / "data").mkdir(parents=True)
    (root / "data" / "run.json").write_text(
        json.dumps({"project_root": str(root.resolve()), "web_port": 3103, "api_port": 8101, "processes": [{"pid": 123, "role": "api", "start_identity": "old-start"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(crabrag_admin, "_port_is_open", lambda _port: False)
    monkeypatch.setattr(crabrag_admin, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(crabrag_admin, "_run_state_process_matches", lambda _item, _root, _ports: False)
    monkeypatch.setattr(crabrag_admin, "_project_process_detected", lambda _root: False)

    assert crabrag_admin.service_is_running(root) is False


def test_global_bun_web_process_is_trusted_by_identity_role_and_owned_custom_port(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    item = {"pid": 456, "role": "web", "start_identity": "start-web"}
    monkeypatch.setattr(crabrag_admin, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        crabrag_admin,
        "_process_runtime_info",
        lambda _pid: {
            "command_line": "C:/Program Files/Bun/bun.exe server/gateway.js",
            "executable": "C:/Program Files/Bun/bun.exe",
            "start_identity": "start-web",
            "ports": [3103],
        },
    )

    assert crabrag_admin._run_state_process_matches(item, root, [3103, 8101]) is True


def test_backup_rejects_reparse_point_in_protected_source_ancestor(tmp_path: Path, monkeypatch):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    (root / "data" / "chroma").mkdir(parents=True)
    (root / "data" / "chroma" / "state.db").write_bytes(b"state")
    monkeypatch.setattr(crabrag_admin, "_path_is_link_or_reparse", lambda path: path == root / "data")
    archive = tmp_path / "backup.zip"

    with pytest.raises(crabrag_admin.BackupError, match="symlink|reparse|escape"):
        crabrag_admin.create_backup(root, archive)

    assert not archive.exists()


@pytest.mark.parametrize("configured", [".", "data/chroma"])
def test_backup_rejects_knowledge_base_that_contains_or_equals_protected_state(tmp_path: Path, configured: str):
    from scripts import crabrag_admin

    root = tmp_path / "CrabRAG"
    (root / "data" / "chroma").mkdir(parents=True)
    docs_root = (root / configured).resolve()
    (root / "data" / "app_settings.json").write_text(
        json.dumps({"knowledge_base_dirs": [str(docs_root)]}), encoding="utf-8"
    )

    with pytest.raises(crabrag_admin.BackupError, match="overlap|protected"):
        crabrag_admin.create_backup(root, tmp_path / "backup.zip")
