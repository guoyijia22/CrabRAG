from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Callable, Iterable
import uuid
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = PROJECT_ROOT / "VERSION"
SOFTWARE_VERSION = VERSION_PATH.read_text(encoding="utf-8").strip() if VERSION_PATH.exists() else "0.0.0"
BACKUP_FORMAT_VERSION = 1
EXIT_OK = 0
EXIT_WARNING = 1
EXIT_ERROR = 2
SERVICE_PORTS = (3003, 8001)

_BACKUP_DIRECTORIES = (
    "config",
    "data/chroma",
    "data/index",
    "data/ui",
)
_BACKUP_FILES = (
    "data/app_settings.json",
    "data/model_api_settings.json",
    "data/rag_settings.json",
)
_RESTORE_UNITS = (*_BACKUP_DIRECTORIES, *_BACKUP_FILES)


class BackupError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _check(name: str, status: str, message: str, **details: object) -> dict[str, object]:
    return {"name": name, "status": status, "message": message, **details}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _bun_version(root: Path) -> str | None:
    candidates = [root / "runtime" / "bun" / ("bun.exe" if os.name == "nt" else "bun")]
    command = shutil.which("bun")
    if command:
        candidates.append(Path(command))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "--version"], capture_output=True, text=True, timeout=5, check=False
            )
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _configured_document_paths(root: Path) -> list[Path]:
    settings = _read_json(root / "data" / "app_settings.json")
    values = settings.get("knowledge_base_dirs")
    paths: list[Path] = []
    if isinstance(values, list):
        for value in values:
            if isinstance(value, str) and value.strip():
                path = Path(value.strip()).expanduser()
                if not path.is_absolute():
                    path = root / path
                paths.append(path.resolve())
    return list(dict.fromkeys(paths))


def _local_model_doctor(root: Path) -> tuple[str, str, dict[str, object]]:
    models_root = root / "runtime" / "models"
    expected = (
        "Qwen3___5-0___8B-ONNX",
        "Qwen3-Embedding-0___6B-ONNX",
        "Qwen3-Reranker-0___6B-ONNX",
    )
    present = [name for name in expected if (models_root / name).is_dir()]
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import onnxruntime"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        runtime_available = probe.returncode == 0
        runtime_error = None if runtime_available else "OnnxRuntimeInitializationError"
    except (OSError, subprocess.TimeoutExpired) as exc:
        runtime_available = False
        runtime_error = type(exc).__name__
    available = runtime_available and len(present) == len(expected)
    status = "ok" if available else "warning"
    message = "local model capability is available" if available else "local model capability is unavailable"
    return status, message, {
        "available": available,
        "runtime_available": runtime_available,
        "runtime_error_type": runtime_error,
        "models_present": present,
        "models_expected": list(expected),
    }


def doctor(project_root: Path = PROJECT_ROOT) -> tuple[dict[str, object], int]:
    root = project_root.resolve()
    checks: list[dict[str, object]] = []
    python_ok = sys.version_info >= (3, 10)
    checks.append(_check("python", "ok" if python_ok else "error", platform.python_version()))
    checks.append(_check("platform", "ok", platform.platform(), machine=platform.machine()))

    env_path = root / "config" / ".env"
    checks.append(
        _check(
            "configuration",
            "ok" if env_path.is_file() else "warning",
            "configuration file found" if env_path.is_file() else "config/.env has not been created",
        )
    )
    docs = _configured_document_paths(root)
    existing_docs = [str(path) for path in docs if path.is_dir()]
    docs_status = "ok" if existing_docs else "warning"
    checks.append(
        _check(
            "knowledge_base",
            docs_status,
            "knowledge base directories are available" if existing_docs else "no available knowledge base directory",
            configured_count=len(docs),
            available_count=len(existing_docs),
        )
    )
    chroma = root / "data" / "chroma"
    checks.append(
        _check("chroma", "ok" if chroma.is_dir() else "warning", "Chroma state found" if chroma.is_dir() else "Chroma state not initialized")
    )
    active_path = root / "data" / "index" / "active.json"
    if not active_path.exists():
        checks.append(_check("generation", "warning", "index generation is not initialized"))
    else:
        active = _read_json(active_path)
        generation_id = str(active.get("active_generation") or "")
        manifest = root / "data" / "index" / "generations" / generation_id / "manifest.json"
        valid = bool(generation_id and manifest.is_file())
        checks.append(
            _check(
                "generation",
                "ok" if valid else "error",
                "active generation is available" if valid else "active generation pointer is invalid",
                active_generation=generation_id or None,
            )
        )
    open_ports = [port for port in SERVICE_PORTS if _port_is_open(port)]
    checks.append(
        _check(
            "service",
            "ok" if len(open_ports) == len(SERVICE_PORTS) else "warning",
            "service is running" if len(open_ports) == len(SERVICE_PORTS) else "service is stopped or partially available",
            open_ports=open_ports,
        )
    )
    bun_version = _bun_version(root)
    checks.append(
        _check(
            "bun",
            "ok" if bun_version == "1.3.14" else "warning",
            f"Bun {bun_version}" if bun_version else "Bun is unavailable",
            version=bun_version,
        )
    )
    generated = [root / "apps" / "web" / "dist" / "index.html", root / "server" / "gateway.js"]
    generated_ok = all(path.is_file() for path in generated)
    checks.append(
        _check(
            "generated_assets",
            "ok" if generated_ok else "warning",
            "generated assets are available" if generated_ok else "generated assets are incomplete",
        )
    )
    model_settings = _read_json(root / "data" / "model_api_settings.json")
    remote_configured = any(
        bool(model_settings.get(key)) for key in ("api_key", "embedding_api_key", "rerank_api_key")
    ) or any(bool(os.getenv(key)) for key in ("CRABRAG_API_KEY", "OPENAI_API_KEY", "SILICONFLOW_API_KEY"))
    checks.append(
        _check(
            "remote_models",
            "ok" if remote_configured else "warning",
            "remote model credentials are configured" if remote_configured else "remote model credentials are not configured",
            configured=remote_configured,
        )
    )
    local_status, local_message, local_details = _local_model_doctor(root)
    checks.append(_check("local_models", local_status, local_message, **local_details))

    summary = {status: sum(item["status"] == status for item in checks) for status in ("ok", "warning", "error")}
    exit_code = EXIT_ERROR if summary["error"] else EXIT_WARNING if summary["warning"] else EXIT_OK
    return {
        "software_version": _version_for_root(root),
        "checked_at": _now(),
        "project_root": str(root),
        "summary": summary,
        "checks": checks,
    }, exit_code


def _version_for_root(root: Path) -> str:
    try:
        return (root / "VERSION").read_text(encoding="utf-8").strip() or SOFTWARE_VERSION
    except OSError:
        return SOFTWARE_VERSION


def _iter_backup_files(root: Path) -> Iterable[tuple[str, Path]]:
    for relative in _BACKUP_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink():
                resolved = path.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                yield resolved.relative_to(root).as_posix(), resolved
    for relative in _BACKUP_FILES:
        path = root / relative
        if path.is_file() and not path.is_symlink():
            yield relative, path.resolve()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def create_backup(project_root: Path, output: Path) -> dict[str, object]:
    root = project_root.resolve()
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        raise BackupError("backup output must be a ZIP file")
    entries: list[tuple[str, bytes]] = []
    file_manifest: list[dict[str, object]] = []
    for relative, path in _iter_backup_files(root):
        content = path.read_bytes()
        entries.append((relative, content))
        file_manifest.append({"path": relative, "sha256": _sha256_bytes(content), "size": len(content)})
    manifest: dict[str, object] = {
        "format_version": BACKUP_FORMAT_VERSION,
        "software_version": _version_for_root(root),
        "created_at": _now(),
        "files": file_manifest,
        "external_knowledge_base_paths": [str(path) for path in _configured_document_paths(root)],
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                for relative, content in entries:
                    bundle.writestr(f"payload/{relative}", content)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return manifest


def _safe_payload_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BackupError("unsafe backup path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
        raise BackupError("unsafe backup path")
    normalized = pure.as_posix()
    if not _is_allowed_restore_path(normalized):
        raise BackupError(f"backup path is not allowed: {normalized}")
    return normalized


def _is_allowed_restore_path(path: str) -> bool:
    return any(path == item or path.startswith(f"{item}/") for item in _RESTORE_UNITS)


def _version_tuple(value: object) -> tuple[int, int, int]:
    try:
        parts = str(value).split(".")
        if len(parts) != 3:
            raise ValueError
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise BackupError("backup software version is invalid") from exc


def _validate_archive(archive: Path, extract_root: Path) -> dict[str, object]:
    try:
        bundle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BackupError("backup archive is invalid") from exc
    with bundle:
        names = [info.filename for info in bundle.infolist()]
        for name in names:
            if "\\" in name:
                raise BackupError("unsafe ZIP path")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise BackupError("unsafe ZIP path")
        if names.count("manifest.json") != 1:
            raise BackupError("backup manifest is missing or duplicated")
        try:
            manifest = json.loads(bundle.read("manifest.json"))
        except (json.JSONDecodeError, KeyError) as exc:
            raise BackupError("backup manifest is invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != BACKUP_FORMAT_VERSION:
            raise BackupError("backup format version is incompatible")
        archive_version = _version_tuple(manifest.get("software_version"))
        current_version = _version_tuple(SOFTWARE_VERSION)
        if archive_version[0] != current_version[0] or archive_version > current_version:
            raise BackupError("backup software version is incompatible")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise BackupError("backup file manifest is invalid")
        declared: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise BackupError("backup file manifest is invalid")
            relative = _safe_payload_path(item.get("path"))
            if relative in declared:
                raise BackupError("backup file path is duplicated")
            declared.add(relative)
            member = f"payload/{relative}"
            if names.count(member) != 1:
                raise BackupError(f"backup payload is missing or duplicated: {relative}")
            content = bundle.read(member)
            if _sha256_bytes(content) != item.get("sha256") or len(content) != item.get("size"):
                raise BackupError(f"backup checksum mismatch: {relative}")
            target = extract_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        actual_payload = {name.removeprefix("payload/") for name in names if name.startswith("payload/") and not name.endswith("/")}
        if actual_payload != declared:
            raise BackupError("backup contains undeclared payload files")
        return manifest


def service_is_running(project_root: Path) -> bool:
    del project_root
    return any(_port_is_open(port) for port in SERVICE_PORTS)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def restore_backup(
    project_root: Path,
    archive: Path,
    *,
    assume_yes: bool = False,
    confirm: Callable[[str], str] = input,
) -> dict[str, object]:
    root = project_root.resolve()
    archive_path = archive.expanduser().resolve()
    if service_is_running(root):
        raise BackupError("CrabRAG service is running; stop it before restore")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crabrag-restore-", dir=root.parent) as temporary_dir:
        extracted = Path(temporary_dir) / "validated"
        extracted.mkdir()
        manifest = _validate_archive(archive_path, extracted)
        if not assume_yes and confirm("Restore will replace CrabRAG configuration and index state. Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            raise BackupError("restore cancelled")

        transaction = Path(temporary_dir) / "transaction"
        transaction.mkdir()
        prepared: list[tuple[Path, Path, Path]] = []
        for index, relative in enumerate(_RESTORE_UNITS):
            source = extracted / relative
            if not source.exists():
                continue
            target = root / relative
            stage = transaction / f"stage-{index}"
            rollback = transaction / f"rollback-{index}"
            if source.is_dir():
                shutil.copytree(source, stage)
            else:
                stage.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, stage)
            prepared.append((target, stage, rollback))

        applied: list[tuple[Path, Path]] = []
        try:
            for target, stage, rollback in prepared:
                target.parent.mkdir(parents=True, exist_ok=True)
                moved_existing = False
                if target.exists():
                    os.replace(target, rollback)
                    moved_existing = True
                try:
                    os.replace(stage, target)
                except Exception:
                    if moved_existing and rollback.exists():
                        os.replace(rollback, target)
                    raise
                applied.append((target, rollback))
        except Exception as exc:  # noqa: BLE001 - rollback every already replaced unit
            for target, rollback in reversed(applied):
                if target.exists():
                    _remove_path(target)
                if rollback.exists():
                    os.replace(rollback, target)
            raise BackupError("restore failed; previous state was recovered") from exc
        return {"manifest": manifest, "restored_files": len(manifest["files"])}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrabRAG administration commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="check local installation health")
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    backup_parser = subparsers.add_parser("backup", help="create a verified state backup")
    backup_parser.add_argument("--output", type=Path, required=True)
    restore_parser = subparsers.add_parser("restore", help="restore a verified state backup")
    restore_parser.add_argument("--archive", type=Path, required=True)
    restore_parser.add_argument("--yes", action="store_true", dest="assume_yes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report, exit_code = doctor(PROJECT_ROOT)
            if args.as_json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                for item in report["checks"]:
                    print(f"[{str(item['status']).upper()}] {item['name']}: {item['message']}")
            return exit_code
        if args.command == "backup":
            manifest = create_backup(PROJECT_ROOT, args.output)
            print(json.dumps({"status": "ok", "output": str(args.output.resolve()), "manifest": manifest}, ensure_ascii=False))
            return EXIT_OK
        result = restore_backup(PROJECT_ROOT, args.archive, assume_yes=args.assume_yes)
        print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
        return EXIT_OK
    except BackupError as exc:
        print(f"CrabRAG admin error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
