from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile


class ReleaseError(RuntimeError):
    pass


REQUIRED_FILES = (
    "VERSION",
    "LICENSE",
    "CHANGELOG.md",
    "README.md",
    "README_ZH.md",
    "README_PORTABLE.md",
    "requirements.txt",
    "package.json",
    "bun.lock",
    "install.ps1",
    "install.sh",
    "run.ps1",
    "run.sh",
    "start.bat",
    "stop.bat",
    "crab-rag.bat",
    "crabrag.skill",
    "server/gateway.js",
    "apps/web/dist/index.html",
    "scripts/crabrag_admin.py",
    "scripts/check_env.py",
    "scripts/stop.ps1",
    "config/.env.example",
    "config/evaluation-dataset.example.json",
)
INCLUDED_DIRECTORIES = (
    "services",
    "apps/web/dist",
    "skills/crabrag-rag",
)
INCLUDED_FILES = (
    *REQUIRED_FILES,
)
FORBIDDEN_PARTS = {
    ".git",
    ".github",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "tests",
    "logs",
    "data",
    "models",
    "bun_api",
}
FORBIDDEN_PREFIXES = ("apps/web/src/", "server/bun_api/", "runtime/models/")
TEXT_SUFFIXES = {
    ".bat", ".css", ".env", ".html", ".js", ".json", ".md", ".mjs", ".ps1", ".py", ".sh", ".txt", ".xml", ".yaml", ".yml",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"']?sk-[A-Za-z0-9_-]{12,}"),
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_allowed(relative: str) -> bool:
    path = Path(relative)
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False
    if relative in {"config/.env", ".env"} or relative.startswith("config/.env.") and relative != "config/.env.example":
        return False
    return not relative.startswith(FORBIDDEN_PREFIXES)


def _iter_release_files(root: Path):
    seen: set[str] = set()
    for relative in INCLUDED_FILES:
        path = root / relative
        if path.is_file() and not path.is_symlink() and _is_allowed(relative):
            seen.add(relative)
            yield relative, path
    for directory_name in INCLUDED_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in seen or not _is_allowed(relative):
                continue
            seen.add(relative)
            yield relative, path


def _check_secrets(relative: str, content: bytes) -> None:
    if Path(relative).suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ReleaseError(f"secret-like content detected in release file: {relative}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _validate_release(bundle_path: Path, expected_files: set[str]) -> None:
    with zipfile.ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
        for required in REQUIRED_FILES:
            if f"CrabRAG/{required}" not in names:
                raise ReleaseError(f"required release file is missing: {required}")
        actual_files = {name.removeprefix("CrabRAG/") for name in names if name.startswith("CrabRAG/") and not name.endswith("/")}
        if actual_files != expected_files | {"release-manifest.json"}:
            raise ReleaseError("release archive contents differ from the validated file list")
        for relative in actual_files:
            if relative != "release-manifest.json" and not _is_allowed(relative):
                raise ReleaseError(f"forbidden release content: {relative}")


def build_release(project_root: Path, output_dir: Path, version: str) -> tuple[Path, Path]:
    root = project_root.resolve()
    expected_version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if version != expected_version:
        raise ReleaseError(f"requested version {version} does not match VERSION {expected_version}")
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise ReleaseError(f"required release files are missing: {', '.join(missing)}")
    entries: list[tuple[str, bytes]] = []
    manifest_files: list[dict[str, object]] = []
    for relative, path in _iter_release_files(root):
        content = path.read_bytes()
        _check_secrets(relative, content)
        entries.append((relative, content))
        manifest_files.append({"path": relative, "sha256": _sha256(content), "size": len(content)})
    manifest = {
        "format_version": 1,
        "software_version": version,
        "target": "windows-x64",
        "runtime": {"python": ">=3.10", "bun": "1.3.14"},
        "files": manifest_files,
    }
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"CrabRAG-v{version}-windows-x64.zip"
    checksum = destination / f"{archive.name}.sha256"
    with tempfile.NamedTemporaryFile(prefix=f".{archive.name}.", suffix=".tmp", dir=destination, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as bundle:
            for relative, content in entries:
                bundle.writestr(_zip_info(f"CrabRAG/{relative}"), content)
            bundle.writestr(
                _zip_info("CrabRAG/release-manifest.json"),
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
            )
        _validate_release(temporary, {relative for relative, _content in entries})
        os.replace(temporary, archive)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    digest = _sha256(archive.read_bytes())
    checksum_temporary = checksum.with_name(f".{checksum.name}.tmp")
    checksum_temporary.write_text(f"{digest}  {archive.name}\n", encoding="ascii", newline="\n")
    os.replace(checksum_temporary, checksum)
    return archive, checksum


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the CrabRAG Windows release archive")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        archive, checksum = build_release(args.root, args.output_dir, args.version)
    except (OSError, ReleaseError) as exc:
        parser.exit(2, f"release build failed: {exc}\n")
    print(archive)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
