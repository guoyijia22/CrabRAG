from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest


def _minimal_release_tree(root: Path) -> None:
    files = {
        "VERSION": "1.1.0\n",
        "LICENSE": "MIT",
        "README.md": "readme",
        "README_ZH.md": "readme zh",
        "README_PORTABLE.md": "portable",
        "requirements.txt": "fastapi==1\n",
        "install.ps1": "Write-Host install",
        "install.sh": "#!/usr/bin/env bash\n",
        "run.ps1": "Write-Host run",
        "run.sh": "#!/usr/bin/env bash\n",
        "start.bat": "@echo off",
        "stop.bat": "@echo off",
        "crab-rag.bat": "@echo off",
        "crabrag.skill": "skill",
        "server/gateway.js": "console.log('gateway')",
        "apps/web/dist/index.html": "<div>CrabRAG</div>",
        "services/rag_api/main.py": "app = object()",
        "scripts/crabrag_admin.py": "print('admin')",
        "scripts/check_env.py": "print('check')",
        "scripts/stop.ps1": "Write-Host stop",
        "config/.env.example": "CRABRAG_API_KEY=\n",
        "skills/crabrag-rag/SKILL.md": "skill",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_release_builder_includes_runtime_files_and_excludes_development_and_user_state(tmp_path: Path):
    from scripts import build_release

    root = tmp_path / "repo"
    _minimal_release_tree(root)
    excluded = {
        "tests/test_secret.py": "secret",
        "apps/web/src/App.tsx": "source",
        "server/bun_api/index.ts": "source",
        ".github/workflows/ci.yml": "dev",
        "data/model_api_settings.json": '{"api_key":"secret"}',
        "config/.env": "CRABRAG_API_KEY=secret",
        "runtime/models/private/model.onnx": "model",
        "node_modules/pkg/index.js": "dependency",
    }
    for name, content in excluded.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    output = tmp_path / "release"

    archive, checksum = build_release.build_release(root, output, "1.1.0")

    assert archive.name == "CrabRAG-v1.1.0-windows-x64.zip"
    assert checksum.name == f"{archive.name}.sha256"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() in checksum.read_text(encoding="ascii")
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "CrabRAG/services/rag_api/main.py" in names
    assert "CrabRAG/apps/web/dist/index.html" in names
    assert "CrabRAG/server/gateway.js" in names
    assert "CrabRAG/config/.env.example" in names
    assert "CrabRAG/scripts/check_env.py" in names
    assert "CrabRAG/scripts/stop.ps1" in names
    for forbidden in excluded:
        assert f"CrabRAG/{forbidden}" not in names


def test_release_builder_rejects_secret_like_files(tmp_path: Path):
    from scripts import build_release

    root = tmp_path / "repo"
    _minimal_release_tree(root)
    (root / "services" / "leaked.py").write_text('API_KEY = "sk-super-secret-value"', encoding="utf-8")

    with pytest.raises(build_release.ReleaseError, match="secret"):
        build_release.build_release(root, tmp_path / "release", "1.1.0")


def test_release_powershell_wrapper_and_ci_contracts_are_present():
    wrapper = Path("scripts/build_release.ps1").read_text(encoding="utf-8")
    windows = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    package = Path("package.json").read_text(encoding="utf-8")

    assert "[string]$Version" in wrapper and "[string]$OutputDir" in wrapper
    assert "build_release.py" in wrapper
    assert "windows-latest" in windows
    assert "ubuntu-latest" in windows
    assert "python-version: '3.10'" in windows
    assert "bun-version: '1.3.14'" in windows
    assert "bun run check" in windows
    assert "install.ps1" in windows and "install.sh" in windows
    assert "start" in windows.lower() and "health" in windows.lower()
    assert '"release:windows"' in package


def test_release_installer_and_stop_scripts_do_not_depend_on_excluded_development_files():
    install_ps1 = Path("install.ps1").read_text(encoding="utf-8")
    install_sh = Path("install.sh").read_text(encoding="utf-8")
    check_env = Path("scripts/check_env.py").read_text(encoding="utf-8")
    stop_bat = Path("stop.bat").read_text(encoding="utf-8")

    assert "package.json" in install_ps1 and "Skipping JavaScript dependency install" in install_ps1
    assert "package.json" in install_sh and "Skipping JavaScript dependency install" in install_sh
    assert '"package.json",' not in check_env
    assert "scripts\\stop.ps1" in stop_bat
    assert Path("scripts/stop.ps1").is_file()
    assert "data\\run.json" in Path("scripts/stop.ps1").read_text(encoding="utf-8")
