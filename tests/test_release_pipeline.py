from __future__ import annotations

import hashlib
from pathlib import Path
import json
import shutil
import subprocess
import sys
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
        "package.json": json.dumps({"dependencies": {"@huggingface/transformers": "4.0.0-next.11"}}),
        "bun.lock": "lockfileVersion = 1\n",
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
        "config/evaluation-dataset.example.json": '{"schema_version": 1, "dataset_id": "example", "dataset_version": "1", "cases": [{"id": "q1", "question": "example"}]}',
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
    assert "CrabRAG/config/evaluation-dataset.example.json" in names
    assert "CrabRAG/scripts/check_env.py" in names
    assert "CrabRAG/scripts/stop.ps1" in names
    assert "CrabRAG/package.json" in names
    assert "CrabRAG/bun.lock" in names
    assert not any("node_modules" in name for name in names)
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
    assert "bash ./install.sh" in windows
    assert "start" in windows.lower() and "health" in windows.lower()
    assert "Expand-Archive" in windows
    assert '$version = (Get-Content -LiteralPath .\\VERSION -Raw).Trim()' in windows
    assert "CrabRAG-v$version-windows-x64.zip" in windows
    assert "@huggingface/transformers" in windows
    assert "stop.bat" in windows and "run.json" in windows
    assert '"release:windows"' in package


def test_ubuntu_core_tests_use_the_development_python_not_release_venv():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "python -m pytest -q" in workflow
    assert "./.venv/bin/python -m pytest" not in workflow
    assert "pytest==" not in requirements


def test_release_installer_and_stop_scripts_do_not_depend_on_excluded_development_files():
    install_ps1 = Path("install.ps1").read_text(encoding="utf-8")
    install_sh = Path("install.sh").read_text(encoding="utf-8")
    check_env = Path("scripts/check_env.py").read_text(encoding="utf-8")
    stop_bat = Path("stop.bat").read_text(encoding="utf-8")

    assert "release-manifest.json" in install_ps1 and "--production" in install_ps1 and "--frozen-lockfile" in install_ps1
    assert "release-manifest.json" in install_sh and "--production" in install_sh and "--frozen-lockfile" in install_sh
    assert '"package.json",' not in check_env
    assert "scripts\\stop.ps1" in stop_bat
    assert Path("scripts/stop.ps1").is_file()
    assert "data\\run.json" in Path("scripts/stop.ps1").read_text(encoding="utf-8")


def test_transformers_is_pinned_and_importable_with_bun():
    package = json.loads(Path("package.json").read_text(encoding="utf-8"))
    assert package["dependencies"]["@huggingface/transformers"] == "4.0.0-next.11"
    bun = Path("runtime/bun/bun.exe")
    if not bun.is_file():
        pytest.skip("project-local Bun is unavailable")
    result = subprocess.run(
        [str(bun), "-e", "await import('@huggingface/transformers'); console.log('transformers-ok')"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "transformers-ok" in result.stdout


def test_windows_gateway_argument_preserves_an_absolute_path_with_spaces(tmp_path: Path):
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    directory = tmp_path / "Program Files" / "CrabRAG"
    directory.mkdir(parents=True)
    gateway = directory / "server" / "gateway.js"
    gateway.parent.mkdir()
    gateway.write_text("stub", encoding="utf-8")
    recorder = directory / "record argv.py"
    output = directory / "argv.json"
    recorder.write_text(
        "import json, pathlib, sys\npathlib.Path(sys.argv[2]).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    script = f"""
$gatewayPath = '{str(gateway).replace("'", "''")}'
$gatewayArgument = '\"' + $gatewayPath + '\"'
$recorderArgument = '\"{str(recorder).replace("'", "''")}\"'
$outputArgument = '\"{str(output).replace("'", "''")}\"'
$process = Start-Process -FilePath '{sys.executable.replace("'", "''")}' -ArgumentList @($recorderArgument, $gatewayArgument, $outputArgument) -Wait -PassThru -WindowStyle Hidden
exit $process.ExitCode
"""
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))[0] == str(gateway)

    source = Path("run.ps1").read_text(encoding="utf-8")
    assert "$GatewayArgument" in source
    assert "-ArgumentList @($GatewayArgument)" in source
