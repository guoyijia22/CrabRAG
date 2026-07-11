from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_cross_platform_install_scripts_are_source_install_friendly():
    install_ps1 = read_text("install.ps1")
    install_sh = read_text("install.sh")

    assert "$PSVersionTable.PSVersion.Major" in install_ps1
    assert "python -m venv" in install_ps1
    assert "runtime\\python\\python.exe" in install_ps1
    assert "pip install -r requirements.txt" in install_ps1
    assert "bun install" in install_ps1
    assert "runtime\\bun\\bun.exe" in install_ps1
    assert "Install-PortableBun" in install_ps1
    assert "bun-windows-x64.zip" in install_ps1
    assert "project-local Bun" in install_ps1
    assert "config\\.env" in install_ps1
    assert "Test-Path $EnvPath" in install_ps1
    assert "PIP_INDEX_URL" in install_ps1
    assert "https://pypi.org/simple" in install_ps1
    assert "scripts\\check_env.py" in install_ps1

    assert "set -Eeuo pipefail" in install_sh
    assert "python3 -m venv" in install_sh
    assert "runtime/python/python" in install_sh
    assert "pip install -r requirements.txt" in install_sh
    assert "bun install" in install_sh
    assert "runtime/bun/bun" in install_sh
    assert "install_portable_bun" in install_sh
    assert "bun-linux-x64.zip" in install_sh
    assert "bun-linux-aarch64.zip" in install_sh
    assert "project-local Bun" in install_sh
    assert "config/.env" in install_sh
    assert '[ -f "$ENV_FILE" ]' in install_sh
    assert "PIP_INDEX_URL" in install_sh
    assert "https://pypi.org/simple" in install_sh
    assert "scripts/check_env.py" in install_sh
    assert "apt install" in install_sh
    assert "dnf install" in install_sh or "yum install" in install_sh


def test_linux_installation_paths_are_hardened_for_source_clones():
    install_sh = read_text("install.sh")
    run_sh = read_text("run.sh")
    local_qwen = read_text("services/rag_api/llm/local_qwen_llm.py")
    gitattributes = read_text(".gitattributes")

    assert "*.sh text eol=lf" in gitattributes
    assert "if ! \"$VENV_PYTHON\" - \"$BUN_RELEASE_BASE_URL\" \"$bun_archive\" \"$bun_sha256\" \"$PORTABLE_BUN\"" in install_sh
    assert "Failed to download project-local Bun" in install_sh
    assert "runtime/bun/bun" in run_sh
    assert '"bun.exe"' in local_qwen
    assert '"bun"' in local_qwen
    assert "shutil.which(\"bun\")" in local_qwen


def test_run_scripts_start_api_and_gateway_with_project_environment():
    run_ps1 = read_text("run.ps1")
    run_sh = read_text("run.sh")
    start_bat = read_text("start.bat")
    cli_bat = read_text("crab-rag.bat")

    for content in (run_ps1, run_sh):
        assert "CRABRAG_ROOT" in content
        assert "CRABRAG_ENV_FILE" in content
        assert "RAG_BASE_URL" in content
        assert "services.rag_api.main:app" in content
        assert "server/gateway.js" in content
        assert "3003" in content
        assert "8001" in content

    assert "run.ps1" in start_bat
    assert ".venv\\Scripts\\python.exe" in cli_bat
    assert "runtime\\python\\python.exe" in cli_bat
    assert "services.rag_api.cli.evidence" in cli_bat


def test_bun_downloads_are_version_pinned_and_verified_before_extraction():
    install_ps1 = read_text("install.ps1")
    install_sh = read_text("install.sh")

    assert 'BunVersion = "1.3.14"' in install_ps1
    assert "releases/download/bun-v1.3.14/bun-windows-x64.zip" in install_ps1
    assert "0a0620930b6675d7ba440e81f4e0e00d3cfbe096c4b140d3fff02205e9e18922" in install_ps1
    assert "Get-FileHash" in install_ps1
    assert install_ps1.index("Get-FileHash") < install_ps1.index("Expand-Archive")
    assert "releases/latest" not in install_ps1
    assert "Test-BunVersion" in install_ps1

    assert 'BUN_VERSION="1.3.14"' in install_sh
    assert "releases/download/bun-v1.3.14" in install_sh
    assert "951ee2aee855f08595aeec6225226a298d3fea83a3dcd6465c09cbccdf7e848f" in install_sh
    assert "a27ffb63a8310375836e0d6f668ae17fa8d8d18b88c37c821c65331973a19a3b" in install_sh
    assert "hashlib.sha256" in install_sh
    assert install_sh.index("hashlib.sha256") < install_sh.index("zipfile.ZipFile")
    assert "releases/latest" not in install_sh
    assert "bun_version_ok" in install_sh


def test_start_batch_preserves_powershell_exit_code():
    start_bat = read_text("start.bat")

    assert 'set "EXIT_CODE=%ERRORLEVEL%"' in start_bat
    assert "if not %EXIT_CODE%==0 pause" in start_bat
    assert "exit /b %EXIT_CODE%" in start_bat


def test_installation_metadata_and_smoke_check_are_documented():
    gitignore = read_text(".gitignore")
    requirements = read_text("requirements.txt")
    env_example = read_text("config/.env.example")
    readme = read_text("README.md")
    check_env = read_text("scripts/check_env.py")

    assert ".venv/" in gitignore
    assert "fastapi==0.136.1" in requirements
    assert "uvicorn[standard]==0.47.0" in requirements
    assert "onnxruntime==1.26.0" in requirements
    assert "httpx==0.28.1" in requirements

    assert "CRABRAG_ROOT" in env_example
    assert "CRABRAG_DOCS_DIR" in env_example
    assert "PORT" in env_example
    assert "RAG_BASE_URL" in env_example
    assert "OPENAI_API_KEY" not in env_example

    assert ".\\install.ps1" in readme
    assert ".\\run.ps1" in readme
    assert "./install.sh" in readme
    assert "./run.sh" in readme
    assert "config/.env" in readme
    assert "scripts/check_env.py" in readme

    assert "check_import" in check_env
    assert '"onnxruntime"' in check_env
    assert "optional_warnings" in check_env
    assert "Local ONNX runtime unavailable" in check_env
    assert "config/.env" in check_env
    assert "apps/web/dist/index.html" in check_env
    assert "server/gateway.js" in check_env
    assert "bun" in check_env


def test_remote_api_mode_imports_main_without_importing_onnxruntime():
    source_files = [
        read_text("services/rag_api/llm/siliconflow_client.py"),
        read_text("services/rag_api/retrieval/optimizations.py"),
        read_text("services/rag_api/llm/local_onnx_embedding.py"),
        read_text("services/rag_api/llm/local_onnx_rerank.py"),
    ]

    assert "from services.rag_api.llm import local_onnx_embedding, local_qwen_llm" not in source_files[0].split("def _local_qwen_llm_module()", 1)[0]
    assert "from services.rag_api.llm import local_onnx_rerank" not in source_files[1].split("def _local_onnx_rerank_module()", 1)[0]
    assert "def _local_onnx_embedding_module()" in source_files[0]
    assert "def _local_onnx_rerank_module()" in source_files[1]
    assert "import onnxruntime as ort" not in source_files[2]
    assert "import onnxruntime as ort" not in source_files[3]
    assert "def _load_onnxruntime()" in source_files[2]
    assert "def _load_onnxruntime()" in source_files[3]
