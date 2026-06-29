# CrabRAG

CrabRAG is a local RAG application with a Python FastAPI backend, a Bun gateway, a bundled web UI, and a CLI evidence interface.

Default URLs:

- Web UI: `http://127.0.0.1:3003/`
- Python API: `http://127.0.0.1:8001/`

## Requirements

- Python 3.10 or newer
- Bun on `PATH`
- Internet access during dependency installation

Node.js, npm, and pnpm are detected for diagnostics, but the bundled web gateway uses Bun.

## Windows Install

Open PowerShell in the project directory:

```powershell
.\install.ps1
.\run.ps1
```

You can also start the app with:

```powershell
.\start.bat
```

If PowerShell blocks script execution, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

## Linux Install

From the project directory:

```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
```

If Python or venv support is missing, install it first:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y python3 python3-venv python3-pip

# CentOS / Rocky / AlmaLinux
sudo dnf install -y python3 python3-pip
```

Install Bun if needed:

```bash
curl -fsSL https://bun.sh/install | bash
```

Restart your shell after installing Bun so `bun` is on `PATH`.

## Configuration

The installer copies `config/.env.example` to `config/.env` only when `config/.env` does not already exist. It never overwrites your existing configuration.

Common settings:

- `CRABRAG_DOCS_DIR`: optional knowledge-base directories. Separate multiple directories with `;`.
- `RAG_BASE_URL`: API endpoint used by the gateway when you start services manually.
- `PORT`: web gateway port when you start `server/gateway.js` manually.

Model API keys and chat model settings are normally configured in the Settings page, not in `config/.env`.

## Local Models

Local models are optional. If you enable local models in Settings, download them manually into:

```text
runtime/models/
```

The Settings page reports missing model files and shows the matching ModelScope or Hugging Face download links.

## Smoke Check

After installation, run:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

or on Linux:

```bash
./.venv/bin/python scripts/check_env.py
```

The check verifies key Python imports, `config/.env`, required directories, the built web UI, the gateway entrypoint, and Bun availability.

## Reinstall Dependencies

Reinstall Python dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Reinstall JavaScript dependencies:

```bash
bun install
```

To rebuild from a clean virtual environment, delete `.venv` and rerun the installer:

```powershell
Remove-Item -LiteralPath .\.venv -Recurse -Force
.\install.ps1
```

```bash
rm -rf .venv
./install.sh
```

Do not delete `data/`, `docs/`, or `config/.env` unless you intentionally want to remove local runtime state, knowledge-base files, or configuration.

## Troubleshooting

- `Python 3.10+ was not found`: install Python 3.10 or newer and rerun the installer.
- `Bun was not found`: install Bun and restart the terminal.
- `API port 8001 is already in use`: stop the process using that port or change the API port.
- `Web port 3003 is already in use`: stop the process using that port or start with another port.
- `config/.env is missing`: rerun the installer or copy `config/.env.example` to `config/.env`.
- The UI opens but API calls fail: confirm the Python API is running at `http://127.0.0.1:8001/api/health`.
