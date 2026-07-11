# CrabRAG

CrabRAG is a local RAG application with a Python FastAPI backend, a Bun gateway, a bundled web UI, and a CLI evidence interface.

Default URLs:

- Web UI: `http://127.0.0.1:3003/`
- Python API: `http://127.0.0.1:8001/`

## Requirements

- Python 3.10 or newer
- Internet access during dependency installation

The installer uses Bun for the bundled web gateway. If Bun is not on `PATH`, the installer downloads a project-local Bun binary into `runtime/bun/`. Node.js, npm, and pnpm are detected for diagnostics only.

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

If Bun is not on `PATH`, the installer downloads a project-local Bun binary into `runtime/bun/`.

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
The ONNX runtime is only required when local embedding or rerank models are enabled. Remote/API mode can start even if ONNX runtime is unavailable on the current machine.

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
It may print an optional warning when ONNX runtime cannot be imported; that warning does not block remote/API mode.

## Reinstall Dependencies

Reinstall Python dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Development and Testing

Normal installation and runtime environments continue to use `requirements.txt`. For development, install the runtime and pinned test dependencies through `requirements-dev.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```

Reinstall JavaScript dependencies:

```bash
runtime/bun/bun install
```

On Windows, use `runtime\bun\bun.exe install`.

The source repository keeps regression tests under `tests/`. Tests, TypeScript/React source, gateway TypeScript source, caches, local configuration, user data, API keys, and local models are excluded from Windows release packages.

Rebuild the web UI and gateway from source, or run the complete verification pipeline:

```bash
bun run build
bun run check
```

## Administration, backup, and restore

Run the cross-platform installation diagnosis (exit code `0` means healthy, `1` means warnings, and `2` means errors):

```powershell
.\.venv\Scripts\python.exe scripts\crabrag_admin.py doctor --json
```

Create and restore a verified backup:

```powershell
.\.venv\Scripts\python.exe scripts\crabrag_admin.py backup --output .\backup.zip
.\.venv\Scripts\python.exe scripts\crabrag_admin.py restore --archive .\backup.zip --yes
```

Backups include local configuration, Chroma, index generations, and application state. External knowledge-base files are never copied; only their normalized directory paths are recorded. Stop CrabRAG before restore. Restore validates the format, software compatibility, paths, and every SHA-256 checksum before changing local state.
Because a backup can contain plaintext API credentials, the command marks this in its manifest and output and applies owner-only file permissions where the operating system supports them. Store backup ZIP files as securely as API keys.

Build the Windows x64 release and its SHA-256 file:

```powershell
.\scripts\build_release.ps1 -Version 1.1.0 -OutputDir .\release
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

## Production index governance

CrabRAG publishes text vectors, categories, and graph artifacts as one atomic index generation. A staging generation is invisible until every build step succeeds, and the previous verified generation is retained for rollback.

Each knowledge-base root may contain `.crabrag-manifest.json`:

```json
{
  "schema_version": 1,
  "knowledge_base_id": "kb-example",
  "documents": [
    {
      "document_id": "policy-a",
      "path": "policy-v2.pdf",
      "version": "2",
      "status": "published",
      "effective_at": "2026-08-01T00:00:00Z",
      "updated_at": "2026-07-11T00:00:00Z",
      "acl": {"visibility": "restricted", "roles": ["sales"], "revision": "7"}
    }
  ]
}
```

Valid states are `draft`, `published`, and `retired`. Unregistered files are automatically registered as public/published/version 1 and produce a high-risk governance warning. Timestamps must be timezone-aware ISO-8601 values.

- `GET /api/index/status` reports current/previous generations, vector reuse, cache, scheduler, cleanup, and warnings.
- `POST /api/index/rollback` rolls an administrator back to the previous compatible generation.
- `CRABRAG_INTERNAL_TOKEN` protects identity headers between the Bun gateway and RAG API; standard run scripts generate it when absent.

The local identity adapter reads `CRABRAG_SUBJECT`, `CRABRAG_ROLES`, `CRABRAG_GROUPS`, `CRABRAG_PERMISSION_REVISION`, and `CRABRAG_LOCAL_ADMIN`. Identity headers supplied directly by a browser are not forwarded.

## Troubleshooting

- `Python 3.10+ was not found`: install Python 3.10 or newer and rerun the installer.
- `Failed to install project-local Bun`: check network access to GitHub releases, or install Bun manually from <https://bun.sh/docs/installation>.
- `Local ONNX runtime unavailable`: remote/API mode can still run. Install the correct ONNX runtime only if you need local embedding or rerank models.
- `API port 8001 is already in use`: stop the process using that port or change the API port.
- `Web port 3003 is already in use`: stop the process using that port or start with another port.
- `config/.env is missing`: rerun the installer or copy `config/.env.example` to `config/.env`.
- The UI opens but API calls fail: confirm the Python API is running at `http://127.0.0.1:8001/api/health`.
