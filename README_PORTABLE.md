# CrabRAG portable package

## Start

Double-click `start.bat`, then open:

http://127.0.0.1:3003

## Stop

Double-click `stop.bat`.

## Directories

- `config\.env`: local model API configuration created from `config\.env.example` during installation; release packages never contain a build-machine configuration or API key.
- `docs\`: knowledge base directory created locally; release packages do not contain document content.
- `data\chroma\`: local Chroma state created at runtime; release packages do not contain user data.
- `.venv\`: Python environment created by `install.ps1` from the pinned runtime manifest in `requirements.txt`.
- `runtime\bun\`: project-local Bun downloaded and checksum-verified by the installer when a compatible Bun is not already available.
- `crab-rag.bat`: evidence-only CLI entry for local tool integrations.
- `crabrag.skill`: UniClaw single-file Skill that calls the evidence-only CLI.

If you replace files in `docs\`, start the system and rebuild the knowledge base from the Knowledge page.

Run `.\.venv\Scripts\python.exe scripts\crabrag_admin.py doctor --json` for structured diagnostics. Before an upgrade, use `.\.venv\Scripts\python.exe scripts\crabrag_admin.py backup --output backup.zip`. Stop the service before running `.\.venv\Scripts\python.exe scripts\crabrag_admin.py restore --archive backup.zip --yes`.
Backup ZIP files may contain plaintext API credentials. Keep them in an owner-only location and do not upload them to source control or public storage.

The release archive excludes tests, development source, caches, logs, user data, secrets, and local model files. The installer recreates required runtime directories and dependencies.

Legacy `ELCQA_*` environment variables are still accepted by backend configuration code, but new integrations should use `crab-rag.bat`, `crabrag.skill`, `CRABRAG_ROOT`, `CRABRAG_HOME`, and `CRABRAG_DOCS_DIR`.
