# CrabRAG portable package

## Start

Double-click `start.bat`, then open:

http://127.0.0.1:3003

## Stop

Double-click `stop.bat`.

## Directories

- `config\.env`: model API configuration copied from the build machine.
- `docs\`: knowledge base documents.
- `data\chroma\`: bundled local Chroma vector database.
- `runtime\python\`: bundled Python runtime and Python dependencies.
- `runtime\bun\`: bundled Bun runtime.
- `crab-rag.bat`: evidence-only CLI entry for local tool integrations.
- `crabrag.skill`: UniClaw single-file Skill that calls the evidence-only CLI.

If you replace files in `docs\`, start the system and rebuild the knowledge base from the Knowledge page.

Legacy `ELCQA_*` environment variables are still accepted by backend configuration code, but new integrations should use `crab-rag.bat`, `crabrag.skill`, `CRABRAG_ROOT`, `CRABRAG_HOME`, and `CRABRAG_DOCS_DIR`.
