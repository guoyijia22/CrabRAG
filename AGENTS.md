# Repository Guidelines

## Project Structure & Module Organization

This repository is a portable enterprise-line compliance RAG package. Python API code lives in `services/rag_api/`, organized by concern: `agent/` for LangGraph QA flow, `document/` for ingestion and parsing, `vector/` for Chroma integration, `graph/` for relationship search, `evaluation/` for evaluation runs, and `logging_utils/` for QA logs. `server/gateway.js` is the bundled Bun gateway that serves the web system and proxies API calls. `apps/web/dist/` contains the built frontend assets; frontend source is not included in this package. Knowledge-base files are stored in `docs/`, runtime state and Chroma data in `data/`, configuration in `config/.env`, and bundled executables/dependencies in `runtime/`.

## Build, Test, and Development Commands

- `start.bat`: starts the bundled Python FastAPI service on `127.0.0.1:8001` and the Bun gateway on `127.0.0.1:3000`.
- `stop.bat`: stops the portable package processes recorded under `data/run/`.
- `runtime\python\python.exe -m uvicorn services.rag_api.main:app --host 127.0.0.1 --port 8001`: runs only the API for backend debugging.
- `runtime\python\python.exe -m pytest`: runs Python tests when test files are added.

Use the bundled runtimes for local verification so dependency versions match the portable package.

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, type hints on public request/response paths, `snake_case` modules/functions, and Pydantic schemas in `schemas.py` or focused settings modules. Keep FastAPI route handlers thin and move retrieval, graph, document, or model logic into the matching subpackage. For JavaScript in `server/`, treat the current file as generated/bundled output; avoid broad rewrites unless rebuilding the gateway from its source.

## Testing Guidelines

No tests are currently checked in, but `pytest` is available. Add tests near the backend surface they cover, using names like `test_ingest.py` or `test_graph_search.py`. Prefer focused tests for document parsing, retrieval decisions, settings persistence, and API responses. When changing RAG behavior, include at least one regression case with a small fixture or mocked model/vector dependency.

## Commit & Pull Request Guidelines

This directory is not currently a Git repository, so no project-specific commit history is available. Use concise imperative commits such as `Add evaluation storage test` or `Fix ingest error handling`. Pull requests should describe the behavior change, list verification commands, call out changes to `config/`, `data/`, or `docs/`, and include screenshots when the web UI changes.

## Security & Configuration Tips

Do not commit real API keys or customer documents. Treat `config/.env`, `data/chroma/`, `logs/`, and generated runtime files as environment-specific unless the package explicitly requires them.
