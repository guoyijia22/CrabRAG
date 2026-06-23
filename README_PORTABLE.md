# enterprise-line-compliance-qa portable package

## Start

Double-click `start.bat`, then open:

http://127.0.0.1:3000

## Stop

Double-click `stop.bat`.

## Directories

- `config\.env`: model API configuration copied from the build machine.
- `docs\`: knowledge base documents.
- `data\chroma\`: bundled local Chroma vector database.
- `runtime\python\`: bundled Python runtime and Python dependencies.
- `runtime\bun\`: bundled Bun runtime.

If you replace files in `docs\`, start the system and rebuild the knowledge base from the Knowledge page.
