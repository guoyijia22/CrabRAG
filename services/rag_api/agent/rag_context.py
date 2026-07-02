from __future__ import annotations

import json
from typing import Any


def build_rag_context(chunks: list[dict[str, Any]], relation_paths: list[dict[str, Any]]) -> str:
    context_chunks = [_chunk_record(index, chunk) for index, chunk in enumerate(chunks, start=1)]
    relationships = [_relationship_record(item) for item in relation_paths]
    entities = _entity_records(relation_paths)
    reference_list = "\n".join(
        f"[{item['reference_id']}] {item['source_file'] or 'unknown'}" for item in context_chunks
    )
    return f"""Knowledge Graph Data (Entity):

```json
{json.dumps(entities, ensure_ascii=False, indent=2)}
```

Knowledge Graph Data (Relationship):

```json
{json.dumps(relationships, ensure_ascii=False, indent=2)}
```

Document Chunks:

```json
{json.dumps(context_chunks, ensure_ascii=False, indent=2)}
```

Reference Document List:

```
{reference_list}
```
"""


def _chunk_record(reference_id: int, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_id": reference_id,
        "source_file": str(chunk.get("source_file") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "category": str(chunk.get("category") or ""),
        "content": str(chunk.get("content") or ""),
    }


def _relationship_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(item.get("path") or ""),
        "description": str(item.get("description") or ""),
        "source_file": str(item.get("source_file") or ""),
        "evidence": str(item.get("evidence") or ""),
        "score": item.get("score", ""),
        "match_source": str(item.get("match_source") or ""),
    }


def _entity_records(relation_paths: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in relation_paths:
        parts = [part.strip() for part in str(item.get("path") or "").split("->") if part.strip()]
        for index, part in enumerate(parts):
            if index % 2 != 0 or not part or part in seen:
                continue
            seen.add(part)
            result.append({"id": part, "label": part})
    return result
