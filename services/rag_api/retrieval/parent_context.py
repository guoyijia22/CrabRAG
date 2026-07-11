from __future__ import annotations

from services.rag_api.rag_settings import RagSettings
from services.rag_api.security import current_retrieval_context
from services.rag_api.vector.chroma_store import fetch_document_parent_chunks

_CHILD_GRANULARITIES = {"paragraph", "sentence"}


def apply_parent_context(chunks: list[dict], settings: RagSettings) -> tuple[list[dict], dict]:
    if not settings.parent_context_enabled:
        return list(chunks), _trace(enabled=False, reason="disabled")

    context = current_retrieval_context()
    allowed_document_ids = context.allowed_document_ids if context else None
    visible_chunks: list[dict] = []
    dropped_unauthorized_count = 0
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        if allowed_document_ids is not None and document_id not in allowed_document_ids:
            dropped_unauthorized_count += 1
            continue
        visible_chunks.append(chunk)

    parent_pairs = {
        (str(chunk.get("document_id") or ""), str(chunk.get("parent_chunk_id") or ""))
        for chunk in visible_chunks
        if str(chunk.get("granularity") or "") in _CHILD_GRANULARITIES
        and chunk.get("document_id")
        and chunk.get("parent_chunk_id")
    }
    if not parent_pairs:
        return visible_chunks, _trace(
            enabled=True,
            reason="no_eligible_children",
            dropped_unauthorized_count=dropped_unauthorized_count,
        )

    parents = fetch_document_parent_chunks(parent_pairs)
    expanded: list[dict] = []
    expanded_count = 0
    missing_parent_count = 0
    for chunk in visible_chunks:
        granularity = str(chunk.get("granularity") or "")
        pair = (str(chunk.get("document_id") or ""), str(chunk.get("parent_chunk_id") or ""))
        if granularity not in _CHILD_GRANULARITIES or not all(pair):
            expanded.append(chunk)
            continue
        parent = parents.get(pair)
        if parent is None:
            missing_parent_count += 1
            expanded.append(chunk)
            continue
        expanded_count += 1
        expanded.append(
            {
                **parent,
                "score": chunk.get("score", parent.get("score", 0)),
                "retrieval_channel": chunk.get("retrieval_channel", parent.get("retrieval_channel", "")),
                "matched_chunk_id": chunk.get("chunk_id", ""),
                "matched_granularity": granularity,
            }
        )

    deduplicated, deduplicated_count = _deduplicate(expanded)
    reason = "expanded" if expanded_count else "parents_missing"
    return deduplicated, _trace(
        enabled=True,
        reason=reason,
        expanded_count=expanded_count,
        missing_parent_count=missing_parent_count,
        dropped_unauthorized_count=dropped_unauthorized_count,
        deduplicated_count=deduplicated_count,
    )


def _deduplicate(chunks: list[dict]) -> tuple[list[dict], int]:
    result: list[dict] = []
    positions: dict[str, int] = {}
    deduplicated_count = 0
    for chunk in chunks:
        identity = str(chunk.get("chunk_id") or "")
        if not identity:
            result.append(chunk)
            continue
        if identity not in positions:
            positions[identity] = len(result)
            result.append(chunk)
            continue
        deduplicated_count += 1
        position = positions[identity]
        if float(chunk.get("score", 0) or 0) > float(result[position].get("score", 0) or 0):
            result[position] = chunk
    return result, deduplicated_count


def _trace(
    *,
    enabled: bool,
    reason: str,
    expanded_count: int = 0,
    missing_parent_count: int = 0,
    dropped_unauthorized_count: int = 0,
    deduplicated_count: int = 0,
) -> dict:
    return {
        "enabled": enabled,
        "expanded_count": expanded_count,
        "missing_parent_count": missing_parent_count,
        "dropped_unauthorized_count": dropped_unauthorized_count,
        "deduplicated_count": deduplicated_count,
        "reason": reason,
    }
