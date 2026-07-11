from __future__ import annotations

import hashlib
import re

from services.rag_api.rag_settings import RagSettings


def expand_multi_vector_chunks(chunks: list[dict], settings: RagSettings) -> list[dict]:
    if not settings.multi_vector_enabled:
        return chunks
    expanded: list[dict] = []
    occurrences_by_document: dict[str, dict[tuple[str, str], int]] = {}
    for chunk in chunks:
        parent_id = chunk["id"]
        base_meta = {**chunk["metadata"], "parent_chunk_id": parent_id}
        document_id = str(base_meta.get("document_id") or base_meta.get("doc_id") or "")
        identity_prefix = document_id or parent_id
        occurrences = occurrences_by_document.setdefault(identity_prefix, {})
        expanded.append(_expanded_chunk(identity_prefix, chunk["content"], "document", base_meta, occurrences))
        for index, paragraph in enumerate(_paragraphs(chunk["content"])):
            expanded.append(_expanded_chunk(identity_prefix, paragraph, "paragraph", {**base_meta, "paragraph_index": index}, occurrences))
        for index, sentence in enumerate(_sentences(chunk["content"])):
            expanded.append(_expanded_chunk(identity_prefix, sentence, "sentence", {**base_meta, "sentence_index": index}, occurrences))
    return expanded


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n{1,}", text) if len(part.strip()) >= 4]


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[。！？；])", text.replace("\n", ""))
    return [part.strip() for part in raw if len(part.strip()) >= 3]


def _expanded_chunk(
    identity_prefix: str,
    content: str,
    granularity: str,
    metadata: dict,
    occurrences: dict[tuple[str, str], int],
) -> dict:
    normalized = re.sub(r"\s+", " ", content).strip()
    chunk_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    occurrence_key = (granularity, chunk_hash)
    occurrence = occurrences.get(occurrence_key, 0) + 1
    occurrences[occurrence_key] = occurrence
    chunk_id = f"{identity_prefix}::{granularity}::{chunk_hash[:20]}::{occurrence:03d}"
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {
            **metadata,
            "granularity": granularity,
            "chunk_hash": chunk_hash,
            "chunk_id": chunk_id,
        },
    }
