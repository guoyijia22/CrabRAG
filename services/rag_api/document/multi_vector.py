from __future__ import annotations

import re

from services.rag_api.rag_settings import RagSettings


def expand_multi_vector_chunks(chunks: list[dict], settings: RagSettings) -> list[dict]:
    if not settings.multi_vector_enabled:
        return chunks
    expanded: list[dict] = []
    for chunk in chunks:
        parent_id = chunk["id"]
        base_meta = {**chunk["metadata"], "parent_chunk_id": parent_id}
        expanded.append({**chunk, "id": f"{parent_id}_document", "metadata": {**base_meta, "granularity": "document"}})
        for index, paragraph in enumerate(_paragraphs(chunk["content"])):
            expanded.append(
                {
                    "id": f"{parent_id}_paragraph_{index:03d}",
                    "content": paragraph,
                    "metadata": {**base_meta, "granularity": "paragraph", "paragraph_index": index},
                }
            )
        for index, sentence in enumerate(_sentences(chunk["content"])):
            expanded.append(
                {
                    "id": f"{parent_id}_sentence_{index:03d}",
                    "content": sentence,
                    "metadata": {**base_meta, "granularity": "sentence", "sentence_index": index},
                }
            )
    return expanded


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n{1,}", text) if len(part.strip()) >= 4]


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[。！？；])", text.replace("\n", ""))
    return [part.strip() for part in raw if len(part.strip()) >= 3]
