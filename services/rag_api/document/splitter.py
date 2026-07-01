from __future__ import annotations

import re

from services.rag_api.document.cleaner import infer_category


SECTION_RE = re.compile(r"(?m)^(第?[一二三四五六七八九十]+[、.．]|[（(][一二三四五六七八九十]+[）)]|\d+[.．])")


def _paragraph_units(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if SECTION_RE.match(line) and current:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _section_title(chunk: str) -> str:
    first = chunk.splitlines()[0].strip()
    return first[:80]


def split_documents(documents: list[dict], chunk_size: int = 600, chunk_overlap: int = 100) -> list[dict]:
    chunks: list[dict] = []
    for doc_index, doc in enumerate(documents, start=1):
        source_file = doc["source_file"]
        source_path = doc["source_path"]
        doc_id = doc.get("doc_id")
        content_hash = doc.get("content_hash")
        category = infer_category(source_file, doc["content"])
        buffer = ""
        chunk_index = 0
        for unit in _paragraph_units(doc["content"]):
            if len(buffer) + len(unit) + 1 <= chunk_size:
                buffer = f"{buffer}\n{unit}".strip()
                continue
            if buffer:
                chunk_index += 1
                chunks.append(_make_chunk(doc_index, chunk_index, buffer, source_file, source_path, category, doc_id, content_hash))
                buffer = buffer[-chunk_overlap:] if chunk_overlap > 0 else ""
            if len(unit) > chunk_size:
                start = 0
                while start < len(unit):
                    piece = unit[start : start + chunk_size]
                    chunk_index += 1
                    chunks.append(_make_chunk(doc_index, chunk_index, piece, source_file, source_path, category, doc_id, content_hash))
                    start += max(1, chunk_size - chunk_overlap)
                buffer = ""
            else:
                buffer = f"{buffer}\n{unit}".strip()
        if buffer:
            chunk_index += 1
            chunks.append(_make_chunk(doc_index, chunk_index, buffer, source_file, source_path, category, doc_id, content_hash))
    return chunks


def _make_chunk(
    doc_index: int,
    chunk_index: int,
    content: str,
    source_file: str,
    source_path: str,
    category: str,
    doc_id: str | None = None,
    content_hash: str | None = None,
) -> dict:
    metadata = {
        "source_file": source_file,
        "category": category,
        "section_title": _section_title(content),
        "chunk_index": chunk_index,
        "source_path": source_path,
    }
    if doc_id:
        metadata["doc_id"] = doc_id
    if content_hash:
        metadata["content_hash"] = content_hash
    return {
        "id": f"{doc_id}::chunk::{chunk_index:04d}" if doc_id else f"doc_{doc_index:02d}_chunk_{chunk_index:04d}",
        "content": content.strip(),
        "metadata": metadata,
    }
