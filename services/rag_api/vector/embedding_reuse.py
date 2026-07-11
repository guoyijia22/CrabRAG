from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from typing import Any


REUSE_PAGE_SIZE = 512


def iter_matching_embeddings(
    client,
    source_names: list[str],
    needed_keys: set[Hashable],
    key_builder: Callable[[str, dict[str, Any]], Hashable | None],
) -> Iterator[tuple[Hashable, list[float]]]:
    remaining = set(needed_keys)
    for source_name in source_names:
        if not remaining:
            return
        try:
            source = client.get_or_create_collection(name=source_name, metadata={"hnsw:space": "cosine"})
        except Exception:
            continue
        offset = 0
        while remaining:
            try:
                result = source.get(
                    include=["documents", "metadatas", "embeddings"],
                    limit=REUSE_PAGE_SIZE,
                    offset=offset,
                )
            except Exception:
                break
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            raw_embeddings = result.get("embeddings")
            embeddings = list(raw_embeddings) if raw_embeddings is not None else []
            for document, metadata, embedding in zip(documents, metadatas, embeddings):
                key = key_builder(str(document or ""), metadata or {})
                if key is None or key not in remaining:
                    continue
                remaining.remove(key)
                yield key, [float(value) for value in embedding]
            page_count = len(documents)
            if page_count < REUSE_PAGE_SIZE:
                break
            offset += page_count
