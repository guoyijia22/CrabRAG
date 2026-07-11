from __future__ import annotations

import re

from services.rag_api.rag_settings import RagSettings

NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85
_CHAR_NGRAM_SIZE = 3
_WORD_NGRAM_SIZE = 2


def apply_candidate_dedup(chunks: list[dict], settings: RagSettings) -> tuple[list[dict], dict]:
    exact, exact_removed_count = _deduplicate_exact(chunks)
    if not settings.dedup_enabled:
        return exact, _trace(
            enabled=False,
            input_count=len(chunks),
            output_count=len(exact),
            exact_removed_count=exact_removed_count,
            near_duplicate_removed_count=0,
            reason="disabled",
        )

    result: list[dict] = []
    scopes: list[str] = []
    ngrams: list[set[str]] = []
    near_duplicate_removed_count = 0
    for index, chunk in enumerate(exact):
        scope = _scope(chunk, index)
        candidate_ngrams = _text_ngrams(str(chunk.get("content") or ""))
        matched_position = next(
            (
                position
                for position, (existing_scope, existing_ngrams) in enumerate(zip(scopes, ngrams))
                if scope == existing_scope
                and _jaccard(candidate_ngrams, existing_ngrams) >= NEAR_DUPLICATE_JACCARD_THRESHOLD
            ),
            None,
        )
        if matched_position is None:
            result.append(chunk)
            scopes.append(scope)
            ngrams.append(candidate_ngrams)
            continue
        near_duplicate_removed_count += 1
        if float(chunk.get("score", 0) or 0) > float(result[matched_position].get("score", 0) or 0):
            result[matched_position] = chunk
            ngrams[matched_position] = candidate_ngrams

    return result, _trace(
        enabled=True,
        input_count=len(chunks),
        output_count=len(result),
        exact_removed_count=exact_removed_count,
        near_duplicate_removed_count=near_duplicate_removed_count,
        reason="near_duplicate_filter",
    )


def _deduplicate_exact(chunks: list[dict]) -> tuple[list[dict], int]:
    result: list[dict] = []
    positions: dict[str, int] = {}
    removed_count = 0
    for chunk in chunks:
        identity = _stable_identity(chunk)
        if identity not in positions:
            positions[identity] = len(result)
            result.append(chunk)
            continue
        removed_count += 1
        position = positions[identity]
        if float(chunk.get("score", 0) or 0) > float(result[position].get("score", 0) or 0):
            result[position] = chunk
    return result, removed_count


def _stable_identity(chunk: dict) -> str:
    chunk_id = str(chunk.get("chunk_id") or "")
    if chunk_id:
        return f"chunk:{chunk_id}"
    chunk_hash = str(chunk.get("chunk_hash") or "")
    if chunk_hash:
        return f"hash:{chunk.get('document_id', '')}:{chunk.get('granularity', '')}:{chunk_hash}"
    return f"content:{chunk.get('source_file', '')}:{chunk.get('content', '')}"


def _scope(chunk: dict, index: int) -> str:
    document_id = str(chunk.get("document_id") or "")
    if document_id:
        return f"document:{document_id}"
    source_file = str(chunk.get("source_file") or "")
    if source_file:
        return f"source:{source_file}"
    return f"unscoped:{index}"


def _text_ngrams(text: str) -> set[str]:
    lowered = text.casefold()
    normalized = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", lowered))
    char_ngrams = _ngrams(list(normalized), _CHAR_NGRAM_SIZE, "c")
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
    word_ngrams = _ngrams(words, _WORD_NGRAM_SIZE, "w")
    return char_ngrams | word_ngrams


def _ngrams(items: list[str], size: int, prefix: str) -> set[str]:
    if not items:
        return set()
    if len(items) < size:
        return {f"{prefix}:{'|'.join(items)}"}
    return {f"{prefix}:{'|'.join(items[index:index + size])}" for index in range(len(items) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _trace(
    *,
    enabled: bool,
    input_count: int,
    output_count: int,
    exact_removed_count: int,
    near_duplicate_removed_count: int,
    reason: str,
) -> dict:
    return {
        "enabled": enabled,
        "input_count": input_count,
        "output_count": output_count,
        "removed_count": input_count - output_count,
        "exact_removed_count": exact_removed_count,
        "near_duplicate_removed_count": near_duplicate_removed_count,
        "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        "reason": reason,
    }
