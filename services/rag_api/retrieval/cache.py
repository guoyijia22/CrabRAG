from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from services.rag_api.security import RetrievalContext


class RetrievalCache:
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 2048, time_func: Callable[[], float] = time.monotonic) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._time = time_func
        self._items: OrderedDict[str, tuple[float, Any, frozenset[str]]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None or item[0] <= self._time():
                if item is not None:
                    self._items.pop(key, None)
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(item[1])

    def set(self, key: str, value: Any, document_tags: set[str] | frozenset[str] | None = None) -> None:
        with self._lock:
            self._items[key] = (self._time() + self.ttl_seconds, copy.deepcopy(value), frozenset(document_tags or set()))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def invalidate_documents(self, document_ids: set[str]) -> int:
        with self._lock:
            keys = [key for key, (_, _, tags) in self._items.items() if tags & document_ids]
            for key in keys:
                self._items.pop(key, None)
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._items), "hits": self._hits, "misses": self._misses}


def retrieval_cache_key(context: RetrievalContext, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "generation_id": context.generation_id,
            "permission_fingerprint": context.permission_fingerprint,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


RETRIEVAL_CACHE = RetrievalCache()
