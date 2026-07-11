from __future__ import annotations

import time
from collections import OrderedDict
from threading import RLock


SESSION_TTL_SECONDS = 30 * 60
MAX_SESSION_ENTRIES = 2048
SESSION_MEMORY: OrderedDict[tuple[str, str, str, str], dict] = OrderedDict()
_LOCK = RLock()
_time = time.monotonic


def get_history(
    session_id: str,
    *,
    subject: str = "anonymous",
    generation_id: str = "legacy",
    permission_fingerprint: str = "legacy",
) -> list[dict[str, str]]:
    key = (subject, session_id, generation_id, permission_fingerprint)
    now = _time()
    with _LOCK:
        _prune_expired(now)
        item = SESSION_MEMORY.get(key)
        if item is None:
            return []
        item["expires_at"] = now + SESSION_TTL_SECONDS
        SESSION_MEMORY.move_to_end(key)
        return [dict(entry) for entry in item.get("history", [])]


def update_memory(
    session_id: str,
    question: str,
    answer: str,
    intent: str,
    entities: list[str],
    *,
    subject: str = "anonymous",
    generation_id: str = "legacy",
    permission_fingerprint: str = "legacy",
) -> None:
    key = (subject, session_id, generation_id, permission_fingerprint)
    now = _time()
    with _LOCK:
        _prune_expired(now)
        item = SESSION_MEMORY.get(key)
        if item is None:
            item = {"history": [], "last_intent": "", "last_entities": [], "expires_at": 0.0}
            SESSION_MEMORY[key] = item
        item["history"].append({"role": "user", "content": question})
        item["history"].append({"role": "assistant", "content": answer[:600]})
        item["history"] = item["history"][-8:]
        item["last_intent"] = intent
        item["last_entities"] = list(entities)
        item["expires_at"] = now + SESSION_TTL_SECONDS
        SESSION_MEMORY.move_to_end(key)
        while len(SESSION_MEMORY) > MAX_SESSION_ENTRIES:
            SESSION_MEMORY.popitem(last=False)


def clear_memory() -> None:
    with _LOCK:
        SESSION_MEMORY.clear()


def _prune_expired(now: float) -> None:
    expired = [key for key, item in SESSION_MEMORY.items() if float(item.get("expires_at") or 0.0) <= now]
    for key in expired:
        SESSION_MEMORY.pop(key, None)
