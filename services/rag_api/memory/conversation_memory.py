from __future__ import annotations

SESSION_MEMORY: dict[tuple[str, str, str, str], dict] = {}


def get_history(
    session_id: str,
    *,
    subject: str = "anonymous",
    generation_id: str = "legacy",
    permission_fingerprint: str = "legacy",
) -> list[dict[str, str]]:
    return SESSION_MEMORY.get((subject, session_id, generation_id, permission_fingerprint), {}).get("history", [])


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
    item = SESSION_MEMORY.setdefault(key, {"history": [], "last_intent": "", "last_entities": []})
    item["history"].append({"role": "user", "content": question})
    item["history"].append({"role": "assistant", "content": answer[:600]})
    item["history"] = item["history"][-8:]
    item["last_intent"] = intent
    item["last_entities"] = entities
