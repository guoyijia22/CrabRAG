from __future__ import annotations

SESSION_MEMORY: dict[str, dict] = {}


def get_history(session_id: str) -> list[dict[str, str]]:
    return SESSION_MEMORY.get(session_id, {}).get("history", [])


def update_memory(session_id: str, question: str, answer: str, intent: str, entities: list[str]) -> None:
    item = SESSION_MEMORY.setdefault(session_id, {"history": [], "last_intent": "", "last_entities": []})
    item["history"].append({"role": "user", "content": question})
    item["history"].append({"role": "assistant", "content": answer[:600]})
    item["history"] = item["history"][-8:]
    item["last_intent"] = intent
    item["last_entities"] = entities
