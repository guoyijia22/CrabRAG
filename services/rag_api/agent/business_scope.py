from __future__ import annotations

from typing import Any

from services.rag_api.app_settings import AppSettings, load_app_settings
from services.rag_api.graph.graph_search import extract_entities


def check_business_scope(question: str, categories: list[str], settings: AppSettings | None = None) -> dict[str, Any]:
    active = settings or load_app_settings()
    text = question.strip()
    excluded = _matched_terms(text, active.out_of_scope_keywords)
    matched_keywords = _matched_terms(text, active.in_scope_keywords)
    matched_categories = _matched_terms(text, categories)
    entities = extract_entities(text)

    score = 0.0
    score += min(0.55, len(matched_keywords) * 0.12)
    score += min(0.25, len(matched_categories) * 0.12)
    score += min(0.25, len(entities) * 0.10)
    score = round(min(1.0, score), 4)

    in_scope = not excluded and score >= active.scope_min_score
    return {
        "in_scope": in_scope,
        "score": score,
        "matched_keywords": matched_keywords,
        "matched_categories": matched_categories,
        "matched_entities": entities,
        "excluded_keywords": excluded,
        "response_type": "normal" if in_scope else "out_of_scope",
        "scope_min_score": active.scope_min_score,
    }


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    result: list[str] = []
    for term in terms:
        value = term.strip()
        if value and value in text and value not in result:
            result.append(value)
    return result
