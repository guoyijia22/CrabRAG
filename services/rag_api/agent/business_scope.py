from __future__ import annotations

from typing import Any

from services.rag_api.app_settings import AppSettings, load_app_settings
from services.rag_api.document.categories import load_kb_categories
from services.rag_api.graph.kb_graph_builder import extract_source_topics
from services.rag_api.graph.graph_search import extract_entities


def check_business_scope(question: str, categories: list[str], settings: AppSettings | None = None) -> dict[str, Any]:
    active = settings or load_app_settings()
    text = question.strip()
    excluded = _matched_terms(text, active.out_of_scope_keywords)
    matched_keywords = _matched_terms(text, active.in_scope_keywords)
    matched_categories = _matched_terms(text, categories)
    matched_source_files, matched_kb_terms = _matched_kb_terms(text)
    entities = extract_entities(text)

    score = 0.0
    score += min(0.55, len(matched_keywords) * 0.12)
    score += min(0.25, len(matched_categories) * 0.12)
    score += min(0.35, len(matched_source_files) * 0.2 + len(matched_kb_terms) * 0.15)
    score += min(0.25, len(entities) * 0.10)
    score = round(min(1.0, score), 4)

    in_scope = not excluded and score >= active.scope_min_score
    return {
        "in_scope": in_scope,
        "score": score,
        "matched_keywords": matched_keywords,
        "matched_categories": matched_categories,
        "matched_source_files": matched_source_files,
        "matched_kb_terms": matched_kb_terms,
        "matched_entities": entities,
        "excluded_keywords": excluded,
        "response_type": "normal" if in_scope else "out_of_scope",
        "scope_min_score": active.scope_min_score,
    }


def _matched_kb_terms(text: str) -> tuple[list[str], list[str]]:
    source_files: list[str] = []
    terms: list[str] = []
    for item in load_kb_categories().get("items", []) or []:
        item_terms = [str(term) for term in item.get("keyword_hits", []) or []]
        for source_file in item.get("source_files", []) or []:
            source = str(source_file)
            source_topics = extract_source_topics(source)
            item_terms.extend(source_topics)
            if source and (source in text or any(topic and topic in text for topic in source_topics)):
                source_files.append(source)
        terms.extend(term for term in item_terms if term and term in text)
    return _dedupe(source_files), _dedupe(terms)


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    result: list[str] = []
    for term in terms:
        value = term.strip()
        if value and value in text and value not in result:
            result.append(value)
    return result


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
