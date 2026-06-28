from __future__ import annotations


def test_business_scope_matches_source_file_topics(monkeypatch):
    from services.rag_api.agent import business_scope
    from services.rag_api.app_settings import AppSettings

    monkeypatch.setattr(
        business_scope,
        "load_kb_categories",
        lambda: {
            "items": [
                {
                    "name": "合规审核",
                    "source_files": ["关于一渠一表调研报告.docx"],
                    "keyword_hits": [],
                }
            ],
            "categories": ["合规审核"],
        },
        raising=False,
    )

    result = business_scope.check_business_scope(
        "关于一渠一表的报告在哪",
        ["合规审核"],
        settings=AppSettings(scope_min_score=0.2),
    )

    assert result["in_scope"] is True
    assert "一渠一表" in result["matched_kb_terms"]
    assert "关于一渠一表调研报告.docx" in result["matched_source_files"]
