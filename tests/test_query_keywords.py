from __future__ import annotations


def test_split_graph_query_keywords_separates_entity_and_relationship_terms():
    from services.rag_api.graph.query_keywords import split_graph_query_keywords

    result = split_graph_query_keywords(
        "地址迁移是否需要审核？",
        "合规审核",
        [],
        nodes=[
            {"id": "地址迁移", "label": "地址迁移"},
            {"id": "合规审核", "label": "合规审核"},
        ],
        edges=[
            {
                "source": "地址迁移",
                "target": "合规审核",
                "label": "审核关系",
                "description": "地址迁移需要合规审核",
            }
        ],
    )

    assert "地址迁移" in result["entity_keywords"]
    assert "合规审核" in result["entity_keywords"]
    assert "审核" in result["relationship_keywords"]
    assert result["fallback"] is False


def test_split_graph_query_keywords_extracts_topic_from_file_lookup():
    from services.rag_api.graph.query_keywords import split_graph_query_keywords

    result = split_graph_query_keywords(
        "关于一渠一表的报告在哪",
        "",
        [],
        nodes=[
            {"id": "一渠一表", "label": "一渠一表", "type": "主题实体"},
            {"id": "关于一渠一表调研报告.docx", "label": "关于一渠一表调研报告.docx", "type": "来源文件"},
        ],
        edges=[
            {
                "source": "关于一渠一表调研报告.docx",
                "target": "一渠一表",
                "label": "提及主题",
            }
        ],
    )

    assert "一渠一表" in result["entity_keywords"]
    assert "报告" in result["relationship_keywords"]
    assert "来源文件" in result["relationship_keywords"]


def test_split_graph_query_keywords_falls_back_to_query_when_no_terms_match():
    from services.rag_api.graph.query_keywords import split_graph_query_keywords

    result = split_graph_query_keywords("完全未知的问题", "", [], nodes=[], edges=[])

    assert result["entity_keywords"] == ["完全未知的问题"]
    assert result["relationship_keywords"] == ["完全未知的问题"]
    assert result["fallback"] is True
