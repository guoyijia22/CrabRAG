from __future__ import annotations

import json


def test_graph_relation_search_uses_dynamic_graph_and_prioritizes_source_file(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_search
    from services.rag_api.vector import chroma_store

    graph_file = tmp_path / "kb_graph.json"
    graph_file.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "企业客户", "label": "企业客户", "type": "客户类型"},
                    {"id": "合规审核", "label": "合规审核", "type": "审核规则"},
                ],
                "edges": [
                    {
                        "source": "企业客户",
                        "target": "合规审核",
                        "label": "地址迁移审核关系",
                        "description": "企业客户地址迁移必须重新进入合规审核。",
                        "evidence": "地址迁移需提交合同和资质材料后再审核。",
                        "source_file": "dynamic_rules.md",
                        "confidence": 0.92,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(graph_search, "KB_GRAPH_PATH", graph_file, raising=False)
    monkeypatch.setattr(
        chroma_store,
        "search_chunks_by_keywords",
        lambda query, intent, entities, top_k: [
            {
                "content": "普通资费材料说明。",
                "source_file": "general.md",
                "score": 0.95,
                "retrieval_channel": "keyword",
            },
            {
                "content": "地址迁移需提交合同和资质材料后再审核。",
                "source_file": "dynamic_rules.md",
                "score": 0.2,
                "retrieval_channel": "keyword",
            },
        ],
    )

    result = graph_search.graph_relation_search("企业客户地址迁移需要审核吗？", "合规审核", top_k=2)

    assert result["relation_paths"][0]["path"] == "企业客户 -> 地址迁移审核关系 -> 合规审核"
    assert result["relation_paths"][0]["source_file"] == "dynamic_rules.md"
    assert result["relation_paths"][0]["evidence"] == "地址迁移需提交合同和资质材料后再审核。"
    assert result["relation_paths"][0]["graph_source"] == "dynamic_graph"
    assert result["chunks"][0]["source_file"] == "dynamic_rules.md"


def test_graph_relation_search_falls_back_to_static_graph_when_dynamic_graph_missing(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_search
    from services.rag_api.vector import chroma_store

    monkeypatch.setattr(graph_search, "KB_GRAPH_PATH", tmp_path / "missing-kb-graph.json", raising=False)
    monkeypatch.setattr(chroma_store, "search_chunks_by_keywords", lambda query, intent, entities, top_k: [])

    result = graph_search.graph_relation_search("企业客户地址迁移需要审核吗？", "合规审核", top_k=3)

    assert any("合规审核" in item["path"] for item in result["relation_paths"])
    assert all(item.get("graph_source") != "dynamic_graph" for item in result["relation_paths"])


def test_graph_relation_search_uses_entity_and_relationship_vector_hits(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_search
    from services.rag_api.vector import chroma_store

    graph_file = tmp_path / "kb_graph.json"
    graph_file.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "一渠一表", "label": "一渠一表", "type": "主题实体"},
                    {"id": "合规审核", "label": "合规审核", "type": "知识分类"},
                    {"id": "关于一渠一表调研报告.docx", "label": "关于一渠一表调研报告.docx", "type": "来源文件"},
                ],
                "edges": [
                    {
                        "source": "关于一渠一表调研报告.docx",
                        "target": "一渠一表",
                        "label": "提及主题",
                        "description": "报告提及一渠一表",
                        "source_file": "关于一渠一表调研报告.docx",
                        "confidence": 0.85,
                    },
                    {
                        "source": "一渠一表",
                        "target": "合规审核",
                        "label": "关联分类",
                        "description": "一渠一表关联合规审核",
                        "source_file": "关于一渠一表调研报告.docx",
                        "confidence": 0.8,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(graph_search, "KB_GRAPH_PATH", graph_file, raising=False)
    monkeypatch.setattr(graph_search, "search_graph_entities", lambda query, top_k: [{"id": "一渠一表", "score": 0.93}])
    monkeypatch.setattr(
        graph_search,
        "search_graph_relationships",
        lambda query, top_k: [{"source": "一渠一表", "target": "合规审核", "label": "关联分类", "score": 0.91}],
    )
    monkeypatch.setattr(
        chroma_store,
        "search_chunks_by_keywords",
        lambda query, intent, entities, top_k: [
            {
                "content": "关于一渠一表调研报告说明。",
                "source_file": "关于一渠一表调研报告.docx",
                "score": 0.7,
                "retrieval_channel": "keyword",
            }
        ],
    )

    result = graph_search.graph_relation_search("关于一渠一表的报告在哪", "", top_k=3)

    match_sources = {item["match_source"] for item in result["relation_paths"]}
    assert "entity_vector" in match_sources
    assert "relationship_vector" in match_sources
    assert result["chunks"][0]["source_file"] == "关于一渠一表调研报告.docx"
    assert any(item["node"] == "graph_query_keywords" for item in result["trace"])
    assert any(item["node"] == "graph_mix_search" for item in result["trace"])


def test_graph_relation_search_falls_back_to_dynamic_literal_when_graph_vectors_fail(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_search
    from services.rag_api.vector import chroma_store

    graph_file = tmp_path / "kb_graph.json"
    graph_file.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "企业客户", "label": "企业客户", "type": "客户类型"},
                    {"id": "合规审核", "label": "合规审核", "type": "审核规则"},
                ],
                "edges": [
                    {
                        "source": "企业客户",
                        "target": "合规审核",
                        "label": "地址迁移审核关系",
                        "description": "企业客户地址迁移必须重新进入合规审核。",
                        "source_file": "dynamic_rules.md",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(graph_search, "KB_GRAPH_PATH", graph_file, raising=False)
    monkeypatch.setattr(graph_search, "search_graph_entities", lambda query, top_k: (_ for _ in ()).throw(RuntimeError("missing graph vdb")))
    monkeypatch.setattr(graph_search, "search_graph_relationships", lambda query, top_k: [])
    monkeypatch.setattr(chroma_store, "search_chunks_by_keywords", lambda query, intent, entities, top_k: [])

    result = graph_search.graph_relation_search("企业客户地址迁移需要审核吗？", "合规审核", top_k=2)

    assert result["relation_paths"][0]["match_source"] == "dynamic_literal"
    assert result["relation_paths"][0]["graph_source"] == "dynamic_graph"
