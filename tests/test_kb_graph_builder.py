from __future__ import annotations


def test_build_kb_graph_creates_topic_file_and_category_nodes():
    from services.rag_api.graph.kb_graph_builder import build_kb_graph

    category_payload = {
        "items": [
            {
                "name": "资费咨询",
                "document_count": 2,
                "chunk_count": 12,
                "source_files": [
                    "V1.12024年中国联通“一渠一表”科技创新0808.pptx",
                    "基于“一渠一表”项目效益评价-20240531.docx",
                ],
                "keyword_hits": ["资费", "费用"],
            },
            {
                "name": "合规审核",
                "document_count": 1,
                "chunk_count": 3,
                "source_files": ["关于一渠一表调研报告.docx"],
                "keyword_hits": ["合规"],
            },
        ]
    }
    documents = [
        {"source_file": "V1.12024年中国联通“一渠一表”科技创新0808.pptx", "content": "一渠一表科技创新材料"},
        {"source_file": "基于“一渠一表”项目效益评价-20240531.docx", "content": "一渠一表项目效益评价"},
        {"source_file": "关于一渠一表调研报告.docx", "content": "一渠一表调研报告"},
    ]
    chunks = [
        {"metadata": {"source_file": "V1.12024年中国联通“一渠一表”科技创新0808.pptx"}},
        {"metadata": {"source_file": "基于“一渠一表”项目效益评价-20240531.docx"}},
        {"metadata": {"source_file": "关于一渠一表调研报告.docx"}},
    ]

    graph = build_kb_graph(category_payload, documents, chunks)

    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {(edge["source"], edge["label"], edge["target"]) for edge in graph["edges"]}
    assert nodes["一渠一表"]["type"] == "主题实体"
    assert nodes["一渠一表"]["source_files"] == [
        "V1.12024年中国联通“一渠一表”科技创新0808.pptx",
        "关于一渠一表调研报告.docx",
        "基于“一渠一表”项目效益评价-20240531.docx",
    ]
    assert nodes["一渠一表"]["evidence_count"] == 3
    assert nodes["关于一渠一表调研报告.docx"]["type"] == "来源文件"
    assert ("资费咨询", "包含文件", "V1.12024年中国联通“一渠一表”科技创新0808.pptx") in edges
    assert ("关于一渠一表调研报告.docx", "提及主题", "一渠一表") in edges
    assert ("一渠一表", "关联分类", "合规审核") in edges


def test_save_kb_graph_writes_dynamic_graph_payload(tmp_path):
    from services.rag_api.graph.kb_graph_builder import save_kb_graph

    path = tmp_path / "kb_graph.json"
    payload = {"nodes": [{"id": "一渠一表", "label": "一渠一表"}], "edges": []}

    saved = save_kb_graph(payload, path)

    assert saved["nodes"][0]["id"] == "一渠一表"
    assert path.read_text(encoding="utf-8").startswith("{")


def test_graph_payload_hides_static_graph_when_no_source_files(monkeypatch):
    from services.rag_api.graph import graph_api
    from services.rag_api.graph.graph_store import static_edges, static_nodes

    monkeypatch.setattr(graph_api, "load_raw_graph", lambda path=None: (static_nodes(), static_edges(), "static_graph"))
    monkeypatch.setattr(graph_api, "load_kb_categories", lambda: {"items": []})

    payload = graph_api.build_graph_payload()

    assert payload["nodes"] == []
    assert payload["edges"] == []
    assert payload["stats"]["node_count"] == 0
    assert payload["stats"]["edge_count"] == 0
    assert payload["stats"]["source_file_count"] == 0
    assert payload["stats"]["graph_source"] == "empty_graph"
    assert payload["stats"]["graph_source_label"] == "暂无知识图谱"
