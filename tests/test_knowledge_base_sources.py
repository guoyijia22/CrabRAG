from __future__ import annotations

from pathlib import Path

import pytest

from services.rag_api.exceptions import DocumentLoadError


def test_app_settings_default_knowledge_base_dirs(tmp_path, monkeypatch):
    from services.rag_api import app_settings

    project_root = tmp_path / "project"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")

    settings = app_settings.load_app_settings()

    assert settings.knowledge_base_dirs == []


def test_app_settings_preserves_empty_knowledge_base_dirs(tmp_path, monkeypatch):
    from services.rag_api import app_settings

    project_root = tmp_path / "project"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")

    saved = app_settings.save_app_settings(app_settings.AppSettings(knowledge_base_dirs=[]))

    assert saved.knowledge_base_dirs == []


def test_runtime_settings_allows_empty_docs_dirs(tmp_path, monkeypatch):
    from services.rag_api import app_settings, config

    project_root = tmp_path / "project"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")
    monkeypatch.setattr(config, "DEFAULT_DOCS_DIR", (project_root / "docs").resolve())
    app_settings.save_app_settings(app_settings.AppSettings(knowledge_base_dirs=[]))
    config.get_settings.cache_clear()

    settings = config.get_settings()

    assert settings.docs_dirs == []
    assert settings.docs_dir == (project_root / "docs").resolve()


def test_app_settings_saves_cleaned_knowledge_base_dirs(tmp_path, monkeypatch):
    from services.rag_api import app_settings

    project_root = tmp_path / "project"
    first = tmp_path / "docs-a"
    second = tmp_path / "docs-b"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")

    saved = app_settings.save_app_settings(
        app_settings.AppSettings(
            knowledge_base_dirs=[
                f"  {first}  ",
                "",
                str(first),
                str(second),
            ]
        )
    )

    assert saved.knowledge_base_dirs == [str(first.resolve()), str(second.resolve())]


def test_runtime_settings_exposes_multiple_docs_dirs(tmp_path, monkeypatch):
    from services.rag_api import app_settings, config

    project_root = tmp_path / "project"
    first = tmp_path / "docs-a"
    second = tmp_path / "docs-b"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")
    app_settings.save_app_settings(app_settings.AppSettings(knowledge_base_dirs=[str(first), str(second)]))
    config.get_settings.cache_clear()

    settings = config.get_settings()

    assert settings.docs_dirs == [first.resolve(), second.resolve()]
    assert settings.docs_dir == first.resolve()


def test_load_documents_reads_multiple_dirs_recursively(tmp_path):
    from services.rag_api.document.loader import load_documents

    first = tmp_path / "first"
    second = tmp_path / "second"
    nested = second / "nested"
    first.mkdir()
    nested.mkdir(parents=True)
    (first / "a.txt").write_text("客户准入材料要求", encoding="utf-8")
    (nested / "b.txt").write_text("资费套餐办理流程", encoding="utf-8")

    docs = load_documents([first, second])

    assert [doc["source_file"] for doc in docs] == ["a.txt", "b.txt"]
    assert any("客户准入" in doc["content"] for doc in docs)
    assert any("资费套餐" in doc["content"] for doc in docs)


def test_load_documents_reads_csv_xlsx_xlsm_and_pptx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    pptx = pytest.importorskip("pptx")
    from services.rag_api.document.loader import load_documents

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "rules.csv").write_text("类别,内容\n资费,套餐价格说明\n", encoding="utf-8")

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "准入"
    sheet.append(["字段", "要求"])
    sheet.append(["营业执照", "必须提供"])
    workbook.save(docs_dir / "table.xlsx")
    workbook.save(docs_dir / "macro.xlsm")

    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "投诉处理"
    textbox = slide.shapes.add_textbox(0, 0, 1000000, 1000000)
    textbox.text = "处理时限为 48 小时"
    presentation.save(docs_dir / "slides.pptx")

    docs = load_documents(docs_dir)
    content_by_file = {doc["source_file"]: doc["content"] for doc in docs}

    assert "套餐价格说明" in content_by_file["rules.csv"]
    assert "营业执照" in content_by_file["table.xlsx"]
    assert "必须提供" in content_by_file["macro.xlsm"]
    assert "投诉处理" in content_by_file["slides.pptx"]
    assert "48 小时" in content_by_file["slides.pptx"]


def test_load_documents_raises_when_no_configured_dirs_have_files(tmp_path):
    from services.rag_api.document.loader import load_documents

    with pytest.raises(DocumentLoadError):
        load_documents([tmp_path / "missing-a", tmp_path / "missing-b"])


def test_health_reports_empty_docs_directory(tmp_path, monkeypatch):
    from services.rag_api.config import Settings
    from services.rag_api import main

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    monkeypatch.setattr(main, "get_settings", lambda: Settings(docs_dirs=[docs_dir], docs_dir=docs_dir))
    monkeypatch.setattr(main, "collection_status", lambda: {"count": 0})

    payload = main.health()

    assert payload["docs_dir_exists"] is True
    assert payload["docs_dir_has_files"] is False

    (docs_dir / "a.txt").write_text("hello", encoding="utf-8")

    payload = main.health()

    assert payload["docs_dir_exists"] is True
    assert payload["docs_dir_has_files"] is True


def test_graph_payload_returns_empty_graph_when_no_dynamic_graph_or_sources(tmp_path, monkeypatch):
    from services.rag_api.graph import graph_api, graph_store

    missing_graph = tmp_path / "missing-kb-graph.json"
    monkeypatch.setattr(graph_api, "KB_GRAPH_PATH", missing_graph)
    monkeypatch.setattr(graph_store, "KB_GRAPH_PATH", missing_graph)
    monkeypatch.setattr(graph_api, "load_kb_categories", lambda: {"items": [], "categories": []})

    payload = graph_api.build_graph_payload()

    assert payload["nodes"] == []
    assert payload["edges"] == []
    assert payload["stats"]["graph_source"] == "empty_graph"
    assert payload["stats"]["node_count"] == 0
    assert payload["stats"]["edge_count"] == 0


def test_kb_categories_empty_when_no_category_file(tmp_path, monkeypatch):
    from services.rag_api.document import categories

    monkeypatch.setattr(categories, "KB_CATEGORIES_PATH", tmp_path / "missing-kb-categories.json")

    payload = categories.load_kb_categories()

    assert payload["items"] == []
    assert payload["categories"] == []
    assert categories.get_category_names() == []


def test_kb_categories_empty_when_category_file_has_no_items(tmp_path, monkeypatch):
    from services.rag_api.document import categories

    categories_path = tmp_path / "kb_categories.json"
    categories_path.write_text('{"items": [], "categories": ["客户准入"], "generated_at": "2026-01-01 00:00:00"}', encoding="utf-8")
    monkeypatch.setattr(categories, "KB_CATEGORIES_PATH", categories_path)

    payload = categories.load_kb_categories()

    assert payload["items"] == []
    assert payload["categories"] == []
    assert categories.get_category_names() == []


def test_kb_categories_returns_real_categories_when_present(tmp_path, monkeypatch):
    from services.rag_api.document import categories

    categories_path = tmp_path / "kb_categories.json"
    categories_path.write_text(
        '{"items": [{"name": "一渠一表", "document_count": 1, "chunk_count": 2, "source_files": ["a.docx"], "keyword_hits": []}], "categories": ["一渠一表"], "generated_at": "2026-01-01 00:00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(categories, "KB_CATEGORIES_PATH", categories_path)

    payload = categories.load_kb_categories()

    assert payload["categories"] == ["一渠一表"]
    assert categories.get_category_names() == ["一渠一表"]


def test_heuristic_classify_does_not_emit_default_category_for_empty_kb(monkeypatch):
    from services.rag_api.agent import heuristics

    monkeypatch.setattr(heuristics, "get_category_names", lambda: [])

    payload = heuristics.heuristic_classify("企业客户需要什么材料？", [], [])

    assert payload["intent"] == ""
    assert payload["question_type"] == "单一规则"
    assert payload["retrieval_mode"] == "vector"


def test_ingest_uses_multiple_docs_dirs_and_returns_all_dirs(tmp_path, monkeypatch):
    from services.rag_api import index_generation
    from services.rag_api.config import Settings
    from services.rag_api.document import doc_status, ingest
    from services.rag_api.rag_settings import RagSettings

    first = tmp_path / "docs-a"
    second = tmp_path / "docs-b"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("客户准入材料", encoding="utf-8")
    captured = {}
    index_root = tmp_path / "data" / "index"
    monkeypatch.setattr(index_generation, "INDEX_ROOT", index_root)
    monkeypatch.setattr(index_generation, "ACTIVE_INDEX_PATH", index_root / "active.json")
    monkeypatch.setattr(index_generation, "GENERATIONS_DIR", index_root / "generations")
    monkeypatch.setattr(doc_status, "DOC_STATUS_PATH", tmp_path / "data" / "doc_status.json")
    monkeypatch.setattr(doc_status, "DOC_SNAPSHOT_DIR", tmp_path / "data" / "snapshots")

    def fake_scan_supported_files(dirs):
        captured["dirs"] = dirs
        return [first / "a.txt"]

    monkeypatch.setattr(ingest, "get_settings", lambda: Settings(docs_dirs=[first, second], docs_dir=first))
    monkeypatch.setattr(ingest, "load_rag_settings", lambda: RagSettings())
    monkeypatch.setattr(ingest, "scan_supported_files", fake_scan_supported_files)
    monkeypatch.setattr(ingest, "load_document", lambda path: {"source_file": "a.txt", "source_path": str(path), "content": "客户准入材料"})
    monkeypatch.setattr(ingest, "split_documents", lambda documents, chunk_size, chunk_overlap: [{"id": "1", "content": "客户准入材料", "metadata": {"source_file": "a.txt", "category": "客户准入"}}])
    monkeypatch.setattr(ingest, "expand_multi_vector_chunks", lambda chunks, rag_settings: chunks)
    monkeypatch.setattr(ingest, "embedding_batch_count", lambda count: 1)
    monkeypatch.setattr(
        ingest,
        "build_generation_chunks",
        lambda chunks, generation_id, full_rebuild=False, progress_callback=None: {
            "chunk_count": len(chunks),
            "reused_embedding_count": 0,
            "embedded_chunk_count": len(chunks),
        },
    )
    monkeypatch.setattr(ingest, "save_kb_categories", lambda documents, chunks, path=None: {"items": [], "categories": []})
    monkeypatch.setattr(ingest, "read_app_config", lambda: (_ for _ in ()).throw(AssertionError("rebuild must not read common questions")), raising=False)
    monkeypatch.setattr(ingest, "generate_common_questions", lambda category_payload: (_ for _ in ()).throw(AssertionError("rebuild must not generate common questions")), raising=False)
    monkeypatch.setattr(ingest, "write_common_questions", lambda questions: (_ for _ in ()).throw(AssertionError("rebuild must not overwrite common questions")), raising=False)
    monkeypatch.setattr(ingest, "ensure_knowledge_base_name", lambda category_payload, documents, chunk_count: ("测试知识库", "test"))
    monkeypatch.setattr(ingest, "generate_graph_schema_suggestion", lambda category_payload, documents, chunks, path=None: {})
    monkeypatch.setattr(ingest, "build_and_save_kb_graph", lambda category_payload, documents, chunks, path=None: {"nodes": [{"id": "客户准入"}], "edges": [{"id": "edge-1"}]}, raising=False)
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_generation",
        lambda nodes, edges, generation_id: {"graph_entity_index_count": len(nodes), "graph_relationship_index_count": len(edges)},
    )

    result = ingest.ingest_knowledge_base()

    assert captured["dirs"] == [first.resolve(), second.resolve()]
    assert result["kb_dir"] == str(first.resolve())
    assert result["kb_dirs"] == [str(first.resolve()), str(second.resolve())]
    assert "common_questions" not in result
    assert "common_question_count" not in result
    assert result["graph_node_count"] == 1
    assert result["graph_edge_count"] == 1
    assert result["graph_entity_index_count"] == 1
    assert result["graph_relationship_index_count"] == 1


def test_frontend_ingest_result_uses_rebuild_summary():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "JSON.stringify(n,null,2)" not in bundle
    assert "重建摘要" in bundle
    assert "graph_node_count" in bundle
    assert "graph_edge_count" in bundle


def test_frontend_empty_docs_directory_status_is_localized():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "`无文件`,`No files`" in bundle
    assert "docs_dir_has_files?`存在`:`无文件`" in bundle
    assert "docs_dir_has_files?`已找到`:`无文件`" in bundle


def test_frontend_chat_input_default_empty_and_enter_submit():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "企业客户办理地址迁移时，是否需要重新进行合规审核？" not in bundle
    assert "[m,h]=(0,C.useState)(``)" in bundle
    assert "placeholder:`输入问题，或输入/选择常用问题`" in bundle
    assert "输入政企专线业务问题" not in bundle
    assert "function s(e){if(e.key===`Enter`&&!e.shiftKey&&!e.altKey){" in bundle
    assert "onKeyDown:s" in bundle
    assert "e.currentTarget.value.trim()" in bundle


def test_frontend_default_app_settings_have_empty_knowledge_base_dirs():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "knowledge_base_dirs:[]" in bundle
    assert "knowledge_base_dirs:[`docs`]" not in bundle


def test_frontend_business_scope_defaults_are_general():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "scope_min_score:0" in bundle
    assert "in_scope_keywords:[]" in bundle
    assert "out_of_scope_keywords:[`股票`,`Stock`]" in bundle
    assert "General knowledge base assistant for local documents." in bundle
    assert "通用本地知识库助手" in bundle


def test_frontend_knowledge_base_has_incremental_and_full_rebuild_controls():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "`新增知识库`,`Add knowledge base`" in bundle
    assert "`知识库重建`,`Full rebuild knowledge base`" in bundle
    assert "/api/ingest/full" in bundle
    assert "confirm(" in bundle
    assert "全量重建" in bundle
    assert "crabrag:knowledge-base-rebuilt" in bundle


def test_frontend_knowledge_base_empty_dirs_prompt_is_localized():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "请先在设置页面的“知识库读取目录（每行一个）”配置文件目录" in bundle
    assert "Configure file directories in Settings > Knowledge base directories (one per line)" in bundle
    assert "(e.docs_dirs??[]).length" in bundle
    assert "D:\\\\cd\\\\docs" not in bundle


def test_frontend_chat_category_list_starts_empty_and_refreshes_after_rebuild():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "children:f.map(e=>" in bundle
    assert "children:(f.length>0?f:N_()).map" not in bundle
    assert "addEventListener(`crabrag:knowledge-base-rebuilt`" in bundle
    assert "removeEventListener(`crabrag:knowledge-base-rebuilt`" in bundle


def test_frontend_graph_main_initial_layout_uses_radial_disk_layout():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "function crGraphRadialDiskLayout(" in bundle
    assert "A=i?crGraphRadialDiskLayout(h):crGraphLayout()" in bundle
    assert "e.layout(i?crGraphRadialDiskLayout(h):crGraphLayout()).run()" in bundle
    assert "e.layout(crGraphLayout()).run()" not in bundle
    assert "A=i?crGraphCircleLayout():crGraphLayout()" not in bundle
    assert "A=i?crGraphClusterLayout():crGraphLayout()" not in bundle
    assert "function crGraphClusterLayout()" not in bundle
    assert (
        "function crGraphRadialDiskLayout(e){let t=crGraphApplyDiskRanks(e);return{name:`preset`"
        in bundle
    )
    assert "avoidOverlap:!0" in bundle
    assert "function crGraphDiskRingCount(" in bundle
    assert "e<=12?2:e<=35?3:e<=90?5:e<=180?6:7" in bundle
    assert "function crGraphDiskPadding(" in bundle
    assert "e<=12?160:e<=35?130:e<=90?100:e<=180?78:58" in bundle
    assert "function crGraphDiskHash(" in bundle
    assert "function crGraphDiskGroupKey(" in bundle
    assert "function crGraphPlaceDiskNodes(" in bundle
    assert "diskRank" in bundle
    assert "diskPosition" in bundle
    assert "positions:e=>e.data(`diskPosition`)||{x:0,y:0}" in bundle
    assert "Math.atan2" in bundle
    assert "Math.cos" in bundle
    assert "Math.sin" in bundle
    assert "minGap" in bundle
    assert "concentric:e=>e.degree()" not in bundle
    assert "concentric:e=>e.data(`diskRank`)||0" not in bundle
    assert "levelWidth:()=>1" not in bundle
    assert "renderer:{name:`canvas`}" in bundle
    assert "Math.max(20,Math.min(44,22+Math.sqrt(r)*4))" in bundle
    assert "Math.max(26,Math.min(62,28+Math.sqrt(r)*7))" not in bundle
    assert "Math.max(330,Math.ceil(e.length*minGap/(Math.PI*2)))" not in bundle


def test_frontend_settings_uses_new_retrieval_module_terms():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "RAG 检索优化与参数" in bundle
    assert "多粒度文本索引" in bundle
    assert "统一重排" in bundle
    assert "多轮追问重写" in bundle
    assert "max_context_tokens" in bundle
    assert "上下文 Token 预算" in bundle
    assert "混合检索" not in bundle
    assert "核心参数调优" not in bundle
    assert "RAG 参数调优" not in bundle
    assert "BM25 权重" not in bundle
    assert "向量权重" not in bundle


def test_frontend_settings_shows_local_model_download_guidance():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    assert "function crLocalModelDownloadUrl(" in bundle
    assert "function crLocalModelStatusPanel(" in bundle
    assert "local_model_status" in bundle
    assert "missing_count" in bundle
    assert "download_urls" in bundle
    assert "expected_dir" in bundle
    assert "missing_files" in bundle
    assert "本地模型文件已检测到" in bundle
    assert "Local model files detected" in bundle
    assert "缺失本地模型文件" in bundle
    assert "Missing local model files" in bundle
    assert "存放目录：" in bundle
    assert "Save to:" in bundle
    assert "缺失文件：" in bundle
    assert "Missing files:" in bundle
    assert "下载地址（ModelScope）" in bundle
    assert "Download (Hugging Face)" in bundle
    assert "https://www.modelscope.cn/models/onnx-community/Qwen3.5-0.8B-ONNX" in bundle
    assert "https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX" in bundle
    assert "https://www.modelscope.cn/models/onnx-community/Qwen3-Embedding-0.6B-ONNX" in bundle
    assert "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX" in bundle
    assert "https://www.modelscope.cn/models/onnx-community/Qwen3-Reranker-0.6B-ONNX" in bundle
    assert "https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX" in bundle
    assert "a.use_local_models&&(0,q.jsx)(crLocalModelStatusPanel,{status:a.local_model_status})" in bundle


def test_frontend_english_language_covers_async_static_labels():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    expected_translations = {
        "内置基础图谱": "Built-in base graph",
        "暂无知识图谱": "No knowledge graph",
        "问答命中子图": "Q&A matched subgraph",
        "知识库动态图谱": "Dynamic knowledge graph",
        "根据知识库分类生成的兜底结构，建议展示实体类型、业务类别、来源文件、证据数量和关系来源。": "Fallback structure generated from knowledge-base categories. Show entity types, business categories, source files, evidence counts, and relation sources.",
        "已加载当前设置": "Settings loaded",
        "大语言模型 Key：": "LLM Key:",
        "Embedding Key：": "Embedding Key:",
        "Rerank Key：": "Rerank Key:",
        "多粒度文本索引": "Multi-granularity text index",
        "重建知识库时生成 document / paragraph / sentence 多粒度文本索引。": "Generate document / paragraph / sentence multi-granularity text indexes when rebuilding the knowledge base.",
        "查询扩展": "Query expansion",
        "生成等价查询，增强文本向量召回覆盖。": "Generate equivalent queries to improve text vector recall coverage.",
        "对 vector / graph / keyword 融合后的候选片段统一精排。": "Rerank the fused vector / graph / keyword candidates together.",
        "多轮聊天中先把追问改写为独立完整查询。": "Rewrite follow-up questions into standalone complete queries in multi-turn chats.",
        "RAG 评测对比": "RAG evaluation comparison",
        "每次评测都会基于当前知识库动态出题，再批量运行基线、单项优化和组合配置，比较召回、溯源、图谱路径、兜底与 Trace 差异。": "Each evaluation dynamically generates questions from the current knowledge base, then runs the baseline, single optimizations, and combined profiles to compare recall, traceability, graph paths, fallback behavior, and trace differences.",
        "一次消耗300-500K,60分钟左右。": "Each run consumes about 300-500K tokens and takes around 60 minutes.",
        "点击“运行评测”后，系统会先根据当前知识库生成题集，再依次执行基线、单项优化和组合配置。完成后会在这里展示哪些配置更适用、哪些配置效果接近或有风险。": "After clicking Run evaluation, the system first generates a question set from the current knowledge base, then runs the baseline, single optimizations, and combined profiles. Results here show which configurations fit better, which are close, and which may be risky.",
        "基于本地规范知识库、Chroma、GraphRAG 与 LangGraph 三节点流转": "Powered by the local knowledge base, Chroma, GraphRAG, and a three-node LangGraph flow",
        "输入问题后，系统会展示分类、检索路径、规范片段和可溯源答复。": "After you ask a question, the system shows classification, retrieval paths, knowledge snippets, and traceable answers.",
        "关系子图": "Relation subgraph",
        "检索来源": "Retrieved sources",
        "图谱路径": "Graph path",
        "已生成新的图谱结构建议，确认后生效": "Generated a new graph structure suggestion. Confirm it to apply.",
        "上次重建知识库花费时间：": "Last knowledge-base rebuild duration: ",
        "Chroma向量数据库": "Chroma vector database",
        "Embedding 向量化": "Embedding",
        "Embedding 模型": "Embedding model",
        "Rerank 排序": "Rerank",
        "Rerank 模型": "Rerank model",
        "；大语言模型：Qwen3.5-0.8B-ONNX（LLM 本地生成已启用，首次回答需加载模型，可能较慢）；向量模型：Qwen3-Embedding-0.6B-ONNX / onnx/model_int8.onnx；Reranker：Qwen3-Reranker-0.6B-ONNX / onnx/model_q4.onnx；切换本地向量模型后需手动重建知识库。": "; LLM: Qwen3.5-0.8B-ONNX (local LLM generation is enabled; the first answer may be slower while the model loads); embedding model: Qwen3-Embedding-0.6B-ONNX / onnx/model_int8.onnx; Reranker: Qwen3-Reranker-0.6B-ONNX / onnx/model_q4.onnx; rebuild the knowledge base manually after switching local embedding models.",
        "实体ID": "Entity ID",
        "实体名称": "Entity name",
        "风险等级": "Risk level",
        "起点实体": "Source entity",
        "终点实体": "Target entity",
        "关系名称": "Relation name",
        "主题实体": "Topic entity",
        "知识分类": "Knowledge category",
        "节点": "Nodes",
        "边": "Edges",
        "扫描文档": "Scanning documents",
        "读取解析": "Reading and parsing",
        "清洗切片": "Cleaning and chunking",
        "多粒度处理": "Multi-granularity processing",
        "写入向量库": "Writing vector store",
        "生成分类": "Generating categories",
        "生成图谱索引": "Generating graph index",
        "生成知识图谱": "Generating knowledge graph",
        "开始重建": "Starting rebuild",
        "完成": "Completed",
        "失败": "Failed",
        "知识库重建完成": "Knowledge-base rebuild completed",
        "知识库重建失败": "Knowledge-base rebuild failed",
        "知识库文档加载失败：未找到可读取的知识库目录或文件。请在设置页配置至少一个有效读取目录，并确认目录中存在 txt、docx、pdf、xlsx、xlsm、csv 或 pptx 文档。": "Knowledge-base document load failed: no readable knowledge-base directory or file was found. Configure at least one valid read directory in Settings and make sure it contains txt, docx, pdf, xlsx, xlsm, csv, or pptx documents.",
    }
    for chinese, english in expected_translations.items():
        assert f"[`{chinese}`,`{english}`]" in bundle

    assert "MutationObserver" in bundle
    assert "crScheduleLanguageApply" in bundle
    assert "crLangApplying" in bundle
    assert "function crFormatLocalizedProgress(" in bundle
    assert "Scanning and reading knowledge-base directories" in bundle
    assert "Read $1 documents, preparing chunks" in bundle
    assert "Generated $1 base chunks" in bundle
    assert "Processed $1 documents, skipped $2, removed $3, generated $4 base chunks" in bundle
    assert "$1 chunks pending indexing" in bundle
    assert "Wrote $1 Chroma vector chunks" in bundle
    assert "Generated $1 knowledge-base categories" in bundle
    assert "Knowledge-base rebuild task created, waiting for background execution" in bundle
    assert "向量化中[：:]" in bundle
    assert "本地向量化中[：:]" in bundle
    assert "Embedding: batch $1 / $2, processed $3 / $4 chunks" in bundle
    assert "Local embedding: batch $1 / $2, processed $3 / $4 chunks" in bundle
    assert "Indexed $1 entity vectors and $2 relationship vectors" in bundle
    assert "crFormatLocalizedProgress(r,e)" in bundle
    assert "function crFormatLocalizedDuration(" in bundle
    assert "function crCurrentLanguage(" in bundle
    assert "function crLocalizeText(" in bundle
    assert "crLocalizeText(e.message||``,n)" in bundle
    assert "crLocalizeText(e.current_step||`准备中`,n)" in bundle
    assert "children:crLocalizeText(e.error,n)" in bundle
    assert (
        "e.duration_label&&e.status!==`running`&&(0,q.jsxs)(`p`,{className:`muted`,children:[crLocalizeText(`本次耗时：`,n),u]}),e.error&&(0,q.jsx)(`div`,{className:`progress-error`,children:crLocalizeText(e.error,n)})"
        in bundle
    )
    assert "children:e.message" not in bundle
    assert "children:e.current_step||`准备中`" not in bundle
    assert bundle.index("crFormatLocalizedProgress(r,e)") < bundle.index("crFormatLocalizedDuration(r,e)")
    assert ".replace(/(\\d+)小时/g,`$1h `)" in bundle
    assert ".replace(/(\\d+)分/g,`$1m `)" in bundle
    assert ".replace(/(\\d+)秒/g,`$1s`)" in bundle
    assert "contains(`conversation`)" not in bundle
    assert "message" in bundle
    assert "source-item" in bundle
    assert "path-item" in bundle


def test_frontend_evaluation_english_language_covers_dynamic_results():
    bundle = Path("apps/web/dist/assets/index-CKowSniJ.js").read_text(encoding="utf-8")

    expected_translations = {
        "查询扩展 + 统一重排": "Query expansion + unified rerank",
        "多轮追问重写 + 统一重排": "Multi-turn follow-up rewrite + unified rerank",
        "多粒度文本索引 + 统一重排": "Multi-granularity text index + unified rerank",
        "全增强配置": "All enhancements",
        "类别命中率": "Category hit rate",
        "来源命中率": "Source hit rate",
        "图谱路径覆盖率": "Graph path coverage",
        "平均耗时": "Average latency",
        "相对基线变化": "Change from baseline",
        "预期字段": "Expected fields",
        "参考片段": "Reference snippets",
        "无错误": "No error",
        "规则兜底生成": "rule fallback",
        "LLM 动态生成": "LLM generated",
        "效果接近": "Close result",
        "建议适用": "Recommended",
        "不建议适用/有风险": "Not recommended / risky",
        "暂无最佳配置说明": "No best-configuration summary yet",
        "正在生成动态题集并启动批量评测": "Generating dynamic question set and starting batch evaluation",
        "评测任务已启动": "Evaluation task started",
        "读取评测进度失败，请确认 Python RAG 服务正在运行": "Failed to read evaluation progress. Confirm the Python RAG service is running.",
        "评测启动失败：请确认 Python RAG 服务已启动，且知识库已完成入库": "Failed to start evaluation. Confirm the Python RAG service is running and the knowledge base is indexed.",
    }
    for chinese, english in expected_translations.items():
        assert f"[`{chinese}`,`{english}`]" in bundle

    assert "function crFormatLocalizedEvaluation(" in bundle
    assert "crFormatLocalizedEvaluation(n,t)" in bundle
    assert "题集生成[：:]" in bundle
    assert "Question set: $1, $2 questions, $3 categories, generated at $4" in bundle
    assert "质量优先综合分" in bundle
    assert "Quality-first score $1, success rate $2, source hit rate $3, graph path coverage $4." in bundle
    assert "本次题集由规则兜底生成" in bundle
    assert "This question set was generated by rule fallback. Monitor LLM question generation availability in future runs." in bundle
    assert "本次题集由 LLM 动态生成" in bundle
    assert "This question set was generated dynamically by the LLM." in bundle
    assert "启用\\s*(\\d+)\\s*项增强" in bundle
    assert "Enable $1 enhancements" in bundle
    assert "类别\\s*(.*?)\\s*\\/\\s*引用\\s*(\\d+)\\s*\\/\\s*耗时\\s*(\\d+)\\s*ms\\s*\\/\\s*(.*)$" in bundle
    assert "Category $1 / citations $2 / latency $3 ms / $4" in bundle
    assert "function crEvalText(" in bundle
    assert "function crEvalDuration(" in bundle
    assert "crEvalText(e.name)" in bundle
    assert "crEvalText(e.summary.recommendation??`适用建议`)" in bundle
    assert "crEvalText(y?.name??`优化项`)" in bundle
    assert "var P_=`baseline_new_retrieval`" in bundle
    assert "baseline_all_off" not in bundle
    assert "S.baseCase&&(0,q.jsx)(H_,{title:crEvalText(`标准新链路基线`),item:S.baseCase})" in bundle
    assert "S.selectedCase&&(0,q.jsx)(H_,{title:crEvalText(y?.name??`优化项`),item:S.selectedCase})" in bundle
    assert "(0,q.jsx)(H_,{title:`标准新链路基线`,item:S.baseCase})" not in bundle
    assert "(0,q.jsx)(`strong`,{children:e.name}),(0,q.jsx)(`small`,{children:G_(e.enabled_switches)})" not in bundle
    assert "value:e.id,children:e.name},e.id))" not in bundle
    assert "(0,q.jsx)(`strong`,{children:e.name}),(0,q.jsx)(`span`,{children:e.summary.recommendation})" not in bundle
    assert "children:e.summary.recommendation" not in bundle
    assert "children:[`类别 `,t.category" not in bundle
    assert "return n<60?`${n} 秒`:`${Math.floor(n/60)} 分 ${n%60} 秒`" not in bundle
    assert "current_question||crEvalText(e.message)" in bundle
