from __future__ import annotations

import json
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


def test_app_settings_api_refreshes_cached_runtime_docs_dirs(tmp_path, monkeypatch):
    from services.rag_api import app_settings, config, main

    project_root = tmp_path / "project"
    first = tmp_path / "docs-a"
    second = tmp_path / "docs-b"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")
    app_settings.save_app_settings(app_settings.AppSettings(knowledge_base_dirs=[str(first)]))
    config.get_settings.cache_clear()
    assert config.get_settings().docs_dirs == [first.resolve()]

    main.update_app_settings(app_settings.AppSettings(knowledge_base_dirs=[str(second)]))

    assert config.get_settings().docs_dirs == [second.resolve()]
    config.get_settings.cache_clear()


def test_app_settings_api_does_not_clear_runtime_cache_when_save_fails(monkeypatch):
    from services.rag_api import app_settings, main

    cleared = []

    def fail_save(settings):
        raise OSError("save failed")

    monkeypatch.setattr(main, "save_app_settings", fail_save)
    monkeypatch.setattr(main.get_settings, "cache_clear", lambda: cleared.append(True))

    with pytest.raises(OSError, match="save failed"):
        main.update_app_settings(app_settings.AppSettings())

    assert cleared == []


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
    def save_categories(documents, chunks, path=None):
        if path is not None:
            path.write_text('{"items": [], "categories": []}', encoding="utf-8")
        return {"items": [], "categories": []}

    monkeypatch.setattr(ingest, "save_kb_categories", save_categories)
    monkeypatch.setattr(ingest, "read_app_config", lambda: (_ for _ in ()).throw(AssertionError("rebuild must not read common questions")), raising=False)
    monkeypatch.setattr(ingest, "generate_common_questions", lambda category_payload: (_ for _ in ()).throw(AssertionError("rebuild must not generate common questions")), raising=False)
    monkeypatch.setattr(ingest, "write_common_questions", lambda questions: (_ for _ in ()).throw(AssertionError("rebuild must not overwrite common questions")), raising=False)
    monkeypatch.setattr(ingest, "ensure_knowledge_base_name", lambda category_payload, documents, chunk_count: ("测试知识库", "test"))
    def save_schema(category_payload, documents, chunks, path=None):
        if path is not None:
            path.write_text("{}", encoding="utf-8")
        return {}

    def save_graph(category_payload, documents, chunks, path=None):
        payload = {"nodes": [{"id": "客户准入"}], "edges": [{"id": "edge-1"}]}
        if path is not None:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    monkeypatch.setattr(ingest, "generate_graph_schema_suggestion", save_schema)
    monkeypatch.setattr(ingest, "build_and_save_kb_graph", save_graph, raising=False)
    monkeypatch.setattr(
        ingest,
        "index_graph_vectors_generation",
        lambda nodes, edges, generation_id, full_rebuild=False: {"graph_entity_index_count": len(nodes), "graph_relationship_index_count": len(edges)},
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


def _frontend_source(*names: str) -> str:
    return "\n".join(
        (Path("apps/web/src") / name).read_text(encoding="utf-8")
        for name in names
    )


def test_frontend_ingest_result_uses_rebuild_summary():
    source = _frontend_source("pages/KnowledgePage.tsx", "api/types.ts")
    page_source = _frontend_source("pages/KnowledgePage.tsx")

    for field in (
        "document_count",
        "chunk_count",
        "reused_embedding_count",
        "embedded_chunk_count",
        "embedding_dimension",
        "graph_node_count",
        "graph_edge_count",
    ):
        assert field in source
    assert "knowledge_base_name" not in page_source
    assert "collection" not in page_source.lower()


def test_frontend_empty_docs_directory_status_is_localized():
    source = _frontend_source("page-i18n.ts", "pages/KnowledgePage.tsx")

    assert 'noFiles: "无文件"' in source
    assert 'noFiles: "No files"' in source
    assert "docs_dir_has_files === false ? text.noFiles" in source


def test_frontend_chat_input_default_empty_and_enter_submit():
    source = _frontend_source("pages/ChatPage.tsx")

    assert 'useState("")' in source
    assert 'event.key === "Enter"' in source
    assert "!event.shiftKey && !event.altKey" in source
    assert "event.preventDefault()" in source
    assert "void submit()" in source


def test_frontend_default_app_settings_have_empty_knowledge_base_dirs():
    source = _frontend_source("App.tsx")

    assert "knowledge_base_dirs: []" in source
    assert 'knowledge_base_dirs: ["docs"]' not in source


def test_frontend_business_scope_defaults_are_general():
    source = _frontend_source("App.tsx")

    assert "scope_min_score: 0" in source
    assert "in_scope_keywords: []" in source
    assert 'out_of_scope_keywords: ["股票", "Stock"]' in source
    assert "General knowledge base assistant for local documents." in source


def test_frontend_knowledge_base_has_incremental_and_full_rebuild_controls():
    source = _frontend_source("pages/KnowledgePage.tsx", "api/client.ts", "page-i18n.ts")

    assert '"/api/ingest/run"' in source
    assert '"/api/ingest/full"' in source
    assert "window.confirm(text.confirmFull)" in source
    assert "全量重建" in source
    assert 'new Event("crabrag:knowledge-base-rebuilt")' in source


def test_frontend_knowledge_base_empty_dirs_prompt_is_localized():
    source = _frontend_source("pages/KnowledgePage.tsx", "page-i18n.ts")

    assert "请先在设置页面的“知识库读取目录（每行一个）”配置文件目录" in source
    assert "Configure file directories in Settings > Knowledge base directories (one per line)" in source
    assert "docsDirs.length" in source
    assert r"D:\\cd\\docs" not in source


def test_frontend_hides_knowledge_base_name_and_collection_controls():
    source = _frontend_source(
        "page-i18n.ts",
        "pages/KnowledgePage.tsx",
        "pages/SettingsPage.tsx",
    )

    assert "知识库集合" not in source
    assert "Knowledge base collection" not in source
    assert "knowledge_base_name" not in source


def test_frontend_chat_category_list_starts_empty_and_refreshes_after_rebuild():
    source = _frontend_source("pages/ChatPage.tsx")

    assert "useState<CategoriesResponse | null>(null)" in source
    assert "names.map" in source
    assert 'addEventListener("crabrag:knowledge-base-rebuilt"' in source
    assert 'removeEventListener("crabrag:knowledge-base-rebuilt"' in source


def test_frontend_graph_main_initial_layout_uses_radial_disk_layout():
    source = _frontend_source("pages/GraphPage.tsx")

    assert "function crGraphRadialDiskLayout(" in source
    assert "function crGraphDiskRingCount(" in source
    assert "Math.cos" in source
    assert "Math.sin" in source
    assert 'data-testid="radial-disk-graph"' in source
    assert "staticGraph" not in source
    assert "crGraphClusterLayout" not in source


def test_frontend_settings_uses_new_retrieval_module_terms():
    source = _frontend_source("pages/SettingsPage.tsx")

    for term in ("RAG 检索优化与参数", "多粒度文本索引", "统一重排", "多轮追问重写", "max_context_tokens", "上下文 Token 预算"):
        assert term in source
    for obsolete in ("核心参数调优", "RAG 参数调优", "BM25 权重", "向量权重"):
        assert obsolete not in source


def test_frontend_settings_shows_local_model_download_guidance():
    source = _frontend_source("pages/SettingsPage.tsx", "api/types.ts")

    for field in ("local_model_status", "missing_count", "download_urls", "expected_dir", "missing_files"):
        assert field in source
    for label in (
        "本地模型文件已检测到",
        "Local model files detected",
        "缺失本地模型文件",
        "Missing local model files",
        "存放目录：",
        "Save to:",
        "下载地址（ModelScope）",
        "Download (Hugging Face)",
    ):
        assert label in source


def test_frontend_english_language_covers_async_static_labels():
    source = _frontend_source("page-i18n.ts", "pages/EvaluationPage.tsx", "pages/GraphPage.tsx", "pages/SettingsPage.tsx")

    expected_pairs = {
        "暂无知识图谱": "No knowledge graph",
        "知识库动态图谱": "Dynamic knowledge graph",
        "当前索引代": "Active generation",
        "上一索引代": "Previous generation",
        "评测进度": "Evaluation progress",
        "来源命中率": "Source hit rate",
        "图谱路径覆盖率": "Graph path coverage",
        "多轮追问重写": "Multi-turn follow-up rewrite",
    }
    for chinese, english in expected_pairs.items():
        assert chinese in source
        assert english in source


def test_frontend_evaluation_omits_cost_and_duration_warning():
    source = _frontend_source("pages/EvaluationPage.tsx", "page-i18n.ts", "App.test.tsx")

    assert "300-500K" not in source
    assert "60分钟左右" not in source
    assert "60 minutes" not in source


def test_frontend_evaluation_english_language_covers_dynamic_results():
    source = _frontend_source("pages/EvaluationPage.tsx", "page-i18n.ts", "api/client.ts")

    for chinese, english in {
        "RAG 评测对比": "RAG Evaluation",
        "运行评测": "Run evaluation",
        "暂无评测记录": "No evaluation runs yet",
        "Profile 指标": "Profile metrics",
        "评测用例": "Cases",
        "质量分": "Quality score",
        "建议": "Recommendation",
        "错误": "Error",
        "答复": "Answer",
    }.items():
        assert chinese in source
        assert english in source
    assert '"/api/evaluations/active"' in source
    assert '"/api/evaluations/run"' in source
    assert "loadProgress: getEvaluationProgress" in source
    assert "loadCompleted: (runId) => getEvaluation(runId)" in source
