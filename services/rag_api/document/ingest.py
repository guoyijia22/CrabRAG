from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.rag_api.config import get_settings, write_common_questions
from services.rag_api.document.categories import save_kb_categories
from services.rag_api.document.common_questions import generate_common_questions
from services.rag_api.document.kb_naming import ensure_knowledge_base_name
from services.rag_api.document.loader import load_documents
from services.rag_api.document.multi_vector import expand_multi_vector_chunks
from services.rag_api.document.splitter import split_documents
from services.rag_api.graph.schema_config import generate_graph_schema_suggestion
from services.rag_api.rag_settings import load_rag_settings
from services.rag_api.vector.chroma_store import add_chunks


ProgressCallback = Callable[[dict[str, Any]], None]


def ingest_knowledge_base(progress_callback: ProgressCallback | None = None) -> dict:
    total_units = 8

    def progress(completed_units: int, current_step: str, message: str) -> None:
        if progress_callback:
            progress_callback(
                {
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "percent": int(completed_units / total_units * 100),
                    "current_step": current_step,
                    "message": message,
                }
            )

    settings = get_settings()
    rag_settings = load_rag_settings()
    progress(1, "扫描文档", "正在扫描并读取知识库目录")
    documents = load_documents(settings.docs_dir)
    progress(2, "读取解析", f"已读取 {len(documents)} 份文档，正在准备切片")
    chunk_size = rag_settings.chunk_size if rag_settings.rag_param_tuning_enabled else 600
    chunk_overlap = rag_settings.chunk_overlap if rag_settings.rag_param_tuning_enabled else 100
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    progress(3, "清洗切片", f"已生成 {len(chunks)} 个基础片段")
    chunks = expand_multi_vector_chunks(chunks, rag_settings)
    progress(4, "多粒度处理", f"待入库片段数量 {len(chunks)}")
    chunk_count = add_chunks(chunks)
    progress(5, "写入向量库", f"已写入 {chunk_count} 个 Chroma 向量片段")
    category_payload = save_kb_categories(documents, chunks)
    progress(6, "生成分类", f"已生成 {len(category_payload['items'])} 个知识库分类")
    common_questions = write_common_questions(generate_common_questions(category_payload))
    progress(7, "生成常用问题", f"已生成 {len(common_questions)} 个常用问题")
    knowledge_base_name, knowledge_base_name_source = ensure_knowledge_base_name(category_payload, documents, chunk_count)
    graph_schema_suggestion = generate_graph_schema_suggestion(category_payload, documents, chunks)
    progress(8, "生成图谱结构建议", "知识库重建完成")
    return {
        "status": "success",
        "kb_dir": str(settings.docs_dir),
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "collection": settings.collection_name,
        "multi_vector_enabled": rag_settings.multi_vector_enabled,
        "categories": category_payload["items"],
        "category_count": len(category_payload["items"]),
        "knowledge_base_name": knowledge_base_name,
        "knowledge_base_name_source": knowledge_base_name_source,
        "common_questions": common_questions,
        "common_question_count": len(common_questions),
        "graph_schema_suggestion": graph_schema_suggestion,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(ingest_knowledge_base(), ensure_ascii=False, indent=2))
