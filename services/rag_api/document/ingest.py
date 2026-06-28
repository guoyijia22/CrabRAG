from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.rag_api.config import get_settings
from services.rag_api.document.categories import save_kb_categories
from services.rag_api.document.kb_naming import ensure_knowledge_base_name
from services.rag_api.document.loader import load_documents
from services.rag_api.document.multi_vector import expand_multi_vector_chunks
from services.rag_api.document.splitter import split_documents
from services.rag_api.graph.kb_graph_builder import build_and_save_kb_graph
from services.rag_api.graph.schema_config import generate_graph_schema_suggestion
from services.rag_api.graph.graph_vector_store import index_graph_vectors
from services.rag_api.rag_settings import load_rag_settings
from services.rag_api.vector.chroma_store import add_chunks, embedding_batch_count


ProgressCallback = Callable[[dict[str, Any]], None]


def ingest_knowledge_base(progress_callback: ProgressCallback | None = None) -> dict:
    total_units = 7

    def progress(completed_units: int, current_step: str, message: str, **extra: Any) -> None:
        if progress_callback:
            percent = int(completed_units / total_units * 100) if total_units else 0
            progress_callback(
                {
                    "completed_units": completed_units,
                    "total_units": total_units,
                    "percent": percent,
                    "current_step": current_step,
                    "message": message,
                    **extra,
                }
            )

    settings = get_settings()
    rag_settings = load_rag_settings()
    docs_dirs = [path.resolve() for path in getattr(settings, "docs_dirs", [settings.docs_dir])]
    progress(1, "扫描文档", "正在扫描并读取知识库目录")
    documents = load_documents(docs_dirs)
    progress(2, "读取解析", f"已读取 {len(documents)} 份文档，正在准备切片")
    chunk_size = rag_settings.chunk_size
    chunk_overlap = rag_settings.chunk_overlap
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    progress(3, "清洗切片", f"已生成 {len(chunks)} 个基础片段")
    chunks = expand_multi_vector_chunks(chunks, rag_settings)
    vector_batch_units = max(1, embedding_batch_count(len(chunks)))
    total_units = 7 + vector_batch_units
    progress(4, "多粒度处理", f"待入库片段数量 {len(chunks)}")
    vector_base_units = 4

    def vector_progress(update: dict[str, Any]) -> None:
        detail_current = int(update.get("detail_current") or 0)
        completed_units = min(vector_base_units + detail_current, vector_base_units + vector_batch_units)
        progress(
            completed_units,
            update.get("current_step", "向量化"),
            update.get("message", "正在向量化知识库片段"),
            detail_current=update.get("detail_current"),
            detail_total=update.get("detail_total"),
        )

    chunk_count = add_chunks(chunks, progress_callback=vector_progress)
    progress(4 + vector_batch_units, "写入向量库", f"已写入 {chunk_count} 个 Chroma 向量片段")
    category_payload = save_kb_categories(documents, chunks)
    progress(5 + vector_batch_units, "生成分类", f"已生成 {len(category_payload['items'])} 个知识库分类")
    kb_graph = build_and_save_kb_graph(category_payload, documents, chunks)
    graph_index_stats = index_graph_vectors(kb_graph.get("nodes", []), kb_graph.get("edges", []))
    progress(
        6 + vector_batch_units,
        "生成图谱索引",
        f"已写入 {graph_index_stats['graph_entity_index_count']} 个实体向量、{graph_index_stats['graph_relationship_index_count']} 个关系向量",
    )
    knowledge_base_name, knowledge_base_name_source = ensure_knowledge_base_name(category_payload, documents, chunk_count)
    graph_schema_suggestion = generate_graph_schema_suggestion(category_payload, documents, chunks)
    progress(7 + vector_batch_units, "生成知识图谱", "知识库重建完成")
    return {
        "status": "success",
        "kb_dir": str(docs_dirs[0]),
        "kb_dirs": [str(path) for path in docs_dirs],
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "collection": settings.collection_name,
        "multi_vector_enabled": rag_settings.multi_vector_enabled,
        "categories": category_payload["items"],
        "category_count": len(category_payload["items"]),
        "knowledge_base_name": knowledge_base_name,
        "knowledge_base_name_source": knowledge_base_name_source,
        "graph_node_count": len(kb_graph.get("nodes", [])),
        "graph_edge_count": len(kb_graph.get("edges", [])),
        **graph_index_stats,
        "graph_schema_suggestion": graph_schema_suggestion,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(ingest_knowledge_base(), ensure_ascii=False, indent=2))
