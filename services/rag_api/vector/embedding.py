from services.rag_api.llm.siliconflow_client import embed_texts


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embed_texts(texts)
