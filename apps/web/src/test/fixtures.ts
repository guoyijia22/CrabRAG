export const appSettings = {
  system_name: "CrabRAG",
  knowledge_base_name: "不得显示的知识库名",
  ui_theme: "red_white" as const,
  ui_language: "zh" as const,
  sidebar_image_url: "",
  knowledge_base_dirs: ["D:/docs"],
  common_questions: ["常用问题示例"],
  business_scope_description: "本地知识问答",
  in_scope_keywords: ["规范"],
  out_of_scope_keywords: ["股票"],
  scope_min_score: 0.2,
  out_of_scope_response: "超出范围",
  no_match_response: "无依据",
};

export const modelSettings = {
  use_local_models: false,
  api_key_set: false,
  api_key_source: "missing" as const,
  api_key_hint: "",
  base_url: "https://api.example/v1",
  openai_compatible: true,
  chat_model: "Qwen/Qwen3.5-9B",
  embedding_api_key_set: false,
  embedding_api_key_source: "missing" as const,
  embedding_api_key_hint: "",
  embedding_provider: "api" as const,
  embedding_base_url: "https://api.example/v1",
  embedding_openai_compatible: true,
  embedding_model: "BAAI/bge-m3",
  embedding_onnx_model_file: "model.onnx" as const,
  rerank_api_key_set: false,
  rerank_api_key_source: "missing" as const,
  rerank_api_key_hint: "",
  rerank_base_url: "https://api.example/v1",
  rerank_onnx_model_file: "model.onnx" as const,
  local_model_status: { base_dir: "runtime/models", missing_count: 3, models: [] },
};

export const ragSettings = {
  multi_vector_enabled: false,
  hybrid_bm25_enabled: false,
  query_expansion_enabled: false,
  rerank_enabled: false,
  context_rewrite_enabled: false,
  dynamic_top_k_enabled: false,
  parent_context_enabled: false,
  rag_param_tuning_enabled: false,
  chunk_size: 600,
  chunk_overlap: 100,
  top_k: 2,
  min_score: 0.35,
  vector_candidate_k: 8,
  max_context_tokens: 6000,
  bm25_weight: 0.5,
  vector_weight: 0.5,
  rerank_provider: "api" as const,
  rerank_model: "BAAI/bge-reranker-v2-m3",
};

export function mockApi(overrides: Record<string, unknown> = {}) {
  const payloads: Record<string, unknown> = {
    "/api/app-settings": appSettings,
    "/api/model-settings": modelSettings,
    "/api/settings": ragSettings,
    "/api/health": { web: "ok", docs_dir_exists: true, chroma: "ok", llm_api: "ok" },
    "/api/categories": { categories: ["规范"], items: [{ name: "规范" }] },
    ...overrides,
  };
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const value = payloads[url];
    if (value === undefined) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    if (value instanceof Response) return value.clone();
    return new Response(JSON.stringify(value), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
}
