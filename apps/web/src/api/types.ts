export type UiLanguage = "en" | "zh";
export type UiTheme = "red_white" | "blue_white" | "classic_green";

export interface AppSettings {
  system_name: string;
  knowledge_base_name: string;
  ui_theme: UiTheme;
  ui_language: UiLanguage;
  sidebar_image_url: string;
  knowledge_base_dirs: string[];
  common_questions: string[];
  business_scope_description: string;
  in_scope_keywords: string[];
  out_of_scope_keywords: string[];
  scope_min_score: number;
  out_of_scope_response: string;
  no_match_response: string;
}

export interface LocalModelStatusItem {
  key: "llm" | "embedding" | "rerank";
  name: string;
  present: boolean;
  expected_dir: string;
  required_files: string[];
  missing_files: string[];
  download_urls: { zh: string; en: string };
}

export interface ModelSettings {
  use_local_models: boolean;
  api_key_set: boolean;
  api_key_source: "settings" | "env" | "missing";
  api_key_hint: string;
  base_url: string;
  openai_compatible: boolean;
  chat_model: string;
  embedding_api_key_set: boolean;
  embedding_api_key_source: "settings" | "env" | "missing";
  embedding_api_key_hint: string;
  embedding_provider: "api" | "local_onnx";
  embedding_base_url: string;
  embedding_openai_compatible: boolean;
  embedding_model: string;
  embedding_onnx_model_file: OnnxModelFile;
  rerank_api_key_set: boolean;
  rerank_api_key_source: "settings" | "env" | "missing";
  rerank_api_key_hint: string;
  rerank_base_url: string;
  rerank_onnx_model_file: OnnxModelFile;
  local_model_status: {
    base_dir: string;
    missing_count: number;
    models: LocalModelStatusItem[];
  };
}

export type OnnxModelFile = "model.onnx" | "model_fp16.onnx" | "model_int8.onnx" | "model_q4.onnx";

export interface ModelSettingsUpdate {
  use_local_models: boolean;
  api_key?: string | null;
  clear_api_key?: boolean;
  base_url: string;
  openai_compatible: boolean;
  chat_model: string;
  embedding_provider: "api" | "local_onnx";
  embedding_api_key?: string | null;
  clear_embedding_api_key?: boolean;
  embedding_base_url: string;
  embedding_openai_compatible: boolean;
  embedding_model: string;
  embedding_onnx_model_file: OnnxModelFile;
  rerank_api_key?: string | null;
  clear_rerank_api_key?: boolean;
  rerank_base_url?: string | null;
  rerank_onnx_model_file: OnnxModelFile;
}

export interface RagSettings {
  multi_vector_enabled: boolean;
  hybrid_bm25_enabled: boolean;
  query_expansion_enabled: boolean;
  rerank_enabled: boolean;
  context_rewrite_enabled: boolean;
  dynamic_top_k_enabled: boolean;
  rag_param_tuning_enabled: boolean;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  min_score: number;
  vector_candidate_k: number;
  max_context_tokens: number;
  bm25_weight: number;
  vector_weight: number;
  rerank_provider: "api" | "local_onnx";
  rerank_model: string;
}

export interface HealthResponse {
  web?: string;
  rag_service?: string;
  docs_dir_exists?: boolean;
  chroma?: string;
  llm_api?: string;
  [key: string]: unknown;
}

export interface IngestProgress {
  run_id: string;
  status: "idle" | "queued" | "running" | "completed" | "failed";
  percent?: number;
  current_step?: string;
  message?: string;
  error?: string | null;
  duration_label?: string;
  [key: string]: unknown;
}

export interface IngestActiveResponse {
  active: IngestProgress | null;
  last_success: IngestProgress | null;
}

export interface IngestResult {
  run_id?: string;
  generation_id?: string;
  document_count?: number;
  chunk_count?: number;
  reused_embedding_count?: number;
  embedded_chunk_count?: number;
  graph_node_count?: number;
  graph_edge_count?: number;
  embedding_dimension?: number;
  [key: string]: unknown;
}

export interface GraphSchema {
  status?: string;
  entity_types?: string[];
  node_fields?: Array<{ key?: string; label?: string; type?: string }>;
  edge_fields?: Array<{ key?: string; label?: string; type?: string }>;
  [key: string]: unknown;
}

export interface GenerationManifest {
  stats?: Record<string, number>;
  warnings?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface IndexStatus {
  active_generation?: string | null;
  previous_generation?: string | null;
  active?: GenerationManifest | null;
  previous?: GenerationManifest | null;
  can_rollback?: boolean;
  cache?: Record<string, unknown>;
  scheduler?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface GraphNode {
  id: string;
  label: string;
  properties?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  properties?: Record<string, unknown>;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: Record<string, unknown>;
  schema?: GraphSchema;
}

export interface LogItem {
  id?: string;
  session_id?: string;
  time?: string;
  created_at?: string;
  intent?: string;
  category?: string;
  question?: string;
  answer?: string;
  retrieval_mode?: string;
  sources?: string[];
  entities?: string[];
  error?: string | null;
  references?: ChatReference[];
  relation_paths?: RelationPath[];
  trace?: TraceItem[];
  [key: string]: unknown;
}

export interface EvaluationProgress {
  run_id?: string;
  status: "idle" | "queued" | "running" | "completed" | "failed";
  percent?: number;
  current_profile?: string;
  current_question?: string;
  message?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface EvaluationRunSummary {
  run_id: string;
  created_at?: string;
  profile_count?: number;
  question_count?: number;
  summary?: Record<string, unknown>;
}

export interface EvaluationRun extends EvaluationRunSummary {
  question_generation?: Record<string, unknown>;
  profiles?: Array<Record<string, unknown>>;
}

export interface CategoriesResponse {
  categories?: string[];
  items?: Array<{ name?: string; [key: string]: unknown }>;
}

export interface ChatRequest {
  question: string;
  session_id?: string;
}

export interface ChatReference {
  source_file?: string;
  source?: string;
  content?: string;
  text?: string;
  score?: number;
  document_id?: string;
  chunk_id?: string;
  parent_chunk_id?: string;
  [key: string]: unknown;
}

export interface RelationPath {
  path?: unknown;
  relation_path?: unknown;
  source_file?: string;
  [key: string]: unknown;
}

export interface TraceItem {
  node?: string;
  detail?: string;
  message?: string;
  [key: string]: unknown;
}

export interface ChatResponse {
  session_id: string;
  index_generation?: string;
  intent?: string;
  question_type?: string;
  retrieval_mode?: string;
  entities?: string[];
  answer: string;
  references?: ChatReference[];
  relation_paths?: RelationPath[];
  trace?: TraceItem[];
  error?: string | null;
}
