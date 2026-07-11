import type {
  AppSettings,
  CategoriesResponse,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  IngestActiveResponse,
  IngestProgress,
  IngestResult,
  GraphSchema,
  IndexStatus,
  GraphResponse,
  LogItem,
  EvaluationProgress,
  EvaluationRun,
  EvaluationRunSummary,
  ModelSettings,
  ModelSettingsUpdate,
  RagSettings,
} from "./types";

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim();
    try {
      const payload = (await response.json()) as { detail?: string };
      message = payload.detail || message;
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function jsonInit(method: "POST" | "PUT", payload: unknown): RequestInit {
  return {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  };
}

export const getHealth = () => apiJson<HealthResponse>("/api/health");
export const getCategories = () => apiJson<CategoriesResponse>("/api/categories");
export const getAppSettings = () => apiJson<AppSettings>("/api/app-settings");
export const putAppSettings = (settings: AppSettings) =>
  apiJson<AppSettings>("/api/app-settings", jsonInit("PUT", settings));
export const getModelSettings = () => apiJson<ModelSettings>("/api/model-settings");
export const putModelSettings = (settings: ModelSettingsUpdate) =>
  apiJson<ModelSettings>("/api/model-settings", jsonInit("PUT", settings));
export const getRagSettings = () => apiJson<RagSettings>("/api/settings");
export const putRagSettings = (settings: RagSettings) =>
  apiJson<RagSettings>("/api/settings", jsonInit("PUT", settings));
export const postChat = (request: ChatRequest) =>
  apiJson<ChatResponse>("/api/chat", jsonInit("POST", request));

export const getActiveIngest = () => apiJson<IngestActiveResponse>("/api/ingest/active");
export const postIngest = (full = false) => apiJson<IngestProgress>(full ? "/api/ingest/full" : "/api/ingest/run", { method: "POST" });
export const getIngestProgress = (runId: string) => apiJson<IngestProgress>(`/api/ingest/${encodeURIComponent(runId)}/progress`);
export const getIngestResult = (runId: string) => apiJson<IngestResult>(`/api/ingest/${encodeURIComponent(runId)}`);
export const getGraphSchemaSuggestion = () => apiJson<GraphSchema>("/api/graph/schema/suggestion");
export const putGraphSchema = (schema: GraphSchema) => apiJson<GraphSchema>("/api/graph/schema", jsonInit("PUT", schema));
export const getIndexStatus = () => apiJson<IndexStatus>("/api/index/status");
export const postIndexRollback = () => apiJson<IndexStatus>("/api/index/rollback", { method: "POST" });
export const getGraph = () => apiJson<GraphResponse>("/api/graph");
export const getLogs = (category = "") => apiJson<{ items: LogItem[] }>(`/api/logs${category ? `?intent=${encodeURIComponent(category)}` : ""}`);
export const getEvaluations = () => apiJson<{ items: EvaluationRunSummary[] }>("/api/evaluations");
export const getActiveEvaluation = () => apiJson<EvaluationProgress>("/api/evaluations/active");
export const postEvaluationRun = () => apiJson<EvaluationProgress>("/api/evaluations/run", { method: "POST" });
export const getEvaluationProgress = (runId: string) => apiJson<EvaluationProgress>(`/api/evaluations/${encodeURIComponent(runId)}/progress`);
export const getEvaluation = (runId: string) => apiJson<EvaluationRun>(`/api/evaluations/${encodeURIComponent(runId)}`);

export function uploadSidebarImage(filename: string, contentType: string, dataBase64: string) {
  return apiJson<AppSettings>(
    "/api/app-settings/sidebar-image",
    jsonInit("PUT", { filename, content_type: contentType, data_base64: dataBase64 }),
  );
}
