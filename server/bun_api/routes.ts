import { Hono } from "hono";

import { readAppConfig } from "./app-config";
import { type GatewayConfig, ragHeaders } from "./config";
import { type Fetcher, proxyBinary, proxyJson } from "./proxy";

export interface RouteDependencies {
  config: GatewayConfig;
  fetcher: Fetcher;
  projectRoot: string;
}

export function createApiRoutes({ config, fetcher, projectRoot }: RouteDependencies): Hono {
  const app = new Hono();
  const api = (path: string) => `${config.ragBaseUrl}/api${path}`;
  const governed = (request: Request, json = false) => (
    ragHeaders(config, json, request.headers.get("authorization") ?? undefined)
  );
  const body = async (request: { json: () => Promise<unknown> }) => JSON.stringify(await request.json());

  app.post("/chat", async (c) => proxyJson(fetcher, api("/chat"), {
    method: "POST", headers: governed(c.req.raw, true), body: await body(c.req),
  }));
  app.get("/categories", async (c) => proxyJson(fetcher, api("/categories"), { headers: governed(c.req.raw) }));

  app.get("/config", async (c) => c.json(await readAppConfig(projectRoot)));
  app.put("/config", async (c) => proxyJson(fetcher, api("/config"), {
    method: "PUT", headers: { "content-type": "application/json" }, body: await body(c.req),
  }));

  app.post("/evaluations/run", async (c) => proxyJson(fetcher, api("/evaluations/run"), { method: "POST", headers: governed(c.req.raw) }));
  app.get("/evaluations", async (c) => proxyJson(fetcher, api("/evaluations"), { headers: governed(c.req.raw) }));
  app.get("/evaluations/active", async (c) => proxyJson(fetcher, api("/evaluations/active"), { headers: governed(c.req.raw) }));
  app.get("/evaluations/:runId/progress", async (c) => proxyJson(fetcher, api(`/evaluations/${encodeURIComponent(c.req.param("runId"))}/progress`), { headers: governed(c.req.raw) }));
  app.get("/evaluations/:runId", async (c) => proxyJson(fetcher, api(`/evaluations/${encodeURIComponent(c.req.param("runId"))}`), { headers: governed(c.req.raw) }));

  app.get("/graph", async (c) => proxyJson(fetcher, api("/graph"), { headers: governed(c.req.raw) }));
  app.post("/graph/subgraph", async (c) => proxyJson(fetcher, api("/graph/subgraph"), {
    method: "POST", headers: governed(c.req.raw, true), body: await body(c.req),
  }));
  app.get("/graph/schema", async (c) => proxyJson(fetcher, api("/graph/schema"), { headers: governed(c.req.raw) }));
  app.get("/graph/schema/suggestion", async (c) => proxyJson(fetcher, api("/graph/schema/suggestion"), { headers: governed(c.req.raw) }));
  app.put("/graph/schema", async (c) => proxyJson(fetcher, api("/graph/schema"), {
    method: "PUT", headers: governed(c.req.raw, true), body: await body(c.req),
  }));

  app.get("/health", async (c) => {
    try {
      const response = await fetcher(api("/health"));
      return c.json({ ...(await response.json() as object), web: "ok" });
    } catch {
      return c.json({ web: "ok", rag_service: "unavailable", docs_dir_exists: false, chroma: "unknown", llm_api: "unknown" });
    }
  });

  app.get("/index/status", async (c) => proxyJson(fetcher, api("/index/status"), { headers: governed(c.req.raw) }));
  app.post("/index/rollback", async (c) => proxyJson(fetcher, api("/index/rollback"), { method: "POST", headers: governed(c.req.raw) }));

  app.post("/ingest", async (c) => proxyJson(fetcher, api("/ingest"), { method: "POST", headers: governed(c.req.raw) }));
  app.post("/ingest/run", async (c) => proxyJson(fetcher, api("/ingest/run"), { method: "POST", headers: governed(c.req.raw) }));
  app.post("/ingest/full", async (c) => proxyJson(fetcher, api("/ingest/full"), { method: "POST", headers: governed(c.req.raw) }));
  app.get("/ingest/active", async (c) => proxyJson(fetcher, api("/ingest/active"), { headers: governed(c.req.raw) }));
  app.get("/ingest/:runId/progress", async (c) => proxyJson(fetcher, api(`/ingest/${encodeURIComponent(c.req.param("runId"))}/progress`), { headers: governed(c.req.raw) }));
  app.get("/ingest/:runId", async (c) => proxyJson(fetcher, api(`/ingest/${encodeURIComponent(c.req.param("runId"))}`), { headers: governed(c.req.raw) }));

  app.get("/logs", async (c) => proxyJson(fetcher, `${api("/logs")}${new URL(c.req.url).search}`, { headers: governed(c.req.raw) }));

  app.get("/settings", async () => proxyJson(fetcher, api("/settings")));
  app.put("/settings", async (c) => proxyJson(fetcher, api("/settings"), {
    method: "PUT", headers: { "content-type": "application/json" }, body: await body(c.req),
  }));
  app.get("/app-settings", async () => proxyJson(fetcher, api("/app-settings")));
  app.put("/app-settings", async (c) => proxyJson(fetcher, api("/app-settings"), {
    method: "PUT", headers: { "content-type": "application/json" }, body: await body(c.req),
  }));
  app.put("/app-settings/sidebar-image", async (c) => proxyJson(fetcher, api("/app-settings/sidebar-image"), {
    method: "PUT", headers: { "content-type": "application/json" }, body: await body(c.req),
  }));
  app.get("/app-assets/sidebar-image", async () => proxyBinary(fetcher, api("/app-assets/sidebar-image")));
  app.get("/model-settings", async () => proxyJson(fetcher, api("/model-settings")));
  app.put("/model-settings", async (c) => proxyJson(fetcher, api("/model-settings"), {
    method: "PUT", headers: { "content-type": "application/json" }, body: await body(c.req),
  }));

  return app;
}
