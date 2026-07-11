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
  const governed = (json = false) => ragHeaders(config, json);
  const body = async (request: { json: () => Promise<unknown> }) => JSON.stringify(await request.json());

  app.post("/chat", async (c) => proxyJson(fetcher, api("/chat"), {
    method: "POST", headers: governed(true), body: await body(c.req),
  }));
  app.get("/categories", async () => proxyJson(fetcher, api("/categories"), { headers: governed() }));

  app.get("/config", async (c) => c.json(await readAppConfig(projectRoot)));
  app.put("/config", async (c) => proxyJson(fetcher, api("/config"), {
    method: "PUT", headers: { "content-type": "application/json" }, body: await body(c.req),
  }));

  app.post("/evaluations/run", async () => proxyJson(fetcher, api("/evaluations/run"), { method: "POST", headers: governed() }));
  app.get("/evaluations", async () => proxyJson(fetcher, api("/evaluations"), { headers: governed() }));
  app.get("/evaluations/active", async () => proxyJson(fetcher, api("/evaluations/active"), { headers: governed() }));
  app.get("/evaluations/:runId/progress", async (c) => proxyJson(fetcher, api(`/evaluations/${encodeURIComponent(c.req.param("runId"))}/progress`), { headers: governed() }));
  app.get("/evaluations/:runId", async (c) => proxyJson(fetcher, api(`/evaluations/${encodeURIComponent(c.req.param("runId"))}`), { headers: governed() }));

  app.get("/graph", async () => proxyJson(fetcher, api("/graph"), { headers: governed() }));
  app.post("/graph/subgraph", async (c) => proxyJson(fetcher, api("/graph/subgraph"), {
    method: "POST", headers: governed(true), body: await body(c.req),
  }));
  app.get("/graph/schema", async () => proxyJson(fetcher, api("/graph/schema"), { headers: governed() }));
  app.get("/graph/schema/suggestion", async () => proxyJson(fetcher, api("/graph/schema/suggestion"), { headers: governed() }));
  app.put("/graph/schema", async (c) => proxyJson(fetcher, api("/graph/schema"), {
    method: "PUT", headers: governed(true), body: await body(c.req),
  }));

  app.get("/health", async (c) => {
    try {
      const response = await fetcher(api("/health"));
      return c.json({ ...(await response.json() as object), web: "ok" });
    } catch {
      return c.json({ web: "ok", rag_service: "unavailable", docs_dir_exists: false, chroma: "unknown", llm_api: "unknown" });
    }
  });

  app.get("/index/status", async () => proxyJson(fetcher, api("/index/status"), { headers: governed() }));
  app.post("/index/rollback", async () => proxyJson(fetcher, api("/index/rollback"), { method: "POST", headers: governed() }));

  app.post("/ingest", async () => proxyJson(fetcher, api("/ingest"), { method: "POST", headers: governed() }));
  app.post("/ingest/run", async () => proxyJson(fetcher, api("/ingest/run"), { method: "POST", headers: governed() }));
  app.post("/ingest/full", async () => proxyJson(fetcher, api("/ingest/full"), { method: "POST", headers: governed() }));
  app.get("/ingest/active", async () => proxyJson(fetcher, api("/ingest/active"), { headers: governed() }));
  app.get("/ingest/:runId/progress", async (c) => proxyJson(fetcher, api(`/ingest/${encodeURIComponent(c.req.param("runId"))}/progress`), { headers: governed() }));
  app.get("/ingest/:runId", async (c) => proxyJson(fetcher, api(`/ingest/${encodeURIComponent(c.req.param("runId"))}`), { headers: governed() }));

  app.get("/logs", async (c) => proxyJson(fetcher, `${api("/logs")}${new URL(c.req.url).search}`, { headers: governed() }));

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
