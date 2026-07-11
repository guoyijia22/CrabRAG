import { join } from "node:path";

import { Hono } from "hono";

import { type GatewayConfig, readGatewayEnvironment } from "./config";
import { type Fetcher } from "./proxy";
import { createApiRoutes } from "./routes";
import { serveReactDist } from "./static";

export { resolveStaticPath } from "./static";

export interface CreateAppOptions {
  config?: GatewayConfig;
  fetch?: Fetcher;
  projectRoot?: string;
  webDistDir?: string;
}

export function createApp(options: CreateAppOptions = {}): Hono {
  const environment = readGatewayEnvironment();
  const config = options.config ?? environment;
  const projectRoot = options.projectRoot ?? environment.projectRoot;
  const webDistDir = options.webDistDir ?? join(projectRoot, "apps", "web", "dist");
  const fetcher = options.fetch ?? fetch;
  const app = new Hono();

  app.route("/api", createApiRoutes({ config, fetcher, projectRoot }));
  app.all("/api", (c) => c.json({ error: "API route not found or gateway route not loaded" }, 404));
  app.all("/api/*", (c) => c.json({ error: "API route not found or gateway route not loaded" }, 404));
  app.get("*", async (c) => {
    const pathname = new URL(c.req.url).pathname;
    if (pathname === "/api" || pathname.startsWith("/api/")) {
      return c.json({ error: "API route not found or gateway route not loaded" }, 404);
    }
    return serveReactDist(pathname, webDistDir, projectRoot);
  });
  return app;
}

if (import.meta.main) {
  const environment = readGatewayEnvironment();
  const port = Number(process.env.PORT ?? 3003);
  const app = createApp({ config: environment, projectRoot: environment.projectRoot });
  Bun.serve({ fetch: app.fetch, hostname: "127.0.0.1", port, idleTimeout: 255 });
  console.log(`Bun API Gateway: http://127.0.0.1:${port}`);
}
