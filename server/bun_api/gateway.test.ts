import { afterEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { readAppConfig } from "./app-config";
import { createApp, resolveStaticPath } from "./index";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  while (temporaryDirectories.length > 0) {
    const directory = temporaryDirectories.pop();
    if (directory) await rm(directory, { recursive: true, force: true });
  }
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "crabrag-gateway-"));
  temporaryDirectories.push(directory);
  return directory;
}

const trustedConfig = {
  ragBaseUrl: "http://rag.test",
  internalToken: "server-token",
  subject: "server-user",
  roles: "reviewer",
  groups: "north",
  permissionRevision: "7",
  localAdmin: true,
};

describe("trusted RAG proxy", () => {
  test("uses only server identity when the browser spoofs CrabRAG headers", async () => {
    let forwarded: Headers | undefined;
    const app = createApp({
      config: trustedConfig,
      fetch: async (_input, init) => {
        forwarded = new Headers(init?.headers);
        return Response.json({ ok: true });
      },
      webDistDir: await temporaryDirectory(),
    });

    const response = await app.request("/api/chat", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-crabrag-internal-token": "browser-token",
        "x-crabrag-subject": "attacker",
        "x-crabrag-roles": "admin",
        "x-crabrag-groups": "attackers",
        "x-crabrag-permission-revision": "999",
        "x-crabrag-admin": "false",
      },
      body: JSON.stringify({ question: "hello" }),
    });

    expect(response.status).toBe(200);
    expect(Object.fromEntries(forwarded!.entries())).toEqual({
      "content-type": "application/json",
      "x-crabrag-admin": "true",
      "x-crabrag-groups": "north",
      "x-crabrag-internal-token": "server-token",
      "x-crabrag-permission-revision": "7",
      "x-crabrag-roles": "reviewer",
      "x-crabrag-subject": "server-user",
    });
  });

  test("adds trusted identity to governed route families and JSON content type to JSON bodies", async () => {
    const calls: Array<{ url: string; headers: Headers }> = [];
    const app = createApp({
      config: trustedConfig,
      fetch: async (input, init) => {
        calls.push({ url: String(input), headers: new Headers(init?.headers) });
        return Response.json({ ok: true });
      },
      webDistDir: await temporaryDirectory(),
    });
    const requests: Array<[string, string, unknown?]> = [
      ["POST", "/api/ingest/run"],
      ["GET", "/api/evaluations"],
      ["GET", "/api/logs?limit=5"],
      ["GET", "/api/graph/schema"],
      ["PUT", "/api/graph/schema", { entity_types: [] }],
      ["GET", "/api/index/status"],
      ["POST", "/api/index/rollback"],
    ];

    for (const [method, path, body] of requests) {
      const response = await app.request(path, {
        method,
        headers: body === undefined ? undefined : { "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      expect(response.status).toBe(200);
    }

    expect(calls.map((call) => new URL(call.url).pathname + new URL(call.url).search)).toEqual([
      "/api/ingest/run",
      "/api/evaluations",
      "/api/logs?limit=5",
      "/api/graph/schema",
      "/api/graph/schema",
      "/api/index/status",
      "/api/index/rollback",
    ]);
    for (const call of calls) {
      expect(call.headers.get("x-crabrag-subject")).toBe("server-user");
      expect(call.headers.get("x-crabrag-internal-token")).toBe("server-token");
    }
    expect(calls[4].headers.get("content-type")).toBe("application/json");
  });

  test("preserves backend status and returns JSON content type", async () => {
    const app = createApp({
      config: trustedConfig,
      fetch: async () => Response.json({ error: "conflict" }, { status: 409 }),
      webDistDir: await temporaryDirectory(),
    });

    const response = await app.request("/api/ingest", { method: "POST" });

    expect(response.status).toBe(409);
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(await response.json()).toEqual({ error: "conflict" });
  });

  test("preserves sidebar image bytes, status, and content type", async () => {
    const bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    const app = createApp({
      config: trustedConfig,
      fetch: async () => new Response(bytes, { status: 206, headers: { "content-type": "image/png" } }),
      webDistDir: await temporaryDirectory(),
    });

    const response = await app.request("/api/app-assets/sidebar-image");

    expect(response.status).toBe(206);
    expect(response.headers.get("content-type")).toBe("image/png");
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
  });

  test("does not expand trusted identity headers to settings writes", async () => {
    let forwarded: Headers | undefined;
    const app = createApp({
      config: trustedConfig,
      fetch: async (_input, init) => {
        forwarded = new Headers(init?.headers);
        return Response.json({ ok: true });
      },
      webDistDir: await temporaryDirectory(),
    });

    const response = await app.request("/api/settings", {
      method: "PUT",
      headers: { "content-type": "application/json", "x-crabrag-admin": "true" },
      body: JSON.stringify({ retrieval_top_k: 5 }),
    });

    expect(response.status).toBe(200);
    expect(Object.fromEntries(forwarded!.entries())).toEqual({ "content-type": "application/json" });
  });

  test("returns JSON 404 for unknown API routes", async () => {
    const app = createApp({ config: trustedConfig, webDistDir: await temporaryDirectory() });

    const response = await app.request("/api/not-a-route");
    const apiRoot = await app.request("/api");
    const unknownPost = await app.request("/api/not-a-route", { method: "POST" });

    expect(response.status).toBe(404);
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(apiRoot.status).toBe(404);
    expect(apiRoot.headers.get("content-type")).toContain("application/json");
    expect(unknownPost.status).toBe(404);
    expect(unknownPost.headers.get("content-type")).toContain("application/json");
  });
});

describe("static application", () => {
  test("serves nested assets and falls back to the SPA index", async () => {
    const dist = await temporaryDirectory();
    await mkdir(join(dist, "assets"));
    await writeFile(join(dist, "assets", "main.js"), "asset");
    await writeFile(join(dist, "index.html"), "<main>CrabRAG</main>");
    const app = createApp({ config: trustedConfig, webDistDir: dist });

    const asset = await app.request("/assets/main.js");
    const spa = await app.request("/knowledge-base/documents");

    expect(asset.status).toBe(200);
    expect(await asset.text()).toBe("asset");
    expect(spa.status).toBe(200);
    expect(await spa.text()).toBe("<main>CrabRAG</main>");
  });

  test("returns 503 HTML when the frontend build is missing", async () => {
    const projectRoot = await temporaryDirectory();
    const app = createApp({ config: trustedConfig, projectRoot, webDistDir: join(projectRoot, "missing-dist") });

    const response = await app.request("/");

    expect(response.status).toBe(503);
    expect(response.headers.get("content-type")).toContain("text/html");
    expect(await response.text()).toContain("Frontend build output was not found");
  });

  test("resolves URL segments portably and rejects plain or encoded traversal", async () => {
    const dist = await temporaryDirectory();

    expect(resolveStaticPath("/assets/main.js", dist)).toBe(join(dist, "assets", "main.js"));
    expect(resolveStaticPath("/../secret.txt", dist)).toBeNull();
    expect(resolveStaticPath("/%2e%2e/secret.txt", dist)).toBeNull();
    expect(resolveStaticPath("/%2e%2e%2fsecret.txt", dist)).toBeNull();
    expect(resolveStaticPath("/%ZZ/secret.txt", dist)).toBeNull();
  });
});

describe("application config", () => {
  test("keeps defaults, legacy-name migration, theme, language, and common questions", async () => {
    const root = await temporaryDirectory();
    await mkdir(join(root, "data"));
    await writeFile(
      join(root, "data", "app_settings.json"),
      JSON.stringify({
        system_name: "QueryBasePortableLab 通用基础查询",
        knowledge_base_name: "业务知识库",
        ui_theme: "classic_green",
        ui_language: "zh",
        common_questions: Array.from({ length: 12 }, (_, index) => `问题${index}`),
      }),
    );

    const config = await readAppConfig(root);

    expect(config).toEqual({
      system_name: "CrabRAG",
      knowledge_base_name: "业务知识库",
      ui_theme: "classic_green",
      ui_language: "zh",
      common_questions: Array.from({ length: 10 }, (_, index) => `问题${index}`),
    });
  });
});

describe("source build contract", () => {
  test("bundles the gateway entrypoint from TypeScript source", async () => {
    const outputDirectory = await temporaryDirectory();

    const result = await Bun.build({
      entrypoints: [join(import.meta.dir, "index.ts")],
      outdir: outputDirectory,
      target: "bun",
    });

    expect(result.success).toBeTrue();
    expect(result.logs).toEqual([]);
    const rebuiltBundle = Bun.file(join(outputDirectory, "index.js"));
    expect(await rebuiltBundle.exists()).toBeTrue();
    expect(await rebuiltBundle.text()).toBe(await Bun.file(join(import.meta.dir, "..", "gateway.js")).text());
  });
});
