import { extname, join, resolve, sep } from "node:path";

import { readAppConfig } from "./app-config";

export function resolveStaticPath(pathname: string, webDistDir: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  if (decoded.includes("\0") || decoded.includes("\\")) return null;
  const segments = decoded.split("/").filter(Boolean);
  if (segments.some((segment) => segment === "." || segment === "..")) return null;
  const root = resolve(webDistDir);
  const candidate = resolve(root, ...segments);
  return candidate === root || candidate.startsWith(`${root}${sep}`) ? candidate : null;
}

export async function serveReactDist(pathname: string, webDistDir: string, projectRoot: string): Promise<Response> {
  const requestedPath = resolveStaticPath(pathname === "/" ? "/index.html" : pathname, webDistDir);
  if (!requestedPath) return new Response("Not found", { status: 404 });

  let filePath = requestedPath;
  let file = Bun.file(filePath);
  if (!await file.exists() && !extname(filePath)) {
    filePath = join(webDistDir, "index.html");
    file = Bun.file(filePath);
  }
  if (await file.exists()) return new Response(file, { headers: { "content-type": contentType(filePath) } });
  if (await Bun.file(join(webDistDir, "index.html")).exists()) return new Response("Not found", { status: 404 });

  const { system_name, ui_language } = await readAppConfig(projectRoot);
  const lang = ui_language === "zh" ? "zh-CN" : "en";
  const message = ui_language === "zh"
    ? "未找到前端构建产物，请先运行 bun run build，然后刷新本页。"
    : "Frontend build output was not found. Run bun run build, then refresh this page.";
  return new Response(
    `<!doctype html><html lang="${lang}"><head><meta charset="utf-8"><title>${escapeHtml(system_name)}</title></head><body><h1>${escapeHtml(system_name)}</h1><p>${escapeHtml(message)}</p></body></html>`,
    { status: 503, headers: { "content-type": "text/html; charset=utf-8" } },
  );
}

function escapeHtml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function contentType(filePath: string): string {
  const extension = extname(filePath).toLowerCase();
  return ({
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
  } as Record<string, string>)[extension] ?? "application/octet-stream";
}
